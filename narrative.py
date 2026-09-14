"""
narrative.py – VLM-Inspired Clinical Narrative Generator (FreeBuff v3.0)
=========================================================================

Generates structured, clinically-coherent narrative reports from the
prediction payload produced by the inference engine.

Two-tier architecture:
  1. Primary path — a lightweight local transformer (Phi-3-mini / similar)
     generates a paragraph-length narrative conditioned on structured
     findings + DICOM metadata + WHO staging.
  2. Fallback path — a deterministic template engine fills the same fields
     when the model backend is unavailable, so the platform degrades
     gracefully (no cascade failures, no hanging).


Responsibilities
-----------------
1. Compose a "clinical snapshot" from the inference payload, DICOM metadata
   and staging payload.
2. Tokenize + run the narrative model (best-effort, with timeouts and
   exception isolation).
3. Sanitize model output (strip conversational filler, enforce the research/
   educational disclaimer, drop hallucinated measurements).
4. Fall back to template-based narrative when the model is unreachable.
5. Return a dict with ``narrative``, ``model_used``, ``confidence_note``,
   ``disclaimer`` and ``sections`` (dict of section → text).

Author: Codebuff AI
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    NARRATIVE_ENABLED,
    NARRATIVE_FALLBACK_TO_TEMPLATE,
    NARRATIVE_INCONCLUSIVE_PHRASE,
    NARRATIVE_LOW_CONFIDENCE_WARN,
    NARRATIVE_MAX_INPUT_TOKEN_BUDGET,
    NARRATIVE_MAX_NEW_TOKENS,
    NARRATIVE_MODEL_BACKEND,
    NARRATIVE_REPHRASER_BACKEND,
    NARRATIVE_REPHRASER_MODEL_NAME,
    NARRATIVE_TEMPERATURE,
    NARRATIVE_TOP_P,
)
from dataset import dicom_meta_to_report_lines
from staging import format_report_text, malignancy_band

logger = logging.getLogger("narrative")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ===========================================================================
# Template-based narrative (deterministic fallback)
# ===========================================================================

_TEMPLATE_SECTIONS = {
    "examination": (
        "Brain MRI examination reviewed across the provided sequences. "
        "The analysis was performed on standardized 144×144 pixel slices "
        "covering T1-weighted contrast-enhanced, T2-weighted and FLAIR "
        "sequences where available."
    ),
    "technique": (
        "The FreeBuff platform processed each sequence through a hybrid "
        "deep-learning pipeline: a multimodal CNN+CBAM and Transformer "
        "dual-branch encoder (EfficientNet or MobileNetV2 backbone, ViT or "
        "Swin transformer branch), fused via a cross-modality attention "
        "transformer into a 128-dimensional deep embedding. A 2048-dimensional "
        "HOG descriptor computed on the native 144×144 grid was PCA-compressed "
        "to 288 dimensions. These were concatenated into a 416-dimensional "
        "hybrid feature vector classified by a calibrated RBF-kernel SVM "
        "(optionally refined by a stacked super-learner ensemble of RBF-SVM, "
        "XGBoost and LightGBM with a regularized logistic-regression meta-classifier)."
    ),
    "findings": (
        "The classifier identified the lesion family as {predicted_class}. "
        "The model confidence for this prediction was {confidence_percent}%. "
        "The full probability distribution across the four classes was: "
        "{probabilities_list}."
    ),
    "staging": (
        "Based on the detected lesion family, the possible WHO (2021, 5th "
        "edition) grade range is {grade_range}. The typical presentation for "
        "this family is {typical_grade}. {malignancy_profile}"
    ),
    "differential": (
        "{differential_text}"
    ),
    "limitations": (
        "Decision-support only. Imaging-based classification is not a substitute "
        "for histopathology and molecular testing (IDH1/2 mutation, 1p/19q "
        "codeletion, MGMT promoter methylation, ATRX). Final WHO grading must "
        "be established by a neuropathologist. The provided sequences may not "
        "represent the entire lesion; multi-sequence, multi-slice evaluation is "
        "recommended for a complete radiological assessment."
    ),
    "recommendation": (
        "{recommendation_text}"
    ),
}


def _build_template_narrative(result: Dict[str, object]) -> Dict[str, object]:
    """Fill the template sections with the structured result."""
    probs = result.get("probabilities", {})
    pred = result.get("predicted_class", "unknown")
    conf = float(result.get("confidence", 0.0))
    staging = result.get("staging", {})

    prob_lines = []
    for cls_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        p = float(probs.get(cls_name, 0.0))
        prob_lines.append(f"{cls_name}: {p*100:.1f}%")
    probabilities_list = "; ".join(prob_lines)

    grade_range = staging.get("grade_range", [])
    grade_range_str = ", ".join(str(g) for g in grade_range) if grade_range else "not applicable"
    typical_grade = staging.get("typical_grade", "not available")
    malignancy = staging.get("malignancy_profile", "not available")

    differential_map = {
        "glioma": (
            "Differential considerations for a suspected glioma include "
            "lymphoma (especially in immunocompetent patients with a "
            "periventricular lesion), metastasis (multiple lesions, "
            "grey-white matter junction), and abscess (restricted diffusion, "
            "ring enhancement with smooth thin walls)."
        ),
        "meningioma": (
            "Differential considerations for a suspected meningioma include "
            "schwannoma (cerebellopontine angle, internal auditory canal), "
            "hemangioblastoma (cerebellar, cyst with mural nodule), "
            "and solitary fibrous tumour (dural-based, but more aggressive)."
        ),
        "pituitary": (
            "Differential considerations for a suspected pituitary lesion "
            "include craniopharyngioma (suprasellar, calcification, "
            "cystic components), Rathke cleft cyst, and parasellar "
            "meningioma."
        ),
        "notumor": (
            "No lesion identified by the classifier in the provided "
            "sequences. Differential considerations depend on the clinical "
            "question; artefacts, flow voids, and normal anatomical variants "
            "can mimic pathology."
        ),
    }
    differential_text = differential_map.get(pred, "No differential provided.")

    recommendation_map = {
        "glioma": (
            "Recommend multi-sequence evaluation including diffusion-weighted "
            "imaging (DWI), perfusion-weighted imaging (PWI / rCBV), and "
            "contrast-enhanced T1-weighted sequences. Stereotactic biopsy or "
            "surgical resection with histopathological and molecular analysis "
            "is indicated for definitive diagnosis and grading."
        ),
        "meningioma": (
            "Recommend follow-up imaging to assess growth rate if the lesion "
            "is small and asymptomatic. For larger or symptomatic lesions, "
            "surgical resection is the primary treatment; Simpson grade of "
            "resection determines recurrence risk. Radiotherapy may be "
            "considered for atypical or anaplastic variants."
        ),
        "pituitary": (
            "Recommend endocrine evaluation (prolactin, IGF-1, cortisol, "
            "ACTH, thyroid function) and visual field testing if the lesion "
            "abuts the optic chiasm. Management depends on the histological "
            "subtype and hormonal activity; options include observation, "
            "pharmacotherapy (e.g. dopamine agonists for prolactinoma), "
            "surgery, or radiotherapy."
        ),
        "notumor": (
            "No actionable lesion identified. Correlate with clinical "
            "presentation and consider additional imaging if symptoms persist "
            "or evolve. Routine follow-up as clinically indicated."
        ),
    }
    recommendation_text = recommendation_map.get(pred, "Clinical correlation recommended.")

    sections = {}
    template = _TEMPLATE_SECTIONS
    sections["examination"] = template["examination"]
    sections["technique"] = template["technique"]
    sections["findings"] = template["findings"].format(
        predicted_class=pred,
        confidence_percent=f"{conf*100:.1f}",
        probabilities_list=probabilities_list,
    )
    sections["staging"] = template["staging"].format(
        grade_range=grade_range_str,
        typical_grade=typical_grade,
        malignancy_profile=malignancy,
    )
    sections["differential"] = template["differential"].format(
        differential_text=differential_text,
    )
    sections["limitations"] = template["limitations"]
    sections["recommendation"] = template["recommendation"].format(
        recommendation_text=recommendation_text,
    )

    narrative = "\n\n".join(
        f"{section.title()}:\n{text}" for section, text in sections.items()
    )

    return {
        "narrative": narrative,
        "model_used": "none (template fallback)",
        "confidence_note": (
            "Low-confidence warning" if conf < NARRATIVE_LOW_CONFIDENCE_WARN
            else "Confidence within acceptable range"
        ),
        "disclaimer": (
            "This report was generated by an AI research platform and is "
            "provided for research and educational purposes only. It must not "
            "replace a formal radiological or pathological report and is not "
            "intended for direct clinical decision-making."
        ),
        "sections": sections,
    }


# ===========================================================================
# Structured clinical snapshot (model input)
# ===========================================================================

def _build_clinical_snapshot(result: Dict[str, object], dicom_meta: Optional[Dict[str, object]] = None) -> str:
    """Build a structured text snapshot for the narrative model input."""
    probs = result.get("probabilities", {})
    pred = result.get("predicted_class", "unknown")
    conf = float(result.get("confidence", 0.0))
    staging = result.get("staging", {})
    modalities = result.get("modalities", [])
    patient_id = result.get("patient_id") or "not provided"
    model_version = result.get("model_version", "?")

    lines = [
        "BRAIN MRI CLINICAL SNAPSHOT",
        "=" * 40,
        f"Patient/Study ID: {patient_id}",
        f"Model version: v{model_version}",
        f"Predicted lesion family: {pred}",
        f"Model confidence: {conf*100:.1f}%",
        "",
        "CONFIDENCE DISTRIBUTION:",
    ]
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        p = float(probs.get(cls, 0.0))
        lines.append(f"  {cls}: {p*100:.1f}%")

    lines.append("")
    lines.append("WHO STAGING REFERENCE:")
    lines.append(f"  Typical presentation: {staging.get('typical_grade', 'N/A')}")
    lines.append(f"  Grade range: {staging.get('grade_range', [])}")
    lines.append(f"  Malignancy profile: {staging.get('malignancy_profile', 'N/A')}")
    lines.append("")

    if dicom_meta:
        lines.append("DICOM METADATA:")
        for line in dicom_meta_to_report_lines(dicom_meta):
            lines.append(f"  {line}")
        lines.append("")

    lines.append("MODALITIES ANALYSED:")
    for m in modalities:
        lines.append(f"  - {m}")
    lines.append("")

    if conf < NARRATIVE_LOW_CONFIDENCE_WARN:
        lines.append(
            f"NOTE: Model confidence ({conf*100:.1f}%) is below the "
            f"{NARRATIVE_LOW_CONFIDENCE_WARN*100:.0f}% threshold. The "
            f"narrative should reflect this uncertainty."
        )
        lines.append("")

    snapshot = "\n".join(lines)
    # Truncate to budget
    if len(snapshot) > NARRATIVE_MAX_INPUT_TOKEN_BUDGET:
        snapshot = snapshot[:NARRATIVE_MAX_INPUT_TOKEN_BUDGET] + "...[truncated]"
    return snapshot


# ===========================================================================
# Model-backed narrative (best-effort)
# ===========================================================================

def _run_narrative_model(snapshot: str) -> Optional[str]:
    """Run the narrative model (best-effort, with isolation).

    Returns None if the model is unavailable or fails.
    """
    if NARRATIVE_MODEL_BACKEND != "transformers":
        return None

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed — narrative model unavailable.")
        return None

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, timeout=30,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            timeout=60,
            torch_dtype="auto",
            device_map="cpu",
        )
    except Exception as exc:
        logger.warning("Failed to load narrative model '%s': %s", model_name, exc)
        return None

    system_prompt = (
        "You are a clinical AI assistant generating a structured radiology "
        "report narrative. Use ONLY the provided structured clinical snapshot. "
        "Do NOT invent lab values, imaging measurements, or diagnoses. Be "
        "clinically accurate, concise, and professional. State clearly when "
        "findings are inconclusive. This is decision-support, not a clinical "
        "diagnosis. Output ONLY the narrative text, no conversational filler."
    )

    user_prompt = (
        f"Structured clinical snapshot:\n\n{snapshot}\n\n"
        f"Generate a structured clinical narrative report with the following "
        f"sections: Examination, Technique, Findings, WHO Staging, "
        f"Differential Considerations, Limitations, Recommendation."
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=NARRATIVE_MAX_INPUT_TOKEN_BUDGET)
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=NARRATIVE_MAX_NEW_TOKENS,
            temperature=NARRATIVE_TEMPERATURE,
            top_p=NARRATIVE_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        elapsed = time.time() - start
        logger.info("Narrative model generated in %.1fs", elapsed)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated portion
        if "Generate a structured" in text:
            text = text.split("Generate a structured", 1)[1]
        # Strip leading/trailing whitespace
        text = text.strip()

        # Sanitize: remove common model artifacts
        text = _sanitize_narrative(text)

        return text

    except Exception as exc:
        logger.warning("Narrative model inference failed: %s", exc)
        return None


def _sanitize_narrative(text: str) -> str:
    """Remove conversational filler and hallucinated measurements."""
    # Remove common model prefixes
    for prefix in [
        "Here is the structured clinical narrative report:",
        "Based on the provided clinical snapshot:",
        "I have generated the following narrative:",
        "Sure, here is the report:",
        "Certainly! Here is the structured narrative:",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove trailing conversational postfixes
    for postfix in [
        "I hope this helps!",
        "Let me know if you have any questions.",
        "Please consult a qualified healthcare professional.",
        "This is not medical advice.",
    ]:
        if text.endswith(postfix):
            text = text[: -len(postfix)].strip()

    # Remove hallucinated measurements (specific numbers that look like
    # imaging measurements not present in the input)
    # Pattern: "measuring X cm" or "X mm lesion" or "X x X cm"
    text = re.sub(
        r'\b\d+(\.\d+)?\s*(cm|mm|ml|ml)\b',
        '[measurement not specified]',
        text,
        flags=re.IGNORECASE,
    )

    # Ensure disclaimer is present
    if "decision-support" not in text.lower() and "not a clinical diagnosis" not in text.lower():
        text += (
            "\n\nNOTE: This is decision-support only and not a clinical "
            "diagnosis. Final diagnosis requires histopathology and molecular "
            "testing by a qualified neuropathologist."
        )

    return text


# ===========================================================================
# Light rephraser (optional polish)
# ===========================================================================

def _rephrase_narrative(narrative: str) -> str:
    """Optionally rephrase the narrative with a lightweight local model."""
    if NARRATIVE_REPHRASER_BACKEND != "transformers":
        return narrative

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return narrative

    model_name = NARRATIVE_REPHRASER_MODEL_NAME

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, timeout=30,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            timeout=60,
            torch_dtype="auto",
            device_map="cpu",
        )
    except Exception as exc:
        logger.debug("Failed to load rephraser model: %s", exc)
        return narrative

    rephraser_prompt = (
        "Rewrite the following clinical narrative to be more concise and "
        "professional while preserving all clinical information. Do NOT add "
        "new information or measurements.\n\n"
        f"{narrative}\n\n"
        "Rewritten narrative:"
    )

    try:
        inputs = tokenizer(rephraser_prompt, return_tensors="pt", truncation=True,
                          max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Rewritten narrative:" in text:
            text = text.split("Rewritten narrative:", 1)[1].strip()
        return text
    except Exception as exc:
        logger.debug("Rephraser failed: %s", exc)
        return narrative


# ===========================================================================
# Main narrative generation entry point
# ===========================================================================

def generate_narrative(
    result: Dict[str, object],
    dicom_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Generate a clinical narrative from the prediction payload.

    Parameters
    ----------
    result : dict
        The prediction payload from FreeBuffInference.predict_uploads().
    dicom_meta : dict, optional
        DICOM metadata dictionary from the uploaded sequences.

    Returns
    -------
    dict with keys:
        - narrative: the full narrative text
        - model_used: which backend was used ("transformers" or "template")
        - confidence_note: note about confidence level
        - disclaimer: research/educational disclaimer
        - sections: dict of section → text
    """
    if not NARRATIVE_ENABLED:
        return _build_template_narrative(result)

    snapshot = _build_clinical_snapshot(result, dicom_meta)

    if NARRATIVE_MODEL_BACKEND == "transformers":
        model_text = _run_narrative_model(snapshot)
        if model_text:
            logger.info("Narrative generated by transformer model.")
            narrative_text = model_text
            model_used = "transformers (Phi-3-mini-4k-instruct)"
        else:
            logger.info("Transformer model unavailable — falling back to template.")
            template_result = _build_template_narrative(result)
            narrative_text = template_result["narrative"]
            model_used = "none (template fallback)"
    else:
        template_result = _build_template_narrative(result)
        narrative_text = template_result["narrative"]
        model_used = "none (template fallback)"

    # Optional rephrasing
    if NARRATIVE_REPHRASER_BACKEND == "transformers":
        try:
            narrative_text = _rephrase_narrative(narrative_text)
            logger.info("Narrative rephrased by local model.")
        except Exception as exc:
            logger.debug("Rephrasing skipped: %s", exc)

    conf = float(result.get("confidence", 0.0))
    confidence_note = (
        "Low-confidence warning — narrative reflects uncertainty."
        if conf < NARRATIVE_LOW_CONFIDENCE_WARN
        else "Confidence within acceptable range."
    )

    # Extract sections from template for structured output
    template_result = _build_template_narrative(result)
    sections = template_result["sections"]

    return {
        "narrative": narrative_text,
        "model_used": model_used,
        "confidence_note": confidence_note,
        "disclaimer": (
            "This report was generated by an AI research platform and is "
            "provided for research and educational purposes only. It must not "
            "replace a formal radiological or pathological report and is not "
            "intended for direct clinical decision-making."
        ),
        "sections": sections,
    }


# ===========================================================================
# Main — smoke test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("narrative.py — self-tests")
    logger.info("=" * 60)

    mock_result = {
        "predicted_class": "glioma",
        "confidence": 0.87,
        "probabilities": {
            "glioma": 0.87,
            "meningioma": 0.08,
            "notumor": 0.03,
            "pituitary": 0.02,
        },
        "staging": {
            "predicted_class": "glioma",
            "confidence": 0.87,
            "probabilities": {
                "glioma": 0.87,
                "meningioma": 0.08,
                "notumor": 0.03,
                "pituitary": 0.02,
            },
            "profile": {
                "grades": [2, 3, 4],
                "typical_grade": "WHO Grade II–IV (diffuse glioma family)",
                "malignancy": "Malignant potential — HIGH.",
                "notes": "Diffuse gliomas infiltrate surrounding parenchyma.",
            },
            "grade_range": [2, 3, 4],
            "typical_grade": "WHO Grade II–IV (diffuse glioma family)",
            "grade_descriptions": [
                "WHO Grade II — low-grade, diffusely infiltrative neoplasm.",
                "WHO Grade III — anaplastic / high-grade malignancy.",
                "WHO Grade IV — highly aggressive (e.g. glioblastoma).",
            ],
            "malignancy_profile": "Malignant potential — HIGH. Grade IV (glioblastoma, IDH-wildtype) is the most common and most aggressive adult primary brain tumour.",
            "clinical_notes": "Diffuse gliomas infiltrate surrounding parenchyma; grading depends on histology plus molecular markers.",
            "disclaimer": "Decision-support only.",
        },
        "modalities": ["t1ce", "t2", "flair"],
        "patient_id": "PAT_001",
        "model_version": 4,
    }

    mock_dicom = {
        "patient_id": "PAT_001",
        "patient_name": "Test^Patient",
        "study_date": "20240115",
        "modality": "MR",
        "protocol_name": "T1w GRE",
        "photometric_interpretation": "MONOCHROME2",
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
        "rows": 144,
        "columns": 144,
    }

    output = generate_narrative(mock_result, mock_dicom)
    assert "narrative" in output
    assert "model_used" in output
    assert "sections" in output
    assert len(output["narrative"]) > 100
    logger.info("✓ Narrative generated (%d chars, model: %s)", len(output["narrative"]), output["model_used"])
    logger.info("✓ Sections: %s", list(output["sections"].keys()))

    # Test template fallback
    template_output = _build_template_narrative(mock_result)
    assert template_output["model_used"] == "none (template fallback)"
    logger.info("✓ Template fallback OK")

    logger.info("=" * 60)
    logger.info("narrative.py self-tests passed ✓")
    logger.info("=" * 60)
