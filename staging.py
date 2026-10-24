"""
staging.py – Automated WHO Grading & Clinical-Staging Reference Engine
======================================================================

Once the SVM classifies a lesion into one of the four tumour families, the
platform maps the result onto the **WHO Classification of Tumours of the
Central Nervous System** grading scale (Grades I–IV) and produces a
malignancy profile.

IMPORTANT — intended use
------------------------
This is a *decision-support* reference table for clinicians and a teaching
aid.  Definite grading always requires histopathology and molecular
markers (IDH, 1p/19q codeletion, MGMT …) — no imaging-only model can
produce a certified WHO grade.  The engine therefore reports the legally
defensible **range of possible grades** for the detected tumour family,
the *typical* grade, and the malignancy profile, and every report carries
the disclaimer.

Grade semantics (WHO CNS 5th ed., 2021, abridged)
-------------------------------------------------
    Grade I   – slow-growing, circumscribed, generally curable by surgery
    Grade II  – low-grade infiltrative neoplasms (e.g. diffuse astrocytoma)
    Grade III – anaplastic / high-grade
    Grade IV  – highly aggressive (e.g. glioblastoma, IDH-wildtype)

Author: Codebuff AI
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WHO grade reference (educational / decision-support only)
# ---------------------------------------------------------------------------
WHO_GRADES: Dict[int, str] = {
    1: "WHO Grade I — slow-growing, typically circumscribed; excellent "
       "prognosis after complete resection (e.g. pilocytic astrocytoma, "
       "most meningiomas).",
    2: "WHO Grade II — low-grade, diffusely infiltrative neoplasm; can "
       "recur or progress to higher grades (e.g. diffuse astrocytoma "
       "IDH-mutant).",
    3: "WHO Grade III — anaplastic / high-grade malignancy with mitotic "
       "activity; aggressive therapy required.",
    4: "WHO Grade IV — highly aggressive (e.g. glioblastoma IDH-wildtype); "
       "rapid growth, microvascular proliferation and necrosis.",
}

# ---------------------------------------------------------------------------
# Per-class clinical profiles
# ---------------------------------------------------------------------------
# Each entry maps the model class onto:
#   grades          – the *possible* WHO grades for that tumour family
#   typical_grade   – the most common presentation
#   malignancy      – short profile string
#   notes           – clinical context / differential guidance
TUMOUR_PROFILES: Dict[str, Dict[str, object]] = {
    "glioma": {
        "grades": [2, 3, 4],
        "typical_grade": "WHO Grade II–IV (diffuse glioma family)",
        "malignancy": ("Malignant potential — HIGH. Grade IV (glioblastoma, "
                       "IDH-wildtype) is the most common and most aggressive "
                       "adult primary brain tumour."),
        "notes": ("Diffuse gliomas infiltrate surrounding parenchyma; grading "
                  "depends on histology plus molecular markers (IDH1/2, "
                  "1p/19q codeletion, ATRX, MGMT promoter). Non-contrast "
                  "enhancement, necrosis and high CBV on perfusion argue for "
                  "higher grade."),
    },
    "meningioma": {
        "grades": [1, 2, 3],
        "typical_grade": "WHO Grade I (most common)",
        "malignancy": ("Malignancy profile — typically LOW (≈ 90 % behave "
                       "benignly); atypical (II) and anaplastic (III) forms "
                       "carry higher recurrence risk."),
        "notes": ("Extra-axial, dural-based lesion. Most are WHO grade I and "
                  "curable by surgery. Imaging signs suggesting grade II/III: "
                  "brain oedema, mushrooming, bone invasion, high ADC "
                  "restriction or high perfusion."),
    },
    "pituitary": {
        "grades": [1, 2],
        "typical_grade": "WHO Grade I pituitary neuroendocrine tumour (PitNET)",
        "malignancy": ("Malignancy profile — LOW. Pituitary adenomas / PitNETs "
                       "are usually benign; atypical (grade II) and metastatic "
                       "carcinoma are exceptional."),
        "notes": ("Sellar/suprasellar mass. Clinical behaviour depends on "
                  "hormonal secretion (prolactinoma, GH, ACTH ...) and invasion "
                  "of the cavernous sinus (Knosp grade) rather than histologic "
                  "grade alone."),
    },
    "notumor": {
        "grades": [],
        "typical_grade": "No tumour — WHO grading not applicable",
        "malignancy": "None detected in the analysed slice(s).",
        "notes": ("No lesion identified by the classifier in the provided "
                  "sequences. This is a screening output — correlate with the "
                  "full radiological study and clinical picture."),
    },
}

# Fallback used when a custom class (e.g. a BraTS-derived class) appears.
_UNKNOWN_PROFILE: Dict[str, object] = {
    "grades": [],
    "typical_grade": "WHO grading not mapped for this class",
    "malignancy": "Not available — custom class outside the platform reference set.",
    "notes": "No staging profile defined. Consult neuro-oncology guidelines.",
}


def _profile_for(class_name: str) -> Dict[str, object]:
    if class_name in TUMOUR_PROFILES:
        return TUMOUR_PROFILES[class_name]
    # The profile keys are lowercase and unpunctuated ("notumor"), but the
    # canonical labels in config.CLASSES are "Glioma", "No_Tumor", ... . Without
    # normalising here, any caller passing a CLASSES name silently received the
    # empty "unknown" profile instead of real staging information.
    normalised = class_name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if normalised in TUMOUR_PROFILES:
        return TUMOUR_PROFILES[normalised]
    logger.warning("No staging profile for class '%s'.", class_name)
    return dict(_UNKNOWN_PROFILE)


# ===========================================================================
# Public API
# ===========================================================================

def grade_summary(grade: int) -> str:
    """Human-readable description of a WHO grade, or a placeholder."""
    return WHO_GRADES.get(grade, "WHO grade not applicable.")


def interpret_prediction(
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
) -> Dict[str, object]:
    """
    Build the complete staging payload for one prediction.

    Parameters
    ----------
    predicted_class : str
        Winning class label (e.g. ``"glioma"``).
    confidence : float
        SVM probability of the winning class (0–1).
    probabilities : dict
        ``{class_label: probability}`` for every class.

    Returns
    -------
    dict with keys ``predicted_class``, ``confidence``, ``probabilities``,
    ``profile``, ``grade_range``, ``typical_grade``, ``grade_descriptions``,
    ``malignancy_profile``, ``clinical_notes``, ``disclaimer``.
    """
    profile = _profile_for(predicted_class)
    grades: List[int] = profile["grades"]  # type: ignore[assignment]

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "profile": profile,
        "grade_range": grades,
        "typical_grade": profile["typical_grade"],
        "grade_descriptions": [grade_summary(g) for g in grades],
        "malignancy_profile": profile["malignancy"],
        "clinical_notes": profile["notes"],
        "disclaimer": (
            "Decision-support only. Imaging-based staging is not a substitute "
            "for histopathology and molecular testing; final WHO grading must "
            "be established by a neuropathologist."
        ),
    }


def malignancy_band(predicted_class: str) -> str:
    """Coarse triage band used by the app badge (LOW / HIGH / NONE).

    The comparisons below are against the lowercase, unpunctuated profile keys,
    but every caller passes a canonical label from config.CLASSES ("Glioma",
    "No_Tumor", ...). Without normalising first, none of the branches matched
    and this returned "LOW" for everything — badging a glioma as low-malignancy
    and a healthy scan as low rather than none. Normalise before comparing.
    """
    key = predicted_class.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key == "notumor":
        return "NONE"
    # Glioma is the only HIGH-malignancy class in the platform reference set.
    # Meningioma/pituitary carry "LOW" profiles even though their texts mention
    # high-grade subtypes — we key off the class rather than the literal
    # substring so the badge stays clinically sane.
    if key == "glioma":
        return "HIGH"
    return "LOW"


def format_report_text(staging: Dict[str, object]) -> str:
    """One block of readable clinical text (used in PDF + app)."""
    lines = [
        f"Predicted lesion family : {staging['predicted_class'].upper()}",
        f"Typical WHO presentation : {staging['typical_grade']}",
        f"Malignancy profile       : {staging['malignancy_profile']}",
        "",
        "Grade range and rationale:",
    ]
    for desc in staging["grade_descriptions"]:
        lines.append(f"  • {desc}")
    lines += ["", f"Clinical notes: {staging['clinical_notes']}"]
    return "\n".join(lines)
