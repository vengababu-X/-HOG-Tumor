#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Streamlit Clinical Dashboard (app.py)
# ==============================================================================
# Standalone interactive clinical dashboard. Loads the trained FreeBuff v4
# model directly (no FastAPI backend required) and provides:
#   * Multi-modal MRI upload (T1c / T2 / FLAIR — single scan also accepted)
#   * TTA inference with the full 4-class confidence spectrum chart
#   * Side-by-side original vs Grad-CAM++ overlay per sequence
#   * Low-confidence safeguarding (< 70% triggers a mandatory review banner)
#   * WHO grading reference (Grades I-IV) and malignancy profile
#   * Hospital-grade ReportLab PDF report download
#
# Run:
#   streamlit run app.py
# (or: python run.py dashboard)
# ==============================================================================
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path when launched from another cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# ---- page config MUST be the first Streamlit command -------------------------
st.set_page_config(
    page_title="FreeBuff v4 — Clinical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    CLASSES,
    LOW_CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
    PDF_REPORT_DIR,
    REGULATORY_DISCLAIMER,
    REGULATORY_SHORTHAND,
    TUMOR_ALERT_THRESHOLD,
)
from inference import MODALITIES, FreeBuffInference
from utils import generate_pdf_report

# ==============================================================================
# CONSTANTS & HELPERS
# ==============================================================================
MODALITY_LABELS: Dict[str, str] = {
    "t1ce": "T1c — T1 post-contrast",
    "t2": "T2 — T2-weighted",
    "flair": "FLAIR — fluid-attenuated",
}

BAND_COLORS: Dict[str, str] = {
    "High": "#2ecc71",
    "Moderate": "#3498db",
    "Low": "#f39c12",
    "Very Low": "#e74c3c",
}


def confidence_band_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.75:
        return "Moderate"
    if confidence >= 0.60:
        return "Low"
    return "Very Low"


def confidence_color(confidence: float) -> str:
    return BAND_COLORS[confidence_band_label(confidence)]


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """(H, W, 3) float [0,1] or uint8 array -> PIL RGB image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ==============================================================================
# MODEL LOADING (cached)
# ==============================================================================
#: Artifact the engine serves. from_dir() resolves the checkpoint from
#: saved_models/ first, then falls back to checkpoints/. Displayed in the
#: sidebar so the dashboard never silently serves a different file than the
#: one training just wrote.
_SERVED_CHECKPOINT_NAME: str = "freebuff_v4.pt"
_SERVED_PCA_NAME: str = "freebuff_pca.pkl"


def _artifact_fingerprint() -> Optional[str]:
    """mtime+size signature of the checkpoint & PCA artifacts on disk.

    Passed as a st.cache_resource argument so the engine reloads automatically
    whenever training overwrites the checkpoint — instead of serving stale
    weights from RAM until the next manual restart.
    """
    parts: List[str] = []
    for sub, name in (("saved_models", _SERVED_CHECKPOINT_NAME),
                      ("checkpoints", _SERVED_CHECKPOINT_NAME),
                      ("saved_models", _SERVED_PCA_NAME)):
        p = _PROJECT_ROOT / sub / name
        if p.exists():
            st_ = p.stat()
            parts.append(f"{sub}/{name}:{int(st_.st_mtime)}:{st_.st_size}")
    return "|".join(parts) if parts else None


@st.cache_resource(show_spinner=True)
def load_engine(fingerprint: Optional[str] = None) -> Tuple[Optional[FreeBuffInference], Optional[str]]:
    """Load the FreeBuff v4 inference engine once per session.

    `fingerprint` keys the cache on the on-disk state of the model artifacts:
    when train.py overwrites the checkpoint, the mtime/size signature changes,
    the cache key changes, and Streamlit reloads the engine — no manual
    restart needed.
    """
    del fingerprint  # cache key only
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        engine = FreeBuffInference.from_dir(_PROJECT_ROOT / "saved_models",
                                            device=device)
        return engine, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


engine, load_error = load_engine(_artifact_fingerprint())


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🧠 FreeBuff v4")
    st.caption("Multi-modal MRI brain tumour classification, "
               "staging & clinical decision support")

    engine, load_error = load_engine()

    st.subheader("System status")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        st.success(f"GPU active — {gpu_name} ({vram_gb:.1f} GB)")
    else:
        st.info("Running on CPU (GPU not detected)")

    if engine is not None:
        st.metric("Model version", f"v{engine.model_version}")
        st.metric("Classes", len(engine.class_names))
        st.metric("HOG-PCA branch", "active" if engine.pca is not None
                  else "deep-only (no PCA artifact)")
        with st.expander("Served artifacts", expanded=False):
            ck_src = getattr(engine, "checkpoint_path", None)
            st.caption(f"Checkpoint: `{Path(ck_src).name if ck_src else _SERVED_CHECKPOINT_NAME}`")
            st.caption(f"Fingerprint: `{_artifact_fingerprint() or 'n/a'}`")
    else:
        st.error("Model not loaded")

    use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                        help="Averages predictions over 48 augmented views. "
                             "More robust, slightly slower.")

    with st.expander("⚠️ Regulatory disclaimer", expanded=False):
        st.caption(REGULATORY_DISCLAIMER)

    st.divider()
    st.caption(f"Python 3.11 · PyTorch {torch.__version__} · "
               f"Streamlit clinical UI")


# ==============================================================================
# MAIN HEADER
# ==============================================================================
st.title("FreeBuff v4 — Clinical Decision Support Dashboard")
st.caption("ResNet50 + CBAM × HOG-PCA hybrid · TTA inference · Grad-CAM++ "
           "explainability · WHO staging reference")

if engine is None:
    st.error(
        "**Model could not be loaded.** "
        f"Details: `{load_error}`\n\n"
        "Train a model first, or place a checkpoint at `saved_models/freebuff_v4.pt`:\n"
        "```bash\npython train.py --data_dir data --num_epochs 60\n```"
    )
    st.stop()

# ==============================================================================
# UPLOAD SECTION
# ==============================================================================
st.subheader("1 · Upload MRI sequences")

col1, col2, col3 = st.columns(3)
uploaders: Dict[str, Any] = {
    "t1ce": col1.file_uploader(MODALITY_LABELS["t1ce"],
                               type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm"],
                               key="up_t1ce"),
    "t2": col2.file_uploader(MODALITY_LABELS["t2"],
                             type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm"],
                             key="up_t2"),
    "flair": col3.file_uploader(MODALITY_LABELS["flair"],
                                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm"],
                                key="up_flair"),
}

patient_id = st.text_input("Patient / study ID (optional)", value="",
                           max_chars=64, placeholder="e.g. ANON-0001")

provided = {m: f for m, f in uploaders.items() if f is not None}
if not provided:
    st.info("Upload at least one MRI sequence to begin. Any missing sequence "
            "is replicated from a present one (logged, never silent).")

analyze_btn = st.button("🔍 Run analysis", type="primary",
                        disabled=not provided,
                        use_container_width=True)


# ==============================================================================
# INFERENCE
# ==============================================================================
def run_analysis(files: Dict[str, Any]) -> None:
    """Execute the full pipeline and stash results in session state."""
    tmp_paths: Dict[str, str] = {}
    try:
        # Persist uploads to temp files (engine takes paths).
        for modality, uploaded in files.items():
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_paths[modality] = tmp.name

        uploads: Dict[str, Optional[np.ndarray]] = {m: None for m in MODALITIES}
        for modality, path in tmp_paths.items():
            uploads[modality] = FreeBuffInference.preprocess_path(path)

        missing = [m for m, a in uploads.items() if a is None]
        if missing and not any(a is not None for a in uploads.values()):
            st.error("No usable scan could be preprocessed.")
            return

        t0 = time.time()
        result = engine.predict_uploads(
            uploads, use_tta=use_tta,
            patient_id=patient_id or None,
        )
        elapsed = time.time() - t0

        # ---- Grad-CAM++ per modality (side-by-side overlays) ---------------
        pred_idx = CLASSES.index(result["predicted_class"]) \
            if result["predicted_class"] in CLASSES else None
        heatmaps = engine.gradcam_maps(uploads, class_idx=pred_idx)

        overlays: Dict[str, np.ndarray] = {}
        originals: Dict[str, np.ndarray] = {}
        for modality, arr in uploads.items():
            if arr is None:
                continue
            base = np.asarray(arr, dtype=np.float32)
            originals[modality] = base
            heat = heatmaps.get(modality)
            if heat is not None:
                overlays[modality] = engine.overlay(base, heat)

        st.session_state["fb_result"] = result
        st.session_state["fb_overlays"] = overlays
        st.session_state["fb_originals"] = originals
        st.session_state["fb_missing"] = missing
        st.session_state["fb_elapsed"] = elapsed
        st.session_state["fb_uploaded_names"] = {
            m: f.name for m, f in files.items()
        }
    finally:
        for p in tmp_paths.values():
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


if analyze_btn and provided:
    with st.spinner("Running TTA inference + Grad-CAM++ …"):
        run_analysis(provided)

# ==============================================================================
# RESULTS RENDERING
# ==============================================================================
result: Optional[Dict[str, Any]] = st.session_state.get("fb_result")

if result is not None:
    confidence = float(result["confidence"])
    predicted = str(result["predicted_class"])
    band = confidence_band_label(confidence)
    low_conf = confidence < LOW_CONFIDENCE_THRESHOLD

    # ---- 2 · PRIMARY FINDING ------------------------------------------------
    st.subheader("2 · Primary finding")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Predicted class", predicted)
    mcol2.metric("Confidence", f"{confidence * 100:.1f}%")
    mcol3.metric("Reliability band", band)
    mcol4.metric("Tumour probability",
                 f"{float(result.get('tumor_probability', 0.0)) * 100:.1f}%")
    st.caption(f"Inference completed in "
               f"{st.session_state.get('fb_elapsed', 0.0):.2f}s · "
               f"TTA {'ON' if use_tta else 'OFF'} · "
               f"modalities analysed: "
               f"{', '.join(result.get('modalities', []))}")

    # ---- LOW-CONFIDENCE SAFEGUARD -------------------------------------------
    if low_conf:
        st.error(
            f"⚠️ **LOW CONFIDENCE SAFEGUARD — {confidence * 100:.1f}% is below the "
            f"{LOW_CONFIDENCE_THRESHOLD * 100:.0f}% reliability threshold.**\n\n"
            "This finding is **provisional** and MUST NOT be used as the basis "
            "for any clinical decision. Mandatory secondary review by a "
            "qualified radiologist is required before the result carries any "
            "clinical weight."
        )
    elif band == "Moderate":
        st.warning(
            f"Confidence is in the **Moderate** band ({confidence * 100:.1f}%). "
            "Correlation with the full imaging series and clinical context is "
            "recommended."
        )
    else:
        st.success(
            f"Confidence is in the **{band}** band ({confidence * 100:.1f}%). "
            "All outputs remain second-read aids requiring verification."
        )

    # Aggregate tumour-probability safety alert.
    if float(result.get("tumor_probability", 0.0)) >= TUMOR_ALERT_THRESHOLD \
            and predicted == "No_Tumor":
        st.warning(
            "**Sensitivity alert:** aggregate tumour probability is "
            f"{result.get('tumor_probability', 0.0) * 100:.1f}% despite a "
            "'No_Tumor' argmax. Radiologist review recommended."
        )

    st.divider()

    # ---- 3 · CONFIDENCE SPECTRUM --------------------------------------------
    st.subheader("3 · Confidence spectrum")
    probs: Dict[str, float] = result["probabilities"]
    left, right = st.columns([3, 2])

    with left:
        prob_series = pd.Series(
            {cls: float(probs.get(cls, 0.0)) for cls in CLASSES}
        ).sort_values(ascending=True)
        st.bar_chart(prob_series, height=320)

    with right:
        rows = sorted(probs.items(), key=lambda kv: -kv[1])
        table = pd.DataFrame(
            [{"Class": cls,
              "Probability": f"{p * 100:.2f}%",
              "Share": p} for cls, p in rows]
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share": st.column_config.ProgressColumn(
                    "Share", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )

    st.divider()

    # ---- 4 · GRAD-CAM++ OVERLAYS (side-by-side) ------------------------------
    st.subheader("4 · Explainability — Grad-CAM++ overlays")
    st.caption("Left: original (pre-processed) sequence · "
               "Right: Grad-CAM++ attention overlay. Red regions contributed "
               "most to the classification decision.")

    overlays: Dict[str, np.ndarray] = st.session_state.get("fb_overlays", {})
    originals: Dict[str, np.ndarray] = st.session_state.get("fb_originals", {})
    if overlays:
        for modality, overlay in overlays.items():
            label = MODALITY_LABELS.get(modality, modality)
            st.markdown(f"**{label}**")
            oc1, oc2 = st.columns(2)
            oc1.image(numpy_to_pil(originals[modality]),
                      caption=f"Original — {label}", use_container_width=True)
            oc2.image(numpy_to_pil(overlay),
                      caption=f"Grad-CAM++ overlay — {label}",
                      use_container_width=True)
            st.divider()
    else:
        st.info("Grad-CAM++ overlays unavailable for this run.")

    # ---- 5 · WHO STAGING REFERENCE -------------------------------------------
    st.subheader("5 · WHO grading & malignancy profile")
    staging: Dict[str, Any] = result.get("staging", {})
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**Typical presentation:** "
                    f"{staging.get('typical_grade', '—')}")
        st.markdown(f"**Possible grade range:** "
                    f"{staging.get('grade_range', '—')}")
        st.markdown(f"**Malignancy profile:** "
                    f"{staging.get('malignancy_profile', '—')}")
    with s2:
        for desc in staging.get("grade_descriptions", []):
            st.markdown(f"- {desc}")
    notes = staging.get("clinical_notes")
    if notes:
        with st.expander("Clinical notes & differential guidance"):
            if isinstance(notes, (list, tuple)):
                for note in notes:
                    st.markdown(f"- {note}")
            else:
                st.markdown(str(notes))
    st.caption(staging.get("disclaimer",
                           "Decision-support only; final WHO grading requires "
                           "histopathology and molecular testing."))

    st.divider()

    # ---- 6 · CLINICAL REASONING ----------------------------------------------
    st.subheader("6 · Automated clinical reasoning")
    with st.expander("Show generated narrative", expanded=not low_conf):
        st.text(result.get("reasoning_text", ""))

    # ---- 7 · PDF REPORT --------------------------------------------------------
    st.subheader("7 · Clinical PDF report")
    st.caption("Generates a hospital-grade ReportLab PDF containing the "
               "finding, confidence spectrum, WHO staging, Grad-CAM++ "
               "overlays, and the regulatory disclaimer.")

    if st.button("📄 Generate PDF report", type="secondary"):
        with st.spinner("Rendering PDF report …"):
            try:
                # Prepare payload expected by utils.generate_pdf_report.
                probs_list = [float(probs.get(cls, 0.0)) for cls in CLASSES]
                gradcam_first = None
                if overlays:
                    first_overlay = next(iter(overlays.values()))
                    gradcam_first = numpy_to_pil(first_overlay)

                pdf_path = generate_pdf_report(
                    inference_results={
                        "prediction": predicted,
                        "confidence": confidence,
                        "probabilities": probs_list,
                        "logits": [0.0] * NUM_CLASSES,
                        "pred_class_idx": CLASSES.index(predicted)
                        if predicted in CLASSES else 0,
                        "image_source": ", ".join(
                            st.session_state.get("fb_uploaded_names", {}).values()
                        ) or "dashboard upload",
                        "tta_used": use_tta,
                        "reasoning_text": result.get("reasoning_text", ""),
                        "confidence_band": band,
                    },
                    stage_info=None,          # auto-looked-up inside the helper
                    gradcam_image=gradcam_first,
                    patient_id=patient_id or None,
                    patient_name=None,
                    exam_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    output_dir=PDF_REPORT_DIR,
                    filename=f"freebuff_report_{patient_id or 'case'}_"
                             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                pdf_bytes = Path(pdf_path).read_bytes()
                st.session_state["fb_pdf"] = pdf_bytes
                st.session_state["fb_pdf_name"] = Path(pdf_path).name
                st.success(f"Report generated: `{Path(pdf_path).name}`")
            except Exception as exc:
                st.error(f"PDF generation failed: {type(exc).__name__}: {exc}")

    if "fb_pdf" in st.session_state:
        st.download_button(
            label="⬇️ Download PDF report",
            data=st.session_state["fb_pdf"],
            file_name=st.session_state.get("fb_pdf_name", "freebuff_report.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(f"{REGULATORY_SHORTHAND} — FreeBuff v4 research prototype, "
           f"{datetime.now().year}.")
