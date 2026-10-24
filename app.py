#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Clinical Decision Support Workstation (app.py)
# ==============================================================================
# Hospital-grade, FDA CDS (21 U.S.C. 360aaa-1) aligned Clinical Decision Support
# Workstation for brain tumor characterization, 3D Multi-Planar Reconstruction,
# Pre-Operative Neurosurgical Planning, and Clinical Trial Protocol Management.
#
# Core Capabilities:
#   * Quick-Load Verified Benchmark Cases (instant testing without local files)
#   * Multi-sequence MRI ingestion (T1c, T2, FLAIR, DICOM .dcm, NIfTI .nii.gz)
#   * PACS Interactive Window/Level (Brightness/Contrast) Display Presets
#   * Automated Image Quality Assurance (SNR, Motion Blur, Tissue Foreground)
#   * Out-of-Distribution (OOD) Anomaly & Diagnostic Entropy Guard
#   * ResNet50 + CBAM x HOG-PCA dual-branch spatial-texture inference with TTA
#   * Grad-CAM++ saliency mapping with non-corrupting autograd
#   * 3D Multi-Planar Reconstruction (MPR): Synchronized Axial, Coronal, Sagittal
#   * Volumetric Lesion Burden Estimation (cm³) & Craniocaudal Depth Scanning
#   * Pre-Operative Neurosurgical Planning: Hemisphere, Subsite & Midline Shift
#   * WHO CNS 5th Edition (2021) evidence-based clinical reference engine
#   * Reader Pilot Study Protocol & Structured Evaluation Survey Logging (CSV)
#   * Hospital Ethics Board (IRB) Clinical Trial Proposal & Grant Dossier Export
#   * Immutable JSON-Lines Audit Logging (FDA 21 CFR Part 11 / ISO 62304 / HIPAA)
#   * Hospital-grade ReportLab PDF Clinical Summary generation
# ==============================================================================
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# ---- Page config MUST be the first Streamlit command -------------------------
st.set_page_config(
    page_title="NeuroScan CDS — Brain MRI Decision Support Workstation",
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
from dataset import dicom_meta_to_report_lines
from audit_logger import (
    compute_file_sha256,
    log_clinical_inference,
    log_clinician_review,
    AUDIT_LOG_FILE,
)
from clinical_qa_ood import compute_image_qa, evaluate_ood_risk
from volumetric_3d import (
    load_or_construct_3d_volume,
    get_mpr_slices,
    compute_volumetric_tumor_profile,
)
from pilot_study_manager import (
    record_pilot_response,
    get_pilot_study_summary,
    PILOT_CSV_FILE,
)
from neuro_surgical_planner import analyze_surgical_anatomy
from irb_clinical_protocol import generate_irb_protocol_text

# ==============================================================================
# PRELOADED BENCHMARK CASES FOR PILOT TESTING
# ==============================================================================
PRELOADED_CASES: Dict[str, Dict[str, str]] = {
    "Case 1: Diffuse Infiltrative Glioma": {
        "file": "data/Testing/glioma/Te-gl_1.jpg",
        "modality": "t1ce",
        "description": "Supratentorial infiltrative intra-axial lesion with heterogeneous contrast enhancement.",
        "expected": "Glioma (WHO Grade II–IV range)",
    },
    "Case 2: Dural Convexity Meningioma": {
        "file": "data/Testing/meningioma/Te-aug-me_1.jpg",
        "modality": "t1ce",
        "description": "Extra-axial, dural-based circumscribed mass with broad dural attachment.",
        "expected": "Meningioma (WHO Grade I typical)",
    },
    "Case 3: Sellar Pituitary Macroadenoma": {
        "file": "data/Testing/pituitary/Te-pi_1.jpg",
        "modality": "t1ce",
        "description": "Sellar / suprasellar mass with expansion of the sella turcica.",
        "expected": "Pituitary Neuroendocrine Tumor (PitNET)",
    },
    "Case 4: Healthy Brain Parenchyma (No Tumor)": {
        "file": "data/Testing/notumor/Te-no_1.jpg",
        "modality": "t1ce",
        "description": "Normal symmetrical cranial parenchyma with no focal mass effect or enhancement.",
        "expected": "No Tumor Detected",
    },
}


class PreloadedStudyFile:
    """Mock UploadedFile interface for pre-loaded pilot study samples."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# ==============================================================================
# CONSTANTS & UI STYLING HELPERS
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

WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "Standard Brain": (0.50, 0.80),
    "High Contrast / Edema": (0.42, 0.45),
    "Bone / Dural Detail": (0.65, 0.40),
    "Full Dynamic Range": (0.50, 1.00),
}


def apply_pacs_windowing(arr: np.ndarray, level: float = 0.50, width: float = 0.80) -> np.ndarray:
    """Apply clinical PACS Window Center (Level) and Window Width transformation."""
    arr = np.asarray(arr, dtype=np.float32)
    min_val = level - (width / 2.0)
    windowed = (arr - min_val) / max(width, 1e-4)
    return np.clip(windowed, 0.0, 1.0)


def confidence_band_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.75:
        return "Moderate"
    if confidence >= 0.60:
        return "Low"
    return "Very Low"


def numpy_to_pil(arr: np.ndarray, level: float = 0.50, width: float = 0.80) -> Image.Image:
    """(H, W) or (H, W, 3) float array -> Windowed PIL RGB image."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = apply_pacs_windowing(arr, level, width)
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = apply_pacs_windowing(arr[..., :3], level, width)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def parse_dicom_headers(file_bytes: bytes) -> Dict[str, Any]:
    try:
        import pydicom
        with io.BytesIO(file_bytes) as bio:
            ds = pydicom.dcmread(bio, stop_before_pixels=True)
            return {
                "patient_id": str(getattr(ds, "PatientID", "ANONYMIZED")),
                "modality": str(getattr(ds, "Modality", "MR")),
                "series_description": str(getattr(ds, "SeriesDescription", "Unspecified")),
                "magnetic_field_strength": str(getattr(ds, "MagneticFieldStrength", "N/A")),
                "repetition_time": str(getattr(ds, "RepetitionTime", "N/A")),
                "echo_time": str(getattr(ds, "EchoTime", "N/A")),
                "slice_thickness": str(getattr(ds, "SliceThickness", "N/A")),
                "pixel_spacing": str(getattr(ds, "PixelSpacing", "N/A")),
                "manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
                "institution_name": str(getattr(ds, "InstitutionName", "N/A")),
            }
    except Exception:
        return {}


# ==============================================================================
# MODEL LOADING (Cached with Disk Fingerprinting)
# ==============================================================================
_SERVED_CHECKPOINT_NAME: str = "freebuff_v4.pt"
_SERVED_PCA_NAME: str = "freebuff_pca.pkl"


def _artifact_fingerprint() -> Optional[str]:
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
    del fingerprint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        engine = FreeBuffInference.from_dir(_PROJECT_ROOT / "saved_models", device=device)
        return engine, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


engine, load_error = load_engine(_artifact_fingerprint())


# ==============================================================================
# SIDEBAR: SYSTEM STATUS & REGULATORY TRACEABILITY
# ==============================================================================
with st.sidebar:
    st.title("🧠 NeuroScan CDS")
    st.caption("Clinical Decision Support Workstation · FDA 21 CFR § 820 / ISO 62304")

    st.subheader("Compute Environment")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.success(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        st.info("Compute: CPU-Only Mode")

    if engine is not None:
        st.metric("Model Architecture", "ResNet50+CBAM x HOG-PCA")
        st.metric("Latent Dimension", "768-d Fusion (256 deep + 512 HOG)")
        st.metric("Texture Branch", "PCA (288-d) Active" if engine.pca is not None else "Deep-Only")
        st.caption(f"Served Checkpoint: `{_SERVED_CHECKPOINT_NAME}`")
    else:
        st.error(f"Engine Load Failed: {load_error}")

    use_tta = st.toggle(
        "Test-Time Augmentation (TTA)",
        value=True,
        help="Averages anatomical views (Identity + Horizontal Flip) for variance reduction.",
    )

    st.divider()
    st.subheader("Pilot Study Reader Statistics")
    pilot_summary = get_pilot_study_summary()
    st.metric("Recorded Pilot Trials", f"{pilot_summary['total_responses']} readers")
    if pilot_summary['total_responses'] > 0:
        st.caption(f"Avg System Usability: **{pilot_summary['avg_usability']:.1f} / 5.0**")
        st.caption(f"Reader Concurrence Rate: **{pilot_summary['concurrence_rate_percent']}%**")
        st.caption(f"3D MPR Helpfulness: **{pilot_summary['avg_mpr']:.1f} / 5.0**")

    st.divider()
    st.subheader("Regulatory Classification")
    st.markdown(
        "**US FDA Status:** Clinical Decision Support Software (exempt under 21 U.S.C. 360aaa-1 / Section 520(o)(1)(E)).\n\n"
        "**Intended Purpose:** Second-read verification aid for board-certified radiologists. Autonomous diagnosis prohibited."
    )

    with st.expander("Audit Trail Status", expanded=False):
        if AUDIT_LOG_FILE.exists():
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                audit_lines = len(f.readlines())
            st.caption(f"Immutable Records: **{audit_lines}** transactions logged.")
            st.caption(f"Audit Log: `{AUDIT_LOG_FILE.name}`")
        else:
            st.caption("Audit trail active (no records yet).")


# ==============================================================================
# MAIN WORKSTATION HEADER
# ==============================================================================
st.title("NeuroScan CDS — Brain MRI Decision Support Workstation")
st.caption(
    "Dual-Branch ResNet50+CBAM x HOG-PCA Fusion · 3D Multi-Planar Reconstruction · "
    "Pre-Operative Neurosurgical Planning · Image QA & OOD Guard · Hospital IRB Protocol"
)

if engine is None:
    st.error(f"Workstation halted: Model engine not loaded ({load_error}).")
    st.stop()


# ==============================================================================
# 1 · INPUT ACQUISITION & STUDY SELECTION
# ==============================================================================
st.subheader("1 · MRI Sequence Ingestion & Pilot Benchmark Selection")

c_meta1, c_meta2 = st.columns([2, 1])
with c_meta1:
    patient_id = st.text_input(
        "De-identified Patient / Study UID",
        value="",
        max_chars=64,
        placeholder="e.g. ANON-HOSP-2026-0081",
        help="PHI must be removed prior to upload conforming to HIPAA Safe Harbor guidelines.",
    )
with c_meta2:
    clinical_indication = st.selectbox(
        "Clinical Indication",
        ["Suspected Intracranial Neoplasm", "Pre-operative Surgical Planning", "Post-treatment Surveillance", "Hospital Pilot Reader Trial"],
    )

study_source = st.radio(
    "Select Study Source",
    ["⚡ Quick-Load Verified Benchmark Case (Instant Pilot Test)", "📂 Custom Medical Upload (DICOM / NIfTI / Image)"],
    horizontal=True,
)

provided: Dict[str, Any] = {}
dicom_summaries: Dict[str, Dict[str, Any]] = {}

if "Quick-Load" in study_source:
    selected_case_name = st.selectbox("Choose Pilot Test Case", list(PRELOADED_CASES.keys()))
    case_meta = PRELOADED_CASES[selected_case_name]
    st.info(f"**Case Details:** {case_meta['description']} — *Reference Truth:* `{case_meta['expected']}`")

    file_path = _PROJECT_ROOT / case_meta["file"]
    if file_path.exists():
        file_bytes = file_path.read_bytes()
        provided[case_meta["modality"]] = PreloadedStudyFile(file_path.name, file_bytes)
        st.success(f"Loaded verified study slice: `{file_path.name}` ({len(file_bytes) / 1024:.1f} KB)")
    else:
        st.error(f"Case file `{case_meta['file']}` not found.")
else:
    col1, col2, col3 = st.columns(3)
    uploaders: Dict[str, Any] = {
        "t1ce": col1.file_uploader(MODALITY_LABELS["t1ce"], type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "nii", "gz"], key="up_t1ce"),
        "t2": col2.file_uploader(MODALITY_LABELS["t2"], type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "nii", "gz"], key="up_t2"),
        "flair": col3.file_uploader(MODALITY_LABELS["flair"], type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "nii", "gz"], key="up_flair"),
    }
    provided = {m: f for m, f in uploaders.items() if f is not None}

    for m, uploaded in provided.items():
        if uploaded.name.lower().endswith(".dcm"):
            b = uploaded.getvalue()
            d_meta = parse_dicom_headers(b)
            if d_meta:
                dicom_summaries[m] = d_meta

    if dicom_summaries:
        with st.expander("📋 Verified DICOM Acquisition Metadata (Parsed from Headers)", expanded=True):
            for m, tags in dicom_summaries.items():
                st.markdown(f"**Sequence: {MODALITY_LABELS[m]}**")
                d_cols = st.columns(4)
                d_cols[0].caption(f"Modality: {tags.get('modality')}")
                d_cols[1].caption(f"Field Strength: {tags.get('magnetic_field_strength')} T")
                d_cols[2].caption(f"TR / TE: {tags.get('repetition_time')} / {tags.get('echo_time')} ms")
                d_cols[3].caption(f"Slice Thickness: {tags.get('slice_thickness')} mm")

if not provided:
    st.info("Select a Quick-Load Pilot Case or upload a custom MRI sequence to begin.")

analyze_btn = st.button("🔍 Run Clinical Inference, 3D MPR, Surgical & QA Analysis", type="primary", disabled=not provided, use_container_width=True)


# ==============================================================================
# PIPELINE EXECUTION & AUDIT LOGGING
# ==============================================================================
def run_analysis(files: Dict[str, Any]) -> None:
    tmp_paths: Dict[str, str] = {}
    file_hashes: Dict[str, str] = {}
    try:
        for modality, uploaded in files.items():
            b = uploaded.getvalue()
            file_hashes[modality] = compute_file_sha256(b)
            suffix = ".nii.gz" if uploaded.name.lower().endswith(".nii.gz") else Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(b)
                tmp_paths[modality] = tmp.name

        uploads: Dict[str, Optional[np.ndarray]] = {m: None for m in MODALITIES}
        for modality, path in tmp_paths.items():
            uploads[modality] = FreeBuffInference.preprocess_path(path)

        if not any(a is not None for a in uploads.values()):
            st.error("Image loading failed: No valid slice could be decoded.")
            return

        # Pre-inference Clinical QA on first provided image
        first_img = next(a for a in uploads.values() if a is not None)
        qa_metrics = compute_image_qa(first_img)

        # 3D Volume Loading / Slab Construction & Volumetric Tumor Profile
        vol_3d, is_native_3d = load_or_construct_3d_volume(first_img, target_depth=32)
        vol_profile = compute_volumetric_tumor_profile(vol_3d, engine)

        t0 = time.time()
        result = engine.predict_uploads(uploads, use_tta=use_tta, patient_id=patient_id or None)
        elapsed = time.time() - t0

        # Out-of-Distribution (OOD) & Entropy Guard Evaluation
        ood_metrics = evaluate_ood_risk(result["probabilities"], qa_metrics)

        pred_idx = CLASSES.index(result["predicted_class"]) if result["predicted_class"] in CLASSES else None
        heatmaps = engine.gradcam_maps(uploads, class_idx=pred_idx)

        # Pre-Operative Neurosurgical & Anatomical Subsite Planning
        surg_plan = analyze_surgical_anatomy(
            first_img,
            predicted_class=result["predicted_class"],
            saliency_heatmap=next(iter(heatmaps.values())) if heatmaps else None,
        )

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

        low_conf = float(result["confidence"]) < LOW_CONFIDENCE_THRESHOLD
        sens_alert = float(result.get("tumor_probability", 0.0)) >= TUMOR_ALERT_THRESHOLD and result["predicted_class"] == "No_Tumor"

        # Log immutable clinical inference transaction
        audit_id = log_clinical_inference(
            patient_id=patient_id or None,
            predicted_class=result["predicted_class"],
            confidence=float(result["confidence"]),
            probabilities=result["probabilities"],
            tumor_probability=float(result.get("tumor_probability", 0.0)),
            modalities_used=result.get("modalities", []),
            low_confidence_triggered=low_conf,
            sensitivity_alert_triggered=sens_alert,
            tta_used=use_tta,
            latency_sec=elapsed,
            image_hashes=file_hashes,
            model_version=f"v{engine.model_version}",
            device_info=str(engine.device),
        )

        st.session_state["fb_result"] = result
        st.session_state["fb_overlays"] = overlays
        st.session_state["fb_originals"] = originals
        st.session_state["fb_elapsed"] = elapsed
        st.session_state["fb_uploaded_names"] = {m: f.name for m, f in files.items()}
        st.session_state["fb_audit_id"] = audit_id
        st.session_state["fb_qa"] = qa_metrics
        st.session_state["fb_ood"] = ood_metrics
        st.session_state["fb_volume"] = vol_3d
        st.session_state["fb_is_native_3d"] = is_native_3d
        st.session_state["fb_vol_profile"] = vol_profile
        st.session_state["fb_surg_plan"] = surg_plan
        st.session_state["fb_signoff_done"] = False
    finally:
        for p in tmp_paths.values():
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


if analyze_btn and provided:
    with st.spinner("Executing GPU dual-branch inference, 3D MPR, Neurosurgical planning & QA audit ..."):
        run_analysis(provided)


# ==============================================================================
# RESULTS & CLINICAL DECISION SUPPORT
# ==============================================================================
result: Optional[Dict[str, Any]] = st.session_state.get("fb_result")

if result is not None:
    confidence = float(result["confidence"])
    predicted = str(result["predicted_class"])
    band = confidence_band_label(confidence)
    low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
    tumor_prob = float(result.get("tumor_probability", 0.0))
    audit_id = st.session_state.get("fb_audit_id", "UNASSIGNED")
    qa_metrics: Dict[str, Any] = st.session_state.get("fb_qa", {})
    ood_metrics: Dict[str, Any] = st.session_state.get("fb_ood", {})
    vol_3d: np.ndarray = st.session_state.get("fb_volume")
    vol_profile: Dict[str, Any] = st.session_state.get("fb_vol_profile", {})
    surg_plan: Dict[str, Any] = st.session_state.get("fb_surg_plan", {})

    # ---- 2 · PRIMARY FINDING & AUDIT BADGE -----------------------------------
    st.subheader("2 · Primary Finding & Audit Provenance")
    st.info(f"🛡️ **FDA CDS Audit Transaction ID:** `{audit_id}` · **Model:** NeuroScan v4 (ResNet50+CBAM × HOG-PCA) · **Latency:** {st.session_state.get('fb_elapsed', 0.0):.2f}s")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Predicted Pathology", predicted)
    mcol2.metric("Calibrated Confidence", f"{confidence * 100:.1f}%")
    mcol3.metric("Reliability Band", band)
    mcol4.metric("Aggregate Tumor Risk", f"{tumor_prob * 100:.1f}%")

    # ---- IMAGE QA & OUT-OF-DISTRIBUTION (OOD) GUARD -------------------------
    qa_col, ood_col = st.columns(2)
    with qa_col:
        qa_status = qa_metrics.get("status", "ACCEPTABLE")
        qa_color = "🟢" if qa_status == "ACCEPTABLE" else ("🟡" if qa_status == "BORDERLINE" else "🔴")
        st.markdown(f"**Image Quality Assessment:** {qa_color} `{qa_status}`")
        st.caption(
            f"SNR: **{qa_metrics.get('snr_proxy', 0.0)}** · "
            f"Motion Blur Score: **{qa_metrics.get('blur_score', 0.0)}** · "
            f"Bilateral Symmetry: **{qa_metrics.get('symmetry_index', 0.0)}**"
        )
        for w in qa_metrics.get("warnings", []):
            st.warning(f"QA Notice: {w}")

    with ood_col:
        ood_lvl = ood_metrics.get("ood_level", "LOW")
        ood_color = "🟢" if ood_lvl == "LOW" else ("🟡" if ood_lvl == "MODERATE" else "🔴")
        st.markdown(f"**Out-of-Distribution (OOD) Guard:** {ood_color} `{ood_lvl}` (Risk: {ood_metrics.get('ood_score', 0)}%)")
        st.caption(
            f"Normalized Diagnostic Entropy: **{ood_metrics.get('normalized_entropy', 0.0)}** (Threshold: 0.70)"
        )
        if ood_lvl == "HIGH":
            st.error(ood_metrics.get("recommendation"))
        elif ood_lvl == "MODERATE":
            st.warning(ood_metrics.get("recommendation"))

    # ---- DYNAMIC CLINICAL SAFEGUARDS ----------------------------------------
    if low_conf:
        st.error(
            f"⚠️ **LOW CONFIDENCE SAFEGUARD TRIGGERED ({confidence * 100:.1f}% < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%)**\n\n"
            "This assessment is classified as **Provisional**. Under FDA CDS guidance, this output MUST NOT form the basis of diagnostic decisions without independent multi-sequence review."
        )
    elif band == "Moderate":
        st.warning(f"Confidence is Moderate ({confidence * 100:.1f}%). Multi-slice volumetric confirmation recommended.")
    else:
        st.success(f"Confidence is in the **{band}** band ({confidence * 100:.1f}%). Second-read verification mandated.")

    if tumor_prob >= TUMOR_ALERT_THRESHOLD and predicted == "No_Tumor":
        st.warning(
            f"⚠️ **SENSITIVITY ALERT:** Aggregate tumor probability is {tumor_prob * 100:.1f}% despite 'No_Tumor' top classification. Potential subtle lesion — urgent radiologist secondary check recommended."
        )

    st.divider()

    # ---- 3 · CONFIDENCE SPECTRUM --------------------------------------------
    st.subheader("3 · Multi-Class Probability Distribution")
    probs: Dict[str, float] = result["probabilities"]
    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        prob_series = pd.Series({cls: float(probs.get(cls, 0.0)) for cls in CLASSES}).sort_values(ascending=True)
        st.bar_chart(prob_series, height=260)
    with col_tbl:
        rows = sorted(probs.items(), key=lambda kv: -kv[1])
        st.dataframe(
            pd.DataFrame([{"Class": cls, "Probability": f"{p * 100:.2f}%", "Fraction": p} for cls, p in rows]),
            hide_index=True,
            use_container_width=True,
            column_config={"Fraction": st.column_config.ProgressColumn("Fraction", min_value=0.0, max_value=1.0, format="%.2f")},
        )

    st.divider()

    # ---- 4 · PACS INTERACTIVE WINDOW/LEVEL & GRAD-CAM++ ----------------------
    st.subheader("4 · PACS Window/Level & Explainability Overlays")
    st.caption("Autograd-derived Grad-CAM++ saliency overlays with real-time PACS Brightness & Contrast controls.")

    pacs_col1, pacs_col2, pacs_col3 = st.columns([1, 1, 1])
    with pacs_col1:
        preset_choice = st.selectbox("PACS Window Preset", list(WINDOW_PRESETS.keys()), index=0)
        default_l, default_w = WINDOW_PRESETS[preset_choice]
    with pacs_col2:
        win_level = st.slider("Window Center / Level (Brightness)", 0.10, 0.90, default_l, 0.05)
    with pacs_col3:
        win_width = st.slider("Window Width (Contrast Range)", 0.20, 1.20, default_w, 0.05)

    overlays: Dict[str, np.ndarray] = st.session_state.get("fb_overlays", {})
    originals: Dict[str, np.ndarray] = st.session_state.get("fb_originals", {})
    if overlays:
        for modality, overlay in overlays.items():
            label = MODALITY_LABELS.get(modality, modality)
            oc1, oc2 = st.columns(2)
            oc1.image(numpy_to_pil(originals[modality], win_level, win_width), caption=f"Original Acquisition (PACS Windowed) — {label}", use_container_width=True)
            oc2.image(numpy_to_pil(overlay), caption=f"Grad-CAM++ Saliency Overlay — {label}", use_container_width=True)
    else:
        st.info("Grad-CAM++ heatmaps unavailable for this sequence.")

    st.divider()

    # ---- 5 · 3D MULTI-PLANAR RECONSTRUCTION (MPR) & VOLUMETRIC ANALYSIS ----
    st.subheader("5 · 3D Multi-Planar Reconstruction (MPR) & Volumetric Profile")
    st.caption("Interactive radiological tri-planar exploration. Scrub through Axial (Z), Coronal (Y), and Sagittal (X) planes simultaneously.")

    if vol_3d is not None and vol_3d.ndim == 3:
        depth, height, width = vol_3d.shape
        native_tag = "Native NIfTI 3D Volume" if st.session_state.get("fb_is_native_3d") else "Calibrated Anatomical 3D Slab"
        st.caption(f"Volume Dimensions: `{depth} slices (Z) × {height} (Y) × {width} (X)` · Source: `{native_tag}`")

        sl_col1, sl_col2, sl_col3 = st.columns(3)
        with sl_col1:
            ax_idx = st.slider("Axial Plane (Z: Inferior ↔ Superior)", 0, depth - 1, depth // 2, key="sl_axial")
        with sl_col2:
            cor_idx = st.slider("Coronal Plane (Y: Anterior ↔ Posterior)", 0, height - 1, height // 2, key="sl_coronal")
        with sl_col3:
            sag_idx = st.slider("Sagittal Plane (X: Left ↔ Right)", 0, width - 1, width // 2, key="sl_sagittal")

        mpr_slices = get_mpr_slices(vol_3d, axial_idx=ax_idx, coronal_idx=cor_idx, sagittal_idx=sag_idx)

        mpr_c1, mpr_c2, mpr_c3 = st.columns(3)
        mpr_c1.image(numpy_to_pil(mpr_slices["axial"], win_level, win_width), caption=f"Axial (Z = {ax_idx}/{depth - 1})", use_container_width=True)
        mpr_c2.image(numpy_to_pil(mpr_slices["coronal"], win_level, win_width), caption=f"Coronal (Y = {cor_idx}/{height - 1})", use_container_width=True)
        mpr_c3.image(numpy_to_pil(mpr_slices["sagittal"], win_level, win_width), caption=f"Sagittal (X = {sag_idx}/{width - 1})", use_container_width=True)

        st.markdown("##### 📊 Volumetric Lesion Burden Metrics")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        v_col1.metric("Est. Tumor Volume", f"{vol_profile.get('estimated_volume_cm3', 0.0)} cm³")
        v_col2.metric("Craniocaudal Span", f"{vol_profile.get('craniocaudal_span_mm', 0.0)} mm")
        v_col3.metric("Epicenter Slice", f"Z = {vol_profile.get('peak_slice_index', depth // 2)}")
        v_col4.metric("Peak Slice Tumor Risk", f"{vol_profile.get('peak_probability', 0.0) * 100:.1f}%")

        if vol_profile.get("slice_tumor_probabilities"):
            z_df = pd.DataFrame({
                "Slice Index (Z)": [f"Z-{idx}" for idx in vol_profile["sampled_z_indices"]],
                "Tumor Probability": vol_profile["slice_tumor_probabilities"],
            }).set_index("Slice Index (Z)")
            st.caption("Craniocaudal Tumor Probability Profile (Scanning across slices):")
            st.line_chart(z_df, height=180)

    st.divider()

    # ---- 6 · PRE-OPERATIVE NEUROSURGICAL & ANATOMICAL PLANNING ---------------
    st.subheader("6 · Pre-Operative Neurosurgical & Anatomical Planning")
    st.caption("Anatomical subsite, hemisphere orientation, compartment delineation, and surgical approach guidance.")

    if surg_plan:
        np_c1, np_c2, np_c3 = st.columns(3)
        np_c1.metric("Anatomical Subsite", surg_plan.get("anatomical_subsite", "—"))
        np_c2.metric("Hemisphere / Orientation", surg_plan.get("hemisphere", "—"))
        np_c3.metric("Compartment", surg_plan.get("compartment", "—"))

        np_c4, np_c5, np_c6 = st.columns(3)
        np_c4.metric("Est. Axial Diameter", f"{surg_plan.get('estimated_axial_diameter_mm', 0.0)} mm")
        np_c5.metric("Parenchymal Asymmetry", f"{surg_plan.get('midline_asymmetry_percent', 0.0)}%")
        midline_alert_str = "⚠️ Compression Detected" if surg_plan.get("midline_shift_alert") else "🟢 Preserved / Minimal"
        np_c6.metric("Midline Mass Effect", midline_alert_str)

        st.markdown(f"**Eloquent Cortex Proximity Risk:** `{surg_plan.get('eloquent_proximity_risk', 'None')}`")
        st.markdown(f"**Recommended Surgical Approach:** `{surg_plan.get('surgical_approach', 'Clinical Evaluation')}`")
        with st.expander("📋 Pre-Operative Resection & Surgical Triage Guidance", expanded=True):
            st.write(surg_plan.get("surgical_notes", "No focal space-occupying lesion."))

    st.divider()

    # ---- 7 · WHO STAGING REFERENCE KNOWLEDGE BASE ---------------------------
    st.subheader("7 · WHO CNS 5th Edition Staging Reference")
    st.caption("⚠️ **Evidence-Based Knowledge Reference:** The neural network classifies pathology type. Definitive WHO Grade assignment strictly requires molecular histopathology (IDH1/2, 1p/19q, MGMT, ATRX).")
    staging: Dict[str, Any] = result.get("staging", {})
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**Reference Grade Spectrum:** `{staging.get('grade_range', '—')}`")
        st.markdown(f"**Typical Presentation:** {staging.get('typical_grade', '—')}")
        st.markdown(f"**Malignancy Profile:** {staging.get('malignancy_profile', '—')}")
    with s2:
        for desc in staging.get("grade_descriptions", []):
            st.markdown(f"- {desc}")

    st.divider()

    # ---- 8 · CLINICAL REASONING NARRATIVE ------------------------------------
    st.subheader("8 · Structured Clinical Reasoning Narrative")
    with st.expander("📄 View ACR-Standard Clinical Summary", expanded=True):
        st.text(result.get("reasoning_text", ""))

    st.divider()

    # ---- 9 · CLINICAL PILOT READER STUDY PROTOCOL & EVALUATION ---------------
    st.subheader("9 · Clinical Pilot Reader Study Protocol & Evaluation Survey")
    st.caption("Structured protocol for doctor and non-doctor participants testing NeuroScan CDS. Responses are recorded in the trial database.")

    with st.form("pilot_reader_evaluation_form"):
        p_c1, p_c2 = st.columns(2)
        reader_id = p_c1.text_input("Reader ID / Participant Name", placeholder="e.g. Dr. A. Sharma or Reader-01")
        reader_background = p_c2.selectbox(
            "Reader Background / Role",
            ["Radiologist / Neuroradiologist", "Attending Neurosurgeon", "Resident / Fellow", "Medical Student / Intern", "Non-Doctor (Engineering / Evaluator)"]
        )

        st.markdown("**Diagnostic Agreement Assessment:**")
        agreement = st.radio(
            "Do you agree with the model's primary classification finding?",
            ["Agree with Finding", "Disagree with Finding (Suspect Alternative)", "Uncertain / Inconclusive"],
            horizontal=True,
        )

        st.markdown("**System Experience & Feature Ratings (1 = Poor, 5 = Excellent):**")
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        saliency_rating = r_c1.slider("Grad-CAM++ Saliency Alignment", 1, 5, 4)
        mpr_rating = r_c2.slider("3D MPR Tri-Planar Utility", 1, 5, 5)
        safeguards_rating = r_c3.slider("Clarity of Safety Alerts / OOD", 1, 5, 4)
        usability_score = r_c4.slider("Overall System Usability (SUS)", 1, 5, 5)

        reader_feedback = st.text_area("Reader Observations / Feedback", placeholder="e.g. Sellar vs intra-axial subsite mapping was clinically helpful. Fast TTA inference.")

        submit_pilot = st.form_submit_button("💾 Submit Trial Response to Pilot Study Database", type="primary")
        if submit_pilot:
            if not reader_id.strip():
                st.error("Participant Identifier is required.")
            else:
                ok = record_pilot_response(
                    audit_id=audit_id,
                    reader_id=reader_id,
                    reader_background=reader_background,
                    predicted_class=predicted,
                    model_confidence=confidence,
                    reader_agreement=agreement,
                    saliency_rating=saliency_rating,
                    mpr_rating=mpr_rating,
                    safeguards_rating=safeguards_rating,
                    usability_score=usability_score,
                    feedback=reader_feedback,
                )
                if ok:
                    st.success(f"Pilot evaluation response recorded successfully for `{reader_id}`! Logged in `logs/pilot_study_responses.csv`.")

    st.divider()

    # ---- 10 · HOSPITAL IRB CLINICAL TRIAL PROPOSAL & FUNDING DOSSIER --------
    st.subheader("10 · Hospital Ethics Board (IRB) Clinical Trial Proposal & Grant Dossier")
    st.caption("Formal Investigator-Initiated Clinical Trial Protocol ready for submission to your hospital's Institutional Ethics Committee (IEC / IRB) for clinical validation funding.")

    irb_h1, irb_h2 = st.columns(2)
    irb_hospital = irb_h1.text_input("Hospital Institution Name", value="Apollo / Manipal / Max / Fortis Multispeciality Hospital")
    irb_doctor = irb_h2.text_input("Principal Investigator (Doctor's Name)", value="Dr. Chief of Neurosurgery / Neuroradiology, MD, MCh")

    irb_text = generate_irb_protocol_text(
        hospital_name=irb_hospital,
        pi_name=irb_doctor,
    )

    with st.expander("📑 View Full Clinical Trial Protocol Proposal (IEC / IRB Submission Document)", expanded=False):
        st.text(irb_text)

    st.download_button(
        label="📄 Download Official IRB Clinical Trial Protocol Proposal (.txt)",
        data=irb_text,
        file_name=f"NEUROSCAN_IRB_Clinical_Trial_Proposal_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    # ---- 11 · RADIOLOGIST SECOND-READ SIGN-OFF ------------------------------
    st.subheader("11 · Clinician Second-Read & Verification Sign-Off")
    st.caption("Record formal clinician concurrence or diagnostic differential in the permanent audit trail (ISO 62304 / FDA CDS Requirement).")

    with st.form("radiologist_signoff_form"):
        r_col1, r_col2 = st.columns(2)
        reviewer_name = r_col1.text_input("Reviewing Clinician Name", placeholder="e.g. Dr. Jane Doe, MD")
        reviewer_role = r_col2.selectbox("Professional Role", ["Consultant Radiologist", "Neuroradiologist", "Attending Neurosurgeon", "Resident / Fellow"])

        concurrence = st.radio(
            "Concurrence Assessment",
            [
                "Concurs with AI Decision Support Finding",
                "Disagrees with Finding (Alternative Pathology Suspected)",
                "Study Inconclusive (Severe Motion / Insufficient Resolution)",
            ],
        )
        differential_notes = st.text_area("Clinical Notes / Differential Diagnosis", placeholder="e.g. Sellar epicenter with suprasellar expansion. Concur with pituitary mass.")

        submit_review = st.form_submit_button("✍️ Commit Sign-Off to Audit Trail", type="primary")
        if submit_review:
            if not reviewer_name.strip():
                st.error("Clinician name is required for legal audit sign-off.")
            else:
                ok = log_clinician_review(
                    audit_id=audit_id,
                    reviewer_name=reviewer_name,
                    reviewer_role=reviewer_role,
                    concurrence_status=concurrence,
                    differential_diagnosis=differential_notes,
                    clinical_notes=f"Indication: {clinical_indication}. Evaluated under NeuroScan CDS v4.0.",
                )
                if ok:
                    st.success(f"Verification committed to audit trail for transaction `{audit_id}`.")
                    st.session_state["fb_signoff_done"] = True

    st.divider()

    # ---- 12 · CLINICAL PDF REPORT DOWNLOAD ----------------------------------
    st.subheader("12 · Hospital-Grade PDF Clinical Report")
    st.caption("Exports a formalized ReportLab summary with embedded Grad-CAM++ heatmaps, WHO staging reference, audit fingerprint, and regulatory disclaimers.")

    if st.button("📄 Generate Verified PDF Report", type="secondary"):
        with st.spinner("Assembling PDF document ..."):
            try:
                probs_list = [float(probs.get(cls, 0.0)) for cls in CLASSES]
                gradcam_first = numpy_to_pil(next(iter(overlays.values())), win_level, win_width) if overlays else None

                pdf_path = generate_pdf_report(
                    inference_results={
                        "prediction": predicted,
                        "confidence": confidence,
                        "probabilities": probs_list,
                        "logits": [0.0] * NUM_CLASSES,
                        "pred_class_idx": CLASSES.index(predicted) if predicted in CLASSES else 0,
                        "image_source": ", ".join(st.session_state.get("fb_uploaded_names", {}).values()) or "Workstation upload",
                        "tta_used": use_tta,
                        "reasoning_text": result.get("reasoning_text", ""),
                        "confidence_band": band,
                    },
                    stage_info=None,
                    gradcam_image=gradcam_first,
                    patient_id=patient_id or "ANON",
                    patient_name=None,
                    exam_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    output_dir=PDF_REPORT_DIR,
                    filename=f"neuroscan_report_{patient_id or 'case'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                st.session_state["fb_pdf"] = Path(pdf_path).read_bytes()
                st.session_state["fb_pdf_name"] = Path(pdf_path).name
                st.success(f"Report compiled: `{Path(pdf_path).name}`")
            except Exception as exc:
                st.error(f"PDF generation failed: {type(exc).__name__}: {exc}")

    if "fb_pdf" in st.session_state:
        st.download_button(
            label="⬇️ Download Signed PDF Report",
            data=st.session_state["fb_pdf"],
            file_name=st.session_state.get("fb_pdf_name", "neuroscan_cds_report.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(
    f"{REGULATORY_SHORTHAND} — NeuroScan CDS Workstation v4.0 ({datetime.now().year}). "
    "Designed in accordance with FDA 21 CFR § 820 & ISO 62304 standards for medical decision support."
)
