#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Neurosurgical Planning & Anatomical Subsite Engine
# ==============================================================================
# Computes pre-operative neurosurgical metrics from brain MRI studies:
#   * Anatomical Hemisphere & Subsite Localization (Frontal, Temporal, Parietal, Sellar)
#   * Intra-axial vs. Extra-axial Compartmental Differentiation
#   * Midline Shift / Parenchymal Asymmetry Quantification
#   * Eloquent Cortex / Optic Pathway Proximity Warning
#   * Surgical Approach Recommendations (Craniotomy vs. Transsphenoidal)
# ==============================================================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def analyze_surgical_anatomy(
    image_slice: np.ndarray,
    predicted_class: str,
    saliency_heatmap: Optional[np.ndarray] = None,
    pixel_spacing_mm: float = 0.9375,
) -> Dict[str, Any]:
    """Perform pre-operative neurosurgical anatomical analysis on the MRI slice.

    Args:
        image_slice: 2D float [0, 1] numpy array (H, W).
        predicted_class: 'Glioma', 'Meningioma', 'Pituitary', or 'No_Tumor'.
        saliency_heatmap: Optional 2D float [0, 1] Grad-CAM++ activation map.
        pixel_spacing_mm: Millimeters per pixel (from DICOM or standard calibration).

    Returns:
        Dict with anatomical_subsite, hemisphere, compartment, midline_shift_alert,
        eloquent_proximity_risk, surgical_approach, and clinical_resectability_notes.
    """
    arr = np.asarray(image_slice, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    rng = float(arr.max() - arr.min())
    norm = (arr - arr.min()) / rng if rng > 0 else arr
    h, w = norm.shape

    # If no tumor, return non-operative baseline
    if predicted_class.lower().replace("_", "") == "notumor":
        return {
            "anatomical_subsite": "Normal Cranial Parenchyma",
            "hemisphere": "Bilateral / Symmetric",
            "compartment": "Non-Neoplastic",
            "midline_asymmetry_percent": 0.0,
            "midline_shift_alert": False,
            "eloquent_proximity_risk": "None",
            "surgical_approach": "Non-Surgical / Clinical Follow-up",
            "surgical_notes": "Symmetric cranial vaults without focal space-occupying lesion or mass effect.",
        }

    # Center of mass of lesion activation (or central bright tissue if no map)
    if saliency_heatmap is not None:
        heat = np.asarray(saliency_heatmap, dtype=np.float32)
        if heat.shape != norm.shape:
            # Rescale or take max
            heat_roi = heat > 0.4
        else:
            heat_roi = heat > 0.4
    else:
        heat_roi = norm > 0.55

    if heat_roi.sum() > 20:
        y_indices, x_indices = np.where(heat_roi)
        cy, cx = float(y_indices.mean()), float(x_indices.mean())
        lesion_radius_px = math.sqrt(float(heat_roi.sum()) / math.pi)
        diameter_mm = round(lesion_radius_px * 2.0 * pixel_spacing_mm, 1)
    else:
        cy, cx = h / 2.0, w / 2.0
        diameter_mm = 0.0

    # 1. Radiological Hemisphere Mapping (Standard Radiological Convention:
    # Image Left is Patient Right; Image Right is Patient Left)
    mid_x = w / 2.0
    if abs(cx - mid_x) < (w * 0.08):
        hemisphere = "Central / Midline"
    elif cx < mid_x:
        hemisphere = "Right Cerebral Hemisphere (Radiological Right)"
    else:
        hemisphere = "Left Cerebral Hemisphere (Radiological Left)"

    # 2. Craniocaudal / Anteroposterior Subsite
    if predicted_class.lower() == "pituitary":
        subsite = "Sellar / Suprasellar Region (Pituitary Fossa)"
        compartment = "Extradural / Sellar"
        surgical_approach = "Endoscopic Endonasal Transsphenoidal Approach (EETA)"
        eloquent_risk = "Optic Chiasm Compression / Visual Field Deficit Risk"
        notes = "Confined to/expanding the sella turcica. Pre-op evaluation must include visual field perimetry and full pituitary endocrine panel."
    elif predicted_class.lower() == "meningioma":
        compartment = "Extra-Axial (Dural-based)"
        if cy < h * 0.35:
            subsite = "Frontal Convexity / Parasagittal"
        elif cy > h * 0.65:
            subsite = "Occipital / Tentorial"
        else:
            subsite = "Sphenoid Wing / Convexity / Parasagittal"
        surgical_approach = "Tailored Craniectomy / Craniotomy with Simpson Resection Target"
        eloquent_risk = "Moderate (Cortical Compression without Intrinsic Parenchymal Infiltration)"
        notes = "Extra-axial dural attachment likely. Favorable plane of cleavage between tumor capsule and arachnoid expected; look for dural tail on contrast study."
    else:  # Glioma
        compartment = "Intra-Axial (Infiltrative Parenchymal)"
        if cy < h * 0.38:
            subsite = "Frontal Lobe (Intra-Axial)"
            eloquent_risk = "Elevated: Proximity to Precentral Motor Gyrus / Broca's Area"
        elif cy > h * 0.62:
            subsite = "Occipito-Parietal Region"
            eloquent_risk = "Moderate: Visual Radiation / Sensorimotor Pathway"
        else:
            subsite = "Frontoparietal / Insulo-Temporal Lobe"
            eloquent_risk = "HIGH: Close proximity to Peri-Sylvian Eloquent Cortex / Wernicke's Area"
        surgical_approach = "Image-Guided Stereotactic Craniotomy (Awake Mapping if Eloquent)"
        notes = "Diffuse infiltrative intra-axial margin. Maximal safe resection with functional boundary preservation recommended. Histology & IDH1/2 molecular testing required."

    # 3. Midline Shift & Parenchymal Asymmetry Analysis
    left_half = norm[:, :int(mid_x)]
    right_half = norm[:, int(mid_x):]
    min_w = min(left_half.shape[1], right_half.shape[1])
    diff_density = abs(float(left_half[:, :min_w].sum()) - float(right_half[:, :min_w].sum()))
    total_density = max(float(norm.sum()), 1e-4)
    asymmetry_percent = round((diff_density / total_density) * 100.0, 1)

    midline_shift = asymmetry_percent >= 12.0

    return {
        "anatomical_subsite": subsite,
        "hemisphere": hemisphere,
        "compartment": compartment,
        "estimated_axial_diameter_mm": diameter_mm,
        "midline_asymmetry_percent": asymmetry_percent,
        "midline_shift_alert": midline_shift,
        "eloquent_proximity_risk": eloquent_risk,
        "surgical_approach": surgical_approach,
        "surgical_notes": notes,
    }

