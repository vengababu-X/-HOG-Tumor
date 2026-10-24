#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Clinical Image Quality Assurance (QA) & OOD Anomaly Guard
# ==============================================================================
# Implements pre-inference anatomical verification, motion artifact detection,
# and Out-of-Distribution (OOD) anomaly estimation to prevent silent failure
# on non-brain images, corrupt acquisitions, or unrepresented pathologies.
# ==============================================================================
from __future__ import annotations

import math
from typing import Any, Dict, Tuple
import numpy as np


def compute_image_qa(image_array: np.ndarray) -> Dict[str, Any]:
    """Perform pre-inference Quality Assurance on a 2D or 3D grayscale/RGB array.

    Args:
        image_array: np.ndarray, 2D (H, W) or 3D (H, W, C), float [0, 1] or uint8.

    Returns:
        Dict with QA metrics: snr_proxy, motion_blur_score, foreground_fraction,
        symmetry_index, and overall qa_status ('ACCEPTABLE', 'BORDERLINE', 'POOR').
    """
    arr = np.asarray(image_array, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        elif arr.shape[0] in (3, 4):
            arr = arr.mean(axis=0)

    # Normalize to [0, 1]
    rng = float(arr.max() - arr.min())
    if rng > 0:
        norm = (arr - arr.min()) / rng
    else:
        norm = arr

    h, w = norm.shape[:2]

    # 1. Foreground tissue mask (brain vs background air)
    bg_mask = norm < 0.08
    fg_mask = norm >= 0.08
    fg_fraction = float(fg_mask.mean())

    # 2. Signal-to-Noise Ratio (SNR) proxy
    bg_std = float(norm[bg_mask].std()) if bg_mask.sum() > 50 else 0.01
    fg_mean = float(norm[fg_mask].mean()) if fg_mask.sum() > 50 else 0.5
    snr_proxy = float(fg_mean / max(bg_std, 1e-4))

    # 3. Motion blur detection via Laplacian variance
    # Discrete 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Simple valid 2D convolution for blur proxy
    pad = np.pad(norm, 1, mode="edge")
    lap = (
        pad[:-2, 1:-1] * kernel[0, 1] +
        pad[1:-1, :-2] * kernel[1, 0] +
        pad[1:-1, 1:-1] * kernel[1, 1] +
        pad[1:-1, 2:] * kernel[1, 2] +
        pad[2:, 1:-1] * kernel[2, 1]
    )
    motion_blur_var = float(lap.var() * 10000.0)

    # 4. Bilateral symmetry check (horizontal flip correlation)
    flipped = np.fliplr(norm)
    corr = float(np.corrcoef(norm.ravel(), flipped.ravel())[0, 1])
    symmetry_index = max(0.0, corr) if not math.isnan(corr) else 0.5

    # QA Status evaluation
    warnings = []
    if fg_fraction < 0.05:
        warnings.append("Insufficient foreground tissue detected (< 5%).")
    if fg_fraction > 0.95:
        warnings.append("Abnormally high foreground coverage (> 95%) — check for saturation.")
    if motion_blur_var < 5.0 and fg_fraction > 0.1:
        warnings.append("Possible severe motion blur artifact detected.")
    if snr_proxy < 3.0:
        warnings.append("Sub-optimal Signal-to-Noise Ratio (SNR < 3.0).")

    if not warnings:
        status = "ACCEPTABLE"
    elif len(warnings) == 1:
        status = "BORDERLINE"
    else:
        status = "POOR"

    return {
        "status": status,
        "foreground_fraction": round(fg_fraction, 4),
        "snr_proxy": round(snr_proxy, 2),
        "blur_score": round(motion_blur_var, 2),
        "symmetry_index": round(symmetry_index, 3),
        "warnings": warnings,
    }


def evaluate_ood_risk(
    probabilities: Dict[str, float],
    qa_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate Out-Of-Distribution (OOD) risk based on entropy & anatomical plausibility.

    Returns:
        Dict with ood_score (0-100), ood_level ('LOW', 'MODERATE', 'HIGH'),
        normalized_entropy, and recommendation string.
    """
    probs = np.array([float(v) for v in probabilities.values()], dtype=np.float32)
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum()

    # Shannon Entropy H = - sum(p * log(p))
    num_classes = len(probs)
    max_entropy = math.log(num_classes)
    entropy = -float(np.sum(probs * np.log(probs)))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Composite OOD Score
    ood_penalty = 0.0
    if norm_entropy > 0.70:
        ood_penalty += (norm_entropy - 0.70) * 100.0  # high uncertainty penalty

    if qa_metrics.get("status") == "POOR":
        ood_penalty += 35.0
    elif qa_metrics.get("status") == "BORDERLINE":
        ood_penalty += 15.0

    symmetry = qa_metrics.get("symmetry_index", 0.8)
    if symmetry < 0.35:
        ood_penalty += 25.0  # non-cranial geometry penalty

    ood_score = min(100.0, max(0.0, ood_penalty))

    if ood_score >= 60.0:
        level = "HIGH"
        rec = "⚠️ HIGH OOD RISK: Input exhibits high diagnostic entropy or anomalous geometry. Prediction suppressed from automated approval."
    elif ood_score >= 35.0:
        level = "MODERATE"
        rec = "MODERATE UNCERTAINTY: Elevated dispersion across differential classes. Multi-sequence verification strongly advised."
    else:
        level = "LOW"
        rec = "In-Distribution: Model confidence and anatomical tissue characteristics conform to benchmark training standards."

    return {
        "ood_score": round(ood_score, 1),
        "ood_level": level,
        "normalized_entropy": round(norm_entropy, 3),
        "recommendation": rec,
    }

