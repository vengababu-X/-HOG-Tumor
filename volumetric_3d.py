#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — 3D Multi-Planar Reconstruction (MPR) & Volumetric Engine
# ==============================================================================
# Provides clinical 3D volumetric MRI processing:
#   * Multi-Planar Reconstruction (MPR): Synchronized Axial, Coronal, Sagittal views
#   * Volumetric depth scanning across craniocaudal slices
#   * 3D Lesion epicenter localization and volumetric tumor burden estimation (cm³)
#   * Native support for NIfTI (.nii, .nii.gz) and slice slab synthesis
# ==============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("neuroscan_cds.volumetric_3d")


def load_or_construct_3d_volume(
    file_or_array: Union[str, Path, np.ndarray],
    target_depth: int = 32,
) -> Tuple[np.ndarray, bool]:
    """Load an authentic 3D volume or construct an interpolated 3D anatomical slab.

    Args:
        file_or_array: Path to .nii/.nii.gz file, or 2D/3D numpy array.
        target_depth: Number of z-slices to synthesize if input is 2D.

    Returns:
        volume: np.ndarray of shape (D, H, W) normalized to [0, 1].
        is_native_3d: True if loaded from a real 3D volume, False if synthetic slab.
    """
    if isinstance(file_or_array, (str, Path)):
        p = Path(file_or_array)
        if p.name.lower().endswith((".nii", ".nii.gz")):
            try:
                import nibabel as nib
                img = nib.load(str(p))
                try:
                    img = nib.as_closest_canonical(img)
                except Exception:
                    pass
                data = img.get_fdata().astype(np.float32)
                if data.ndim == 3:
                    # nibabel gives (H, W, D); transpose to (D, H, W)
                    vol = np.transpose(data, (2, 0, 1))
                else:
                    vol = data
                # Normalize
                rng = float(vol.max() - vol.min())
                if rng > 0:
                    vol = (vol - vol.min()) / rng
                return vol, True
            except Exception as exc:
                logger.warning("Failed to load NIfTI (%s); falling back.", exc)

    arr = np.asarray(file_or_array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    elif arr.ndim == 3 and arr.shape[0] not in (1, 3, 4) and arr.shape[0] > 5:
        # Already a 3D volume (D, H, W)
        rng = float(arr.max() - arr.min())
        if rng > 0:
            arr = (arr - arr.min()) / rng
        return arr, True

    if arr.ndim == 2:
        h, w = arr.shape
        rng = float(arr.max() - arr.min())
        base_slice = (arr - arr.min()) / rng if rng > 0 else arr

        # Construct smooth anatomical 3D slab along Z axis
        z_indices = np.linspace(-1.5, 1.5, target_depth)
        slab = []
        for z in z_indices:
            # Gaussian attenuation factor simulating cranial volume slice taper
            attenuation = math_exp = np.exp(-0.5 * (z ** 2))
            # Smooth scale variation simulating cranial curvature
            scaled_slice = base_slice * attenuation
            slab.append(scaled_slice.astype(np.float32))

        vol = np.stack(slab, axis=0)  # Shape (target_depth, H, W)
        return vol, False

    raise ValueError(f"Unsupported volume shape: {arr.shape}")


def get_mpr_slices(
    volume: np.ndarray,
    axial_idx: Optional[int] = None,
    coronal_idx: Optional[int] = None,
    sagittal_idx: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract orthogonal Multi-Planar Reconstruction (MPR) slices from 3D volume.

    Args:
        volume: np.ndarray of shape (D, H, W), where D is Axial (Z), H is Y, W is X.
        axial_idx: Index along Z axis (0 to D-1).
        coronal_idx: Index along Y axis (0 to H-1).
        sagittal_idx: Index along X axis (0 to W-1).

    Returns:
        Dict with 'axial', 'coronal', 'sagittal' 2D numpy arrays.
    """
    d, h, w = volume.shape

    z = int(np.clip(axial_idx if axial_idx is not None else d // 2, 0, d - 1))
    y = int(np.clip(coronal_idx if coronal_idx is not None else h // 2, 0, h - 1))
    x = int(np.clip(sagittal_idx if sagittal_idx is not None else w // 2, 0, w - 1))

    # 1. Axial Plane: Transverse slice through X-Y
    axial_slice = volume[z, :, :]

    # 2. Coronal Plane: Frontal slice through X-Z (resampled to square aspect)
    coronal_raw = volume[:, y, :]  # Shape (D, W)
    # Resize to (H, W) for display
    coronal_img = Image.fromarray((coronal_raw * 255.0).astype(np.uint8))
    coronal_slice = np.asarray(coronal_img.resize((w, h), Image.Resampling.BILINEAR)) / 255.0

    # 3. Sagittal Plane: Lateral profile slice through Y-Z (resampled to square aspect)
    sagittal_raw = volume[:, :, x]  # Shape (D, H)
    sagittal_img = Image.fromarray((sagittal_raw * 255.0).astype(np.uint8))
    sagittal_slice = np.asarray(sagittal_img.resize((h, h), Image.Resampling.BILINEAR)) / 255.0

    return {
        "axial": axial_slice,
        "coronal": coronal_slice,
        "sagittal": sagittal_slice,
        "coords": {"z": z, "y": y, "x": x},
        "shape": (d, h, w),
    }


def compute_volumetric_tumor_profile(
    volume: np.ndarray,
    engine: Any,
    slice_thickness_mm: float = 5.0,
    pixel_spacing_mm: float = 0.9375,
) -> Dict[str, Any]:
    """Scan through 3D volume slices to produce a craniocaudal tumor profile and burden metrics.

    Returns:
        Dict with z_indices, slice_tumor_probs, peak_slice_index,
        craniocaudal_span_mm, and estimated_volume_cm3.
    """
    d, h, w = volume.shape
    step = max(1, d // 16)  # sample up to 16-24 slices for responsive speed
    sampled_indices = list(range(0, d, step))
    if (d - 1) not in sampled_indices:
        sampled_indices.append(d - 1)

    slice_probs = []
    tumor_areas_pixels = []

    for idx in sampled_indices:
        sl = volume[idx]
        rng = float(sl.max() - sl.min())
        norm = (sl - sl.min()) / rng if rng > 0 else sl
        # Skip pure air background
        if float((norm > 0.1).mean()) < 0.05:
            slice_probs.append(0.0)
            tumor_areas_pixels.append(0)
            continue

        pil_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
        arr = engine.preprocess_path_from_image(pil_img)
        res = engine.predict_uploads({"t1ce": arr, "t2": None, "flair": None}, use_tta=False)
        tp = float(res.get("tumor_probability", 0.0))
        slice_probs.append(tp)

        # Estimate lesion area in pixels if tumor probability is elevated
        if tp >= 0.40:
            # Simple threshold proxy of hyperintense/hypointense lesion tissue
            lesion_mask = (norm > 0.45) & (norm < 0.95)
            tumor_areas_pixels.append(int(lesion_mask.sum()))
        else:
            tumor_areas_pixels.append(0)

    probs_arr = np.array(slice_probs, dtype=np.float32)
    peak_local_idx = int(probs_arr.argmax())
    peak_slice_idx = sampled_indices[peak_local_idx]
    peak_prob = float(probs_arr.max())

    # Calculate craniocaudal span (slices where tumor prob >= 0.35)
    elevated = [sampled_indices[i] for i, p in enumerate(slice_probs) if p >= 0.35]
    if elevated:
        z_min, z_max = min(elevated), max(elevated)
        span_slices = max(1, z_max - z_min + 1)
        craniocaudal_span_mm = span_slices * slice_thickness_mm
    else:
        craniocaudal_span_mm = 0.0

    # Calculate estimated volumetric burden in cm³ (V = sum(area_mm²) * thickness_mm / 1000)
    pixel_area_mm2 = (pixel_spacing_mm ** 2)
    total_volume_mm3 = sum(area * pixel_area_mm2 * (slice_thickness_mm * step) for area in tumor_areas_pixels)
    estimated_volume_cm3 = round(total_volume_mm3 / 1000.0, 2)

    return {
        "sampled_z_indices": sampled_indices,
        "slice_tumor_probabilities": [round(float(p), 3) for p in slice_probs],
        "peak_slice_index": peak_slice_idx,
        "peak_probability": round(peak_prob, 3),
        "craniocaudal_span_mm": round(craniocaudal_span_mm, 1),
        "estimated_volume_cm3": estimated_volume_cm3,
        "n_slices_total": d,
    }

