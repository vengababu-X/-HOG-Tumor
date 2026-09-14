#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Inference & Explainability Engine (inference.py)
# ==============================================================================
# End-to-end inference stack:
#   * TTA (test-time augmentation) — flips / rotations / scales, probability
#     averaged across views for variance reduction near decision boundaries.
#   * Grad-CAM++ heatmap rendering with detached feature hooks (safe overlay).
#   * Clinical reasoning narrative generation (deterministic template).
#   * WHO staging lookups (Grades I-IV) via the WHO_STAGING reference table.
#   * `FreeBuffInference` — the multi-modal (T1c / T2 / FLAIR) engine consumed
#     by the Streamlit dashboard, the CLI, and the explainability harness.
#
# Run a self-test:  python inference.py
# ==============================================================================
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from config import (
    CLASSES,
    DEFAULT_MODEL_SAVE_PATH,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    GRADCAM_EXPORT_DIR,
    GRADCAM_TARGET_LAYER_NAME,
    HOG_PCA_COMPONENTS,
    INPUT_SIZE,
    INTENSITY_CLIP_MAX,
    INTENSITY_CLIP_MIN,
    LOW_CONFIDENCE_THRESHOLD,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    REGULATORY_SHORTHAND,
    TTA_FULL,
    TTA_NUM_SAMPLES,
    TTA_ROTATIONS_DEG,
    TTA_SCALE_FACTORS,
    TTA_SOFTMAX_TEMP,
    WHO_STAGING,
)
from model import FreeBuffV4, GradCAMPlusPlus, load_checkpoint
from dataset import (
    HOGExtractor,
    img_tensor_to_numpy_gray,
    load_dicom_slice,
    load_nifti_volume,
    normalize_intensity,
)
from preprocessing import image_to_tensor, preprocess_pil_image

logger = logging.getLogger("freebuff_v4.inference")

#: Indices of every class that represents an actual tumour (i.e. != notumor).
TUMOR_CLASS_INDICES: Tuple[int, ...] = tuple(
    idx for idx, name in enumerate(CLASSES) if name.lower() != "notumor"
)

#: Aggregate tumour-probability threshold for the sensitivity-first alert.
TUMOR_ALERT_THRESHOLD: float = 0.30


# ==============================================================================
# SENSITIVITY-FIRST CLINICAL DECISION RULE
# ==============================================================================
def clinical_decision(
    probabilities: Tensor,
    tumor_threshold: float = TUMOR_ALERT_THRESHOLD,
    no_tumor_index: int = 2,
) -> Dict[str, Any]:
    """Apply the safety-first decision rule without hiding uncertainty.

    If the aggregate tumour probability exceeds `tumor_threshold` while the
    argmax class is 'No_Tumor', the highest-probability tumour class is
    surfaced alongside an explicit review alert (the raw probabilities are
    always reported unchanged, so nothing is ever hidden from the reader).
    """
    probs = probabilities.detach().float()
    tumor_indices = list(TUMOR_CLASS_INDICES)
    tumor_probability = float(1.0 - probs[no_tumor_index].item()) if 'no_tumor_index' in locals() else float(1.0 - probs[2].item())
    argmax_class = int(probs.argmax().item())

    if tumor_probability >= tumor_threshold and argmax_class == no_tumor_index:
        best_tumor = tumor_indices[int(probs[tumor_indices].argmax().item())]
        selected_class = best_tumor
    else:
        selected_class = argmax_class

    return {
        "pred_class": selected_class,
        "tumor_probability": tumor_probability,
        "potential_tumor": tumor_probability >= tumor_threshold,
        "safety_alert": (
            "Potential Tumor Detected: requires radiologist review."
            if tumor_probability >= tumor_threshold else ""
        ),
        "threshold_override": selected_class != argmax_class,
    }


def tumor_safety_alert(
    probabilities: Tensor,
    threshold: float = TUMOR_ALERT_THRESHOLD,
) -> Dict[str, Any]:
    """Flag meaningful aggregate tumour probability without overriding the
    argmax diagnosis (companion to `clinical_decision`)."""
    probs = probabilities.detach().float()
    tumor_probability = float(1.0 - probs[2].item())
    return {
        "potential_tumor": tumor_probability >= threshold,
        "tumor_probability": tumor_probability,
        "safety_alert": (
            "Potential Tumor Detected: requires radiologist review."
            if tumor_probability >= threshold else ""
        ),
    }


# ==============================================================================
# TTA TRANSFORMS
# ==============================================================================
def _make_tta_fn(
    flip: str = "none",
    rotation_deg: int = 0,
    scale: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """Build one deterministic (3, H, W) -> (3, H, W) TTA transform."""

    def tta_fn(img: Tensor) -> Tensor:
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"TTA expects a (3, H, W) tensor, got {tuple(img.shape)}")
        x = img
        if flip == "horizontal":
            x = TF.hflip(x)
        elif flip == "vertical":
            x = TF.vflip(x)
        elif flip == "both":
            x = TF.hflip(TF.vflip(x))
        if rotation_deg != 0:
            x = TF.rotate(x, angle=-float(rotation_deg),
                          interpolation=TF.InterpolationMode.BILINEAR,
                          expand=False, fill=0.0)
        if scale != 1.0:
            h, w = x.shape[1], x.shape[2]
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            x = TF.resize(x, [new_h, new_w],
                          interpolation=TF.InterpolationMode.BILINEAR)
            if new_h > h or new_w > w:
                top, left = max(0, (new_h - h) // 2), max(0, (new_w - w) // 2)
                x = TF.crop(x, top=top, left=left, height=h, width=w)
            elif new_h < h or new_w < w:
                pad_h, pad_w = h - new_h, w - new_w
                x = TF.pad(x, [pad_w // 2, pad_h // 2,
                               pad_w - pad_w // 2, pad_h - pad_h // 2], fill=0.0)
            x = TF.resize(x, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
        return x.contiguous()

    return tta_fn


def build_tta_transforms(
    flips: Optional[Sequence[str]] = None,
    rotations_deg: Optional[Sequence[int]] = None,
    scales: Optional[Sequence[float]] = None,
    full: bool = TTA_FULL,
    max_views: Optional[int] = TTA_NUM_SAMPLES,
) -> List[Callable[[Tensor], Tensor]]:
    """Assemble the deterministic TTA view set.

    With `full=True` the cartesian product of flips x rotations x scales is
    generated (48 views with the default config); `max_views` downsamples the
    list deterministically (evenly spaced) to bound GPU cost on 4 GB cards.
    """
    flips = flips if flips is not None else ["none", "horizontal", "vertical", "both"]
    rotations_deg = rotations_deg if rotations_deg is not None else TTA_ROTATIONS_DEG
    scales = scales if scales is not None else TTA_SCALE_FACTORS

    transforms: List[Callable[[Tensor], Tensor]] = []
    for flip in flips:
        for rot in rotations_deg:
            for scale in scales:
                transforms.append(_make_tta_fn(flip, rot, scale))
                if not full:
                    break
            if not full:
                break
        if not full:
            break

    if max_views is not None and len(transforms) > max_views and max_views > 0:
        idx = np.linspace(0, len(transforms) - 1, max_views).astype(int)
        transforms = [transforms[i] for i in dict.fromkeys(idx.tolist())]

    logger.debug("TTA view count: %d", len(transforms))
    return transforms


# ==============================================================================
# TTA PREDICTOR
# ==============================================================================
class TTAPredictor:
    """Aggregates logits/probabilities across augmented views of one input."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transforms: Optional[List[Callable[[Tensor], Tensor]]] = None,
        softmax_temp: float = TTA_SOFTMAX_TEMP,
        use_hog: bool = False,
        hog_extractor: Optional[HOGExtractor] = None,
        pca: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.transforms = transforms or build_tta_transforms()
        self.softmax_temp = max(float(softmax_temp), 1e-4)
        self.use_hog = use_hog
        self.hog_extractor = hog_extractor
        self.pca = pca

    def predict(
        self,
        image: Tensor,
        hog_pca: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """TTA prediction for a single (3, H, W) image tensor.

        Returns avg_logits (C,), avg_probs (C,), pred_class, confidence,
        the safety-alert payload, and per-view probability/logit lists.
        """
        image_4d = image.unsqueeze(0) if image.dim() == 3 else image
        image_4d = image_4d.to(self.device, dtype=next(self.model.parameters()).dtype)

        hog_batch = None
        if hog_pca is not None:
            hog_batch = (hog_pca.unsqueeze(0) if hog_pca.dim() == 1 else hog_pca
                         ).to(self.device, dtype=torch.float32)

        all_logits: List[Tensor] = []
        all_probs: List[Tensor] = []
        with torch.inference_mode():
            for fn in self.transforms:
                view = fn(image_4d.squeeze(0)).unsqueeze(0)
                out = self.model(view, hog_pca=hog_batch)
                logits = out["logits"].squeeze(0).detach().float()
                all_logits.append(logits)
                all_probs.append(torch.softmax(logits / self.softmax_temp, dim=0))

        stacked = torch.stack(all_logits, dim=0)                    # (T, C)
        per_view_probs = torch.softmax(stacked / self.softmax_temp, dim=1)
        avg_logits = stacked.mean(dim=0)
        avg_probs = per_view_probs.mean(dim=0)
        avg_probs = avg_probs / avg_probs.sum().clamp_min(1e-8)

        decision = clinical_decision(avg_probs)
        safety = tumor_safety_alert(avg_probs)

        return {
            "avg_logits": avg_logits,
            "avg_probs": avg_probs,
            "pred_class": int(decision["pred_class"]),
            "confidence": float(avg_probs[decision["pred_class"]].item()),
            **safety,
            "threshold_override": decision["threshold_override"],
            "per_transform_probs": all_probs,
            "per_transform_logits": all_logits,
        }

    def predict_batch(
        self,
        images: Tensor,
        hog_pcas: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """TTA prediction for a (B, 3, H, W) batch."""
        all_logits: List[Tensor] = []
        with torch.inference_mode():
            for fn in self.transforms:
                aug = torch.stack([fn(img) for img in images], dim=0)
                out = self.model(aug, hog_pca=hog_pcas)
                all_logits.append(out["logits"].detach().float())
        stacked = torch.stack(all_logits, dim=0)                    # (T, B, C)
        avg_logits = stacked.mean(dim=0)
        avg_probs = torch.softmax(stacked / self.softmax_temp, dim=2).mean(dim=0)
        avg_probs = avg_probs / avg_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        argmax_classes = avg_probs.argmax(dim=1)
        tumor_probs = avg_probs[:, list(TUMOR_CLASS_INDICES)].sum(dim=1)
        override = (tumor_probs >= TUMOR_ALERT_THRESHOLD) & \
                   (argmax_classes == CLASSES.index("No_Tumor")
                    if "No_Tumor" in CLASSES else argmax_classes < 0)
        tumor_choice = avg_probs[:, list(TUMOR_CLASS_INDICES)].argmax(dim=1)
        tumor_tensor = torch.tensor(TUMOR_CLASS_INDICES, device=avg_probs.device)
        pred_classes = argmax_classes.clone()
        pred_classes[override] = tumor_tensor[tumor_choice[override]]

        return {
            "avg_logits": avg_logits,
            "avg_probs": avg_probs,
            "pred_classes": pred_classes,
            "confidences": avg_probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1),
            "tumor_probabilities": tumor_probs,
            "potential_tumor": tumor_probs >= TUMOR_ALERT_THRESHOLD,
            "threshold_override": override,
            "per_transform_logits": [stacked[t] for t in range(stacked.size(0))],
        }


def fit_temperature(logits: Tensor, labels: Tensor, max_iter: int = 50) -> float:
    """Fit a scalar softmax temperature on held-out validation logits (LBFGS).

    Persist the returned value and pass it as `softmax_temp` at inference.
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError("logits must be (N, C) and labels must be (N,)")
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    log_t = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure() -> Tensor:
        opt.zero_grad()
        temp = log_t.exp().clamp(0.05, 10.0)
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 10.0).item())


# ==============================================================================
# GRAD-CAM++ OVERLAY RENDERING
# ==============================================================================
def render_gradcam_overlay(
    model: nn.Module,
    image: Tensor,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    alpha: float = GRADCAM_ALPHA,
    colormap: str = GRADCAM_COLORMAP,
    output_path: Optional[Path] = None,
) -> Tuple[Any, Tensor]:
    """Compute (and optionally save) a Grad-CAM++ overlay for one image.

    Returns (PIL RGB overlay, heatmap tensor). On any internal failure a
    neutral grey overlay is returned so the clinical UI never crashes.
    """
    from PIL import Image as PILImage

    device = device or next(model.parameters()).device
    model.eval()
    image = image.detach().contiguous().to(device)

    engine = GradCAMPlusPlus(model, target_layer_name=GRADCAM_TARGET_LAYER_NAME,
                             device=device)
    try:
        heatmap, _details = engine.generate_heatmap(image.unsqueeze(0),
                                                    class_idx=class_idx)
    except Exception as exc:
        logger.warning("Grad-CAM++ generation failed (%s); using fallback.", exc)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        return (PILImage.new("RGB", (w, h), (128, 128, 128)),
                torch.zeros(1, 1, h, w, device=device))
    finally:
        engine.remove_hooks()

    overlay = GradCAMPlusPlus.overlay_heatmap_on_image(
        image, heatmap, alpha=alpha, colormap=colormap,
    )
    overlay_np = overlay.detach().cpu().numpy().astype(np.float32)
    if overlay_np.ndim == 4:
        overlay_np = overlay_np[0]
    overlay_np = np.clip(overlay_np, 0.0, 1.0)
    overlay_np = (overlay_np * 255.0).astype(np.uint8).transpose(1, 2, 0)
    overlay_pil = PILImage.fromarray(overlay_np, mode="RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_pil.save(str(output_path))
        logger.info("Grad-CAM++ overlay saved to %s", output_path)

    return overlay_pil, heatmap


# ==============================================================================
# SINGLE-IMAGE PREPROCESSING
# ==============================================================================
def preprocess_image_for_inference(
    image_source: Union[str, Path, Image.Image, np.ndarray],
    size: Tuple[int, int] = INPUT_SIZE,
    normalize: bool = True,
    as_tensor: bool = True,
) -> Union[Tensor, Image.Image]:
    """Load/convert any supported image source into a model-ready input.

    Supports PNG/JPG paths, DICOM slices (.dcm), NIfTI volumes (.nii),
    PIL images, and numpy arrays. Returns a normalized (3, H, W) tensor.
    """
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        ext = path.suffix.lower()
        if ext in (".dcm", ".dicom"):
            arr = load_dicom_slice(path)
            arr = normalize_intensity(arr, INTENSITY_CLIP_MIN, INTENSITY_CLIP_MAX)
            img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
        elif ext in (".nii", ".gz", ".nifti") or str(path).endswith(".nii.gz"):
            vol = load_nifti_volume(path)
            slice_2d = vol[vol.shape[0] // 2]
            slice_2d = normalize_intensity(slice_2d, INTENSITY_CLIP_MIN,
                                           INTENSITY_CLIP_MAX)
            img = Image.fromarray((slice_2d * 255.0).astype(np.uint8), mode="L")
        else:
            img = Image.open(path)
    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            img = Image.fromarray(
                (np.clip(image_source, 0, 1) * 255).astype(np.uint8), mode="L")
        elif image_source.ndim == 3:
            img = Image.fromarray(image_source.astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    if not as_tensor:
        return img
    if normalize:
        return image_to_tensor(img, size=size)
    prepared = preprocess_pil_image(img, size=size)
    return torch.from_numpy(np.array(prepared, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).contiguous()


# ==============================================================================
# WHO STAGING LOOKUP (GRADES I-IV)
# ==============================================================================
def lookup_who_stage(
    class_name: str,
    grade: Optional[int] = None,
) -> Dict[str, str]:
    """WHO staging record for `class_name` (optionally a specific grade 1-4).

    Returns the grade dict (grade_label, typical_imaging_features,
    malignancy, growth_pattern, recommended_intervention_protocol), with a
    safe fallback for unknown classes.
    """
    unknown = {
        "grade_label": "Unknown",
        "typical_imaging_features": "No staging information available for this class.",
        "malignancy": "Unknown",
        "growth_pattern": "Unknown",
        "recommended_intervention_protocol": "Consult a specialist.",
    }
    staging_dict = WHO_STAGING.get(class_name)
    if staging_dict is None:
        return unknown
    if grade is None:
        # Default to the most representative entry (grade 1 / first present).
        return dict(staging_dict.get(1, next(iter(staging_dict.values()))))
    grade = int(grade)
    return dict(staging_dict.get(grade, staging_dict.get(1, unknown)))


# ==============================================================================
# CLINICAL REASONING NARRATIVE
# ==============================================================================
def generate_clinical_reasoning(
    pred_class: str,
    confidence: float,
    probs: Optional[Union[Tensor, Sequence[float]]] = None,
    image_path: Optional[str] = None,
    stage_info: Optional[Dict[str, str]] = None,
) -> str:
    """Deterministic clinical reasoning narrative for one prediction."""
    lines: List[str] = []
    lines.append("FINDINGS")
    lines.append(f"  Predicted class      : {pred_class}")
    lines.append(f"  Prediction confidence: {confidence * 100:.1f}%")
    lines.append(f"  Reliability band     : {confidence_band(confidence)}")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        lines.append("  WARNING: confidence below the 70% clinical threshold; "
                     "secondary review required.")
    lines.append("")

    if probs is not None:
        lines.append("CLASS PROBABILITY SPECTRUM")
        values = probs.detach().cpu().tolist() if isinstance(probs, Tensor) else list(probs)
        for idx, cls in enumerate(CLASSES):
            if idx < len(values):
                bar = "#" * int(round(float(values[idx]) * 30))
                lines.append(f"  {cls:<12} {float(values[idx]) * 100:6.2f}%  {bar}")
        lines.append("")

    if stage_info is not None:
        lines.append("WHO STAGING REFERENCE")
        lines.append(f"  Grade              : {stage_info.get('grade_label', 'N/A')}")
        lines.append(f"  Imaging features   : {stage_info.get('typical_imaging_features', 'N/A')}")
        lines.append(f"  Malignancy         : {stage_info.get('malignancy', 'N/A')}")
        lines.append(f"  Growth pattern     : {stage_info.get('growth_pattern', 'N/A')}")
        lines.append("  Recommended action : "
                     f"{stage_info.get('recommended_intervention_protocol', 'N/A')}")
        lines.append("")

    lines.append(f"Source image: {image_path or 'in-memory upload'}")
    lines.append("")
    lines.append(f"DISCLAIMER: {REGULATORY_SHORTHAND}")
    return "\n".join(lines)


def generate_diagnostic_summary(
    results: Dict[str, Any],
    image_path: Optional[str] = None,
) -> str:
    """Reasoning narrative assembled from a prediction-result dict."""
    pred_class_name = results.get("prediction", results.get("predicted_class", "Unknown"))
    conf = float(results.get("confidence", 0.0))
    probs = results.get("avg_probs", results.get("probabilities"))
    if isinstance(probs, dict):
        probs = [probs.get(c, 0.0) for c in CLASSES]
    stage_info = lookup_who_stage(pred_class_name)
    return generate_clinical_reasoning(pred_class_name, conf, probs,
                                       image_path, stage_info)


# ==============================================================================
# CONFIDENCE BANDS & LOW-CONFIDENCE SAFEGUARD
# ==============================================================================
def is_low_confidence(confidence: float, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
    """True when confidence falls below the clinical safeguard threshold."""
    return float(confidence) < threshold


def confidence_band(confidence: float) -> str:
    """Map confidence to a clinical reporting band."""
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.75:
        return "Moderate"
    if confidence >= 0.60:
        return "Low"
    return "Very Low"


# ==============================================================================
# END-TO-END SINGLE-IMAGE PIPELINE
# ==============================================================================
def run_inference_on_image(
    model: nn.Module,
    image_source: Union[str, Path, Image.Image, np.ndarray],
    device: torch.device,
    use_tta: bool = True,
    use_hog: bool = False,
    hog_extractor: Optional[HOGExtractor] = None,
    pca: Optional[Any] = None,
    gradcam: bool = True,
    gradcam_output_path: Optional[Path] = None,
    class_idx_override: Optional[int] = None,
    temperature: float = TTA_SOFTMAX_TEMP,
) -> Dict[str, Any]:
    """Preprocess -> TTA predict -> Grad-CAM++ -> reasoning for one image."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    image_tensor = preprocess_image_for_inference(image_source, as_tensor=True)
    assert isinstance(image_tensor, Tensor)

    # ---- HOG branch ----------------------------------------------------------
    hog_pca_tensor: Optional[Tensor] = None
    if use_hog and hog_extractor is not None:
        gray = img_tensor_to_numpy_gray(image_tensor)
        hog_raw = hog_extractor.extract(gray)
        expected = int(getattr(model, "hog_pca_components", 288))
        if pca is not None:
            hog_proj = pca.transform(hog_raw.reshape(1, -1)).ravel()
            hog_proj = np.asarray(hog_proj, dtype=np.float32)
            if hog_proj.shape[0] == expected:
                hog_pca_tensor = torch.from_numpy(hog_proj)
            else:
                logger.warning(
                    "HOG-PCA produced %d dims, model expects %d; HOG branch disabled.",
                    hog_proj.shape[0], expected,
                )
        elif hog_raw.shape[0] == expected:
            hog_pca_tensor = torch.from_numpy(hog_raw.astype(np.float32))
        else:
            logger.info(
                "No PCA fitted and raw HOG is %d-d (model expects %d-d); "
                "running on the deep branch only.",
                hog_raw.shape[0], expected,
            )

    # ---- prediction ------------------------------------------------------------
    if use_tta:
        predictor = TTAPredictor(
            model, device,
            transforms=build_tta_transforms(),
            softmax_temp=temperature,
            use_hog=use_hog, hog_extractor=hog_extractor, pca=pca,
        )
        pred = predictor.predict(image_tensor, hog_pca=hog_pca_tensor)
    else:
        batch = image_tensor.unsqueeze(0).to(device, dtype=next(model.parameters()).dtype)
        hog_batch = None if hog_pca_tensor is None else hog_pca_tensor.unsqueeze(0)
        with torch.inference_mode():
            out = model(batch, hog_pca=hog_batch)
            logits = out["logits"].squeeze(0).float()
            probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=0)
        decision = clinical_decision(probs)
        pred = {
            "avg_logits": logits,
            "avg_probs": probs,
            "pred_class": int(decision["pred_class"]),
            "confidence": float(probs[decision["pred_class"]].item()),
            **tumor_safety_alert(probs),
            "threshold_override": decision["threshold_override"],
            "per_transform_probs": [probs],
            "per_transform_logits": [logits],
        }

    # ---- Grad-CAM++ ------------------------------------------------------------
    overlay_pil, heatmap_tensor = None, None
    if gradcam:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        target = class_idx_override if class_idx_override is not None else pred["pred_class"]
        overlay_pil, heatmap_tensor = render_gradcam_overlay(
            model, image_tensor, class_idx=target, device=device,
            output_path=Path(gradcam_output_path) if gradcam_output_path else None,
        )

    prediction_name = CLASSES[pred["pred_class"]]
    stage_info = lookup_who_stage(prediction_name)
    reasoning = generate_clinical_reasoning(
        prediction_name, pred["confidence"], pred["avg_probs"],
        str(image_source) if isinstance(image_source, (str, Path)) else None,
        stage_info,
    )

    return {
        "prediction": prediction_name,
        "confidence": pred["confidence"],
        "pred_class_idx": pred["pred_class"],
        "potential_tumor": bool(pred.get("potential_tumor", False)),
        "tumor_probability": float(pred.get("tumor_probability", 0.0)),
        "safety_alert": pred.get("safety_alert", ""),
        "low_confidence": is_low_confidence(pred["confidence"]),
        "confidence_band": confidence_band(pred["confidence"]),
        "probabilities": pred["avg_probs"].detach().cpu().tolist(),
        "logits": pred["avg_logits"].detach().cpu().tolist(),
        "tta_used": use_tta,
        "gradcam_overlay": overlay_pil,
        "heatmap": heatmap_tensor,
        "stage_info": stage_info,
        "reasoning_text": reasoning,
        "image_source": (str(image_source) if isinstance(image_source, (str, Path))
                         else "in-memory"),
    }


# ==============================================================================
# BATCH INFERENCE
# ==============================================================================
def run_batch_inference(
    model: nn.Module,
    image_sources: Sequence[Union[str, Path, Image.Image, np.ndarray]],
    device: torch.device,
    batch_size: int = 8,
    use_tta: bool = True,
    use_hog: bool = False,
    hog_extractor: Optional[HOGExtractor] = None,
    pca: Optional[Any] = None,
    progress_desc: str = "Batch inference",
) -> List[Dict[str, Any]]:
    """Run inference over many image sources with optional batching."""
    model.eval()
    results: List[Dict[str, Any]] = []
    total = len(image_sources)

    for start in range(0, total, batch_size):
        chunk = list(image_sources[start:start + batch_size])
        logger.info("%s: batch %d/%d",
                    progress_desc, start // batch_size + 1,
                    (total + batch_size - 1) // batch_size)
        for src in chunk:
            results.append(run_inference_on_image(
                model, src, device,
                use_tta=use_tta, use_hog=use_hog,
                hog_extractor=hog_extractor, pca=pca,
                gradcam=False,
            ))
    return results


# ==============================================================================
# MULTI-MODAL ENGINE (T1c / T2 / FLAIR) — DASHBOARD & CLI BACKEND
# ==============================================================================
#: Canonical modality keys accepted by `FreeBuffInference.predict_uploads`.
MODALITIES: Tuple[str, ...] = ("t1ce", "t2", "flair")

#: Version reported in result payloads and PDF reports.
ENGINE_MODEL_VERSION: str = "4.0"


class FreeBuffInference:
    """Multi-sequence inference engine.

    Wraps the PyTorch `FreeBuffV4` model and exposes:
      * `from_dir(models_dir)` — load the engine from a checkpoint folder
      * `preprocess_path(path)` — file -> model-ready numpy image
      * `predict_uploads({modality: np.ndarray})` — full prediction payload
      * `gradcam_maps` / `overlay` — per-modality explainability

    The three input sequences are averaged into one composite slice before
    classification, which lets a single 2D classifier serve the multi-modal
    protocol; missing sequences are replicated from a present one (logged,
    never silent).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Sequence[str] = CLASSES,
        model_version: str = ENGINE_MODEL_VERSION,
        use_hog: bool = True,
        hog_extractor: Optional[HOGExtractor] = None,
        pca: Optional[Any] = None,
        softmax_temp: float = TTA_SOFTMAX_TEMP,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.class_names = list(class_names)
        self.model_version = model_version
        self.use_hog = use_hog
        self.hog_extractor = hog_extractor or (HOGExtractor() if use_hog else None)
        self.pca = pca
        self.softmax_temp = max(float(softmax_temp), 1e-4)
        #: Path of the checkpoint actually served by this engine instance
        #: (resolved by from_dir). Displayed by the dashboard so a stale-
        #: artifact situation is visible at a glance.
        self.checkpoint_path: Optional[str] = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        self.manifest: Dict[str, Any] = {"artifact_version": 4,
                                         "class_names": self.class_names}

    # -------------------------------------------------------------- loader
    @classmethod
    def from_dir(
        cls,
        models_dir: Union[str, Path] = DEFAULT_MODEL_SAVE_PATH.parent,
        device: Optional[torch.device] = None,
        checkpoint_name: str = "freebuff_v4.pt",
        pca_name: str = "pca.pkl",
    ) -> "FreeBuffInference":
        """Load the engine from `models_dir` (checkpoint + optional PCA)."""
        models_dir = Path(models_dir)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # The checkpoint may live in models_dir itself or in checkpoints/.
        candidates = [
            models_dir / checkpoint_name,
            models_dir.parent / "checkpoints" / checkpoint_name,
            DEFAULT_MODEL_SAVE_PATH,
        ]
        checkpoint_path = next((c for c in candidates if Path(c).exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint named {checkpoint_name!r} found in {models_dir} "
                f"(searched: {[str(c) for c in candidates]})"
            )

        # Build the model WITHOUT pretrained download (weights come from file),
        # then load with the error-proof loader (legacy keys + shape matching).
        model = build_engine_model(device=device, pretrained=False)
        load_checkpoint(checkpoint_path, model, device=device, strict=False,
                        verbose=True)
        self_resolved = Path(checkpoint_path)
        logger.info("Engine serving checkpoint: %s", self_resolved)

        # Optional PCA artifact for the HOG branch. Prefer the current-format
        # artifact written by train.py (freebuff_pca.pkl), then fall back to
        # any legacy artifact shipped alongside the checkpoint.
        # FIX: probe the HOG extractor dimension ONCE, up front, and sort the
        # candidates to prefer compatible artifacts. Previously the fixed order
        # tried legacy v3 `pca.pkl` (6144-d deep-extractor features) BEFORE
        # `freebuff_pca.pkl` (2048-d HOG); the shape-guard rejected the legacy
        # artifact but the error-path semantics made the intended artifact
        # fragile to ordering. Compatibility-first ordering is deterministic.
        probe: Optional[np.ndarray] = None
        try:
            probe = HOGExtractor().extract(
                np.zeros(INPUT_SIZE, dtype=np.float32))
        except Exception:
            probe = None
        probe_dim = int(probe.shape[0]) if probe is not None else None

        def _pca_compatible(candidate_name: str) -> int:
            """Rank artifacts: 0 = known-compatible, 1 = unknown, 2 = incompatible."""
            if probe_dim is None:
                return 1  # cannot verify — treat as neutral
            try:
                import joblib
                candidate = joblib.load(models_dir / candidate_name)
            except Exception:
                return 2
            expected_in = getattr(candidate, "n_features_in_", None)
            if expected_in is None:
                return 1
            return 0 if int(expected_in) == probe_dim else 2

        ranked = sorted(
            dict.fromkeys((pca_name, "freebuff_pca.pkl", "pca.pkl")),
            key=_pca_compatible,
        )
        pca = None
        for candidate_name in ranked:
            pca_path = models_dir / candidate_name
            if not pca_path.exists():
                continue
            try:
                import joblib
                candidate = joblib.load(pca_path)
            except Exception as exc:
                logger.warning("PCA artifact %s unreadable (%s).", pca_path, exc)
                continue
            # Validate the artifact against the live HOG extractor dimension.
            expected_in = getattr(candidate, "n_features_in_", None)
            if (expected_in is not None and probe is not None
                    and int(expected_in) != int(probe.shape[0])):
                logger.warning(
                    "PCA artifact %s expects %d features but the current HOG "
                    "extractor produces %d — incompatible (retrain via train.py "
                    "to refresh). Running on the deep branch only.",
                    candidate_name, int(expected_in), int(probe.shape[0]),
                )
                continue
            pca = candidate
            logger.info("HOG-PCA transformer loaded from %s", pca_path)
            break

        return cls(model=model, device=device, pca=pca,
                   checkpoint_path=self_resolved)

    # --------------------------------------------------------- preprocessing
    @staticmethod
    def preprocess_path(path: Union[str, Path], size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
        """File path -> (H, W, 3) float32 RGB array in [0, 1] (engine contract)."""
        tensor = preprocess_image_for_inference(path, size=size, as_tensor=True)
        assert isinstance(tensor, Tensor)
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        # Undo ImageNet normalization for a display-friendly [0, 1] image.
        mean = np.asarray(NORMALIZE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(NORMALIZE_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = np.clip(arr * std + mean, 0.0, 1.0).astype(np.float32)
        return arr

    # -------------------------------------------------------------- internals
    def _compose_input(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Average provided modalities into one composite (H, W, 3) image.

        Missing modalities are replicated from the first present one (with a
        log line). Returns (composite_image, list_of_modalities_used).
        """
        present: Dict[str, np.ndarray] = {
            m: np.asarray(a, dtype=np.float32)
            for m, a in uploads.items() if a is not None
        }
        if not present:
            raise ValueError("predict_uploads received no usable image data.")

        used: List[str] = []
        stack: List[np.ndarray] = []
        reference: Optional[np.ndarray] = None
        for modality in MODALITIES:
            arr = present.get(modality)
            if arr is None:
                if reference is None:
                    continue
                arr = reference
                logger.info("Modality %s missing — replicated from %s.",
                            modality, used[0])
            if reference is None:
                reference = arr
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            stack.append(arr)
            used.append(modality)

        # Pad to exactly 3 channels if fewer modalities were provided.
        while len(stack) < 3:
            stack.append(stack[-1])
            if used:
                logger.info("Composite padded to 3 channels using %s.", used[-1])

        composite = np.mean(stack[:3], axis=0).astype(np.float32)
        return composite, used

    def _hog_for(self, composite: np.ndarray) -> Optional[Tensor]:
        """PCA-projected HOG tensor for the composite, or None.

        Dimension guard: the HOG branch expects `model.hog_pca_components`
        (288) features. Raw HOG is 2048-d, so without a fitted PCA artifact
        the projection is impossible — we return None (the model then uses
        its zero-embedding fallback) rather than crash with a shape error.
        """
        if not self.use_hog or self.hog_extractor is None:
            return None
        gray = composite.mean(axis=-1)  # (H, W) in [0, 1]
        try:
            raw = self.hog_extractor.extract(gray.astype(np.float32))
        except Exception as exc:
            logger.warning("HOG extraction failed (%s); continuing without it.", exc)
            return None
        expected = int(getattr(self.model, "hog_pca_components", HOG_PCA_COMPONENTS))
        if self.pca is not None:
            try:
                proj = self.pca.transform(raw.reshape(1, -1)).ravel()
                proj = np.asarray(proj, dtype=np.float32)
                if proj.shape[0] != expected:
                    logger.warning(
                        "HOG-PCA produced %d dims but the model expects %d; "
                        "disabling the HOG branch for this run.",
                        proj.shape[0], expected,
                    )
                    return None
                return torch.from_numpy(proj)
            except Exception as exc:
                logger.warning("HOG-PCA projection failed (%s); branch disabled.", exc)
                return None
        # No PCA artifact: only pass raw HOG through if the dims happen to match.
        if raw.shape[0] == expected:
            return torch.from_numpy(raw.astype(np.float32))
        logger.info(
            "No PCA artifact available and raw HOG is %d-d (model expects %d-d); "
            "running on the deep branch only.",
            raw.shape[0], expected,
        )
        return None

    # ------------------------------------------------------------- prediction
    def predict_uploads(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
        use_tta: bool = True,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full multi-modal prediction payload.

        Returns a dict with (superset of) keys: predicted_class, confidence,
        probabilities ({class: float}), staging, model_version, modalities,
        patient_id, date, low_confidence, safety_alert, tumor_probability,
        reasoning_text.
        """
        self.model.eval()
        composite, used = self._compose_input(uploads)

        tensor = torch.from_numpy(composite.transpose(2, 0, 1)).unsqueeze(0)
        # Re-apply ImageNet normalization for the model.
        mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)

        hog_tensor = self._hog_for(composite)

        if use_tta:
            predictor = TTAPredictor(
                self.model, self.device,
                transforms=build_tta_transforms(),
                softmax_temp=self.softmax_temp,
                use_hog=self.use_hog,
                hog_extractor=self.hog_extractor, pca=self.pca,
            )
            pred = predictor.predict(tensor[0], hog_pca=hog_tensor)
        else:
            hog_batch = None if hog_tensor is None else hog_tensor.unsqueeze(0)
            with torch.inference_mode():
                out = self.model(tensor, hog_pca=hog_batch)
                logits = out["logits"].squeeze(0).float()
                probs = torch.softmax(logits / self.softmax_temp, dim=0)
            decision = clinical_decision(probs)
            pred = {
                "avg_probs": probs,
                "pred_class": int(decision["pred_class"]),
                "confidence": float(probs[decision["pred_class"]].item()),
                **tumor_safety_alert(probs),
            }

        predicted_name = self.class_names[pred["pred_class"]]
        probabilities = {
            self.class_names[i]: float(p)
            for i, p in enumerate(pred["avg_probs"].detach().cpu().tolist())
        }

        # WHO staging via the dedicated reference engine.
        from staging import interpret_prediction
        staging = interpret_prediction(
            predicted_class=_staging_key(predicted_name),
            confidence=pred["confidence"],
            probabilities=probabilities,
        )

        reasoning = generate_clinical_reasoning(
            predicted_name, pred["confidence"],
            pred["avg_probs"], None, lookup_who_stage(predicted_name),
        )

        return {
            "predicted_class": predicted_name,
            "confidence": float(pred["confidence"]),
            "probabilities": probabilities,
            "staging": staging,
            "model_version": self.model_version,
            "modalities": used,
            "patient_id": patient_id,
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "low_confidence": is_low_confidence(pred["confidence"]),
            "confidence_band": confidence_band(pred["confidence"]),
            "safety_alert": pred.get("safety_alert", ""),
            "tumor_probability": float(pred.get("tumor_probability", 0.0)),
            "potential_tumor": bool(pred.get("potential_tumor", False)),
            "reasoning_text": reasoning,
        }

    # ------------------------------------------------------------ explainers
    def gradcam_maps(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
        class_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Per-modality Grad-CAM++ maps ({modality: (H, W) float in [0, 1]})."""
        self.model.eval()
        maps: Dict[str, np.ndarray] = {}
        for modality, arr in uploads.items():
            if arr is None:
                continue
            img = np.asarray(arr, dtype=np.float32)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1).to(self.device)
            tensor = (tensor - mean) / std

            engine = GradCAMPlusPlus(self.model,
                                     target_layer_name=GRADCAM_TARGET_LAYER_NAME,
                                     device=self.device)
            try:
                heatmap, _ = engine.generate_heatmap(tensor, class_idx=class_idx)
                maps[modality] = (heatmap.squeeze().detach().cpu().numpy()
                                  .astype(np.float32))
            except Exception as exc:
                logger.warning("Grad-CAM++ failed for %s: %s", modality, exc)
                maps[modality] = np.zeros((tensor.shape[-2], tensor.shape[-1]),
                                          dtype=np.float32)
            finally:
                engine.remove_hooks()
        return maps

    def score_cam_maps(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
        class_idx: Optional[int] = None,
        max_channels: int = 64,
    ) -> Dict[str, np.ndarray]:
        """Per-modality Score-CAM maps (gradient-free masking baseline).

        Each target-layer channel map is upsampled to the input resolution,
        normalized into a mask, and the drop/increase in the target class
        score under that mask weights the channel's contribution. A
        deterministic even subsample of channels bounds the cost.
        """
        self.model.eval()
        maps: Dict[str, np.ndarray] = {}
        for modality, arr in uploads.items():
            if arr is None:
                continue
            try:
                maps[modality] = self._score_cam_single(
                    arr, class_idx=class_idx, max_channels=max_channels)
            except Exception as exc:
                logger.warning("Score-CAM failed for %s: %s", modality, exc)
                maps[modality] = np.zeros(INPUT_SIZE, dtype=np.float32)
        return maps

    def _score_cam_single(
        self,
        arr: np.ndarray,
        class_idx: Optional[int],
        max_channels: int,
    ) -> np.ndarray:
        """Score-CAM for one image (see `score_cam_maps`)."""
        img = np.asarray(arr, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1).to(self.device)
        tensor = (tensor - mean) / std

        # Capture target-layer activations (forward-only hook).
        target = self.model.get_gradcam_target_layer() \
            if hasattr(self.model, "get_gradcam_target_layer") else None
        if target is None:
            for name, module in self.model.named_modules():
                if name.endswith("." + GRADCAM_TARGET_LAYER_NAME):
                    target = module
                    break
        if target is None:
            raise RuntimeError("Score-CAM: no target layer found.")

        captured: Dict[str, Tensor] = {}

        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            captured["act"] = out.detach()

        handle = target.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                base_out = self.model(tensor)
                logits = base_out["logits"] if isinstance(base_out, dict) else base_out
                if class_idx is None:
                    class_idx = int(logits.argmax(dim=1).item())
                base_score = float(logits[0, class_idx].item())
                activations = captured["act"]  # (1, C, h, w)
        finally:
            handle.remove()

        c = activations.shape[1]
        channel_ids = (np.linspace(0, c - 1, min(max_channels, c))
                       .astype(int).tolist())
        h_in, w_in = tensor.shape[-2], tensor.shape[-1]

        weights: List[float] = []
        masks_up: List[np.ndarray] = []
        with torch.inference_mode():
            for ch in channel_ids:
                act_ch = activations[:, ch:ch + 1]
                mask = F.interpolate(act_ch, size=(h_in, w_in), mode="bilinear",
                                     align_corners=False)
                m = mask.squeeze().cpu().numpy()
                m_min, m_max = float(m.min()), float(m.max())
                if m_max - m_min < 1e-6:
                    weights.append(0.0)
                    masks_up.append(np.zeros((h_in, w_in), dtype=np.float32))
                    continue
                m = (m - m_min) / (m_max - m_min)
                masks_up.append(m.astype(np.float32))
                masked = tensor * torch.from_numpy(m).view(1, 1, h_in, w_in).to(self.device)
                out_m = self.model(masked)
                logits_m = out_m["logits"] if isinstance(out_m, dict) else out_m
                weights.append(float(logits_m[0, class_idx].item()) - base_score)

        w_arr = np.asarray(weights, dtype=np.float32)
        # Softmax over score deltas (positive contributions dominate).
        w_exp = np.exp(w_arr - w_arr.max())
        w_arr = w_exp / max(float(w_exp.sum()), 1e-8)

        heatmap = np.zeros((h_in, w_in), dtype=np.float32)
        for weight, mask in zip(w_arr, masks_up):
            heatmap += weight * mask
        heatmap = np.clip(heatmap, 0.0, None)
        peak = float(heatmap.max())
        if peak > 1e-6:
            heatmap = heatmap / peak
        return heatmap.astype(np.float32)

    def attention_rollout_maps(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
        class_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Per-modality CBAM spatial-attention maps.

        For a pure-CNN architecture the analogue of transformer attention
        rollout is the learned CBAM spatial attention map — captured directly
        from `backbone.cbam.spatial_att` with a forward-only hook.
        """
        self.model.eval()
        maps: Dict[str, np.ndarray] = {}
        # CBAM lives at backbone.cbam. SpatialAttention.forward returns the
        # REFINED features (B, 2048, h, w); its inner `conv` stack (Conv -> BN
        # -> Sigmoid) outputs the actual (B, 1, h, w) attention map, so hook
        # the conv when available.
        cbam = getattr(getattr(self.model, "backbone", None), "cbam", None)
        spatial_att = getattr(cbam, "spatial_att", None)
        if spatial_att is not None and hasattr(spatial_att, "conv"):
            hook_target = spatial_att.conv
        else:
            hook_target = spatial_att
        if spatial_att is None:
            logger.warning("attention_rollout_maps: CBAM spatial attention "
                           "module not found; returning empty maps.")
            return {m: np.zeros(INPUT_SIZE, dtype=np.float32)
                    for m, a in uploads.items() if a is not None}

        captured: Dict[str, Tensor] = {}

        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            t = out if isinstance(out, Tensor) else out[0]
            captured["att"] = t.detach()

        handle = hook_target.register_forward_hook(hook)
        try:
            for modality, arr in uploads.items():
                if arr is None:
                    continue
                img = np.asarray(arr, dtype=np.float32)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1).to(self.device)
                tensor = ((tensor - mean) / std).to(self.device)
                try:
                    with torch.inference_mode():
                        self.model(tensor)
                    att = captured.get("att")
                    if att is None:
                        raise RuntimeError("attention hook captured nothing")
                    att_up = F.interpolate(att, size=(tensor.shape[-2], tensor.shape[-1]),
                                           mode="bilinear", align_corners=False)
                    m = att_up.squeeze().detach().cpu().numpy().astype(np.float32)
                    m_min, m_max = float(m.min()), float(m.max())
                    if m_max - m_min > 1e-6:
                        m = (m - m_min) / (m_max - m_min)
                    maps[modality] = m
                except Exception as exc:
                    logger.warning("attention_rollout failed for %s: %s", modality, exc)
                    maps[modality] = np.zeros((tensor.shape[-2], tensor.shape[-1]),
                                              dtype=np.float32)
        finally:
            handle.remove()
        return maps

    def explain_maps(
        self,
        uploads: Dict[str, Optional[np.ndarray]],
        class_idx: Optional[int] = None,
        mode: str = "auto",
    ) -> Dict[str, np.ndarray]:
        """Dispatch explainability by mode.

        ``auto`` prefers Grad-CAM++ (well-defined for this architecture) and
        falls back to Score-CAM if gradient computation is unavailable.
        """
        if mode in ("grad_cam", "auto"):
            try:
                return self.gradcam_maps(uploads, class_idx=class_idx)
            except Exception as exc:
                if mode == "grad_cam":
                    raise
                logger.warning("Grad-CAM++ unavailable (%s); falling back to "
                               "Score-CAM.", exc)
        if mode in ("score_cam", "auto"):
            return self.score_cam_maps(uploads, class_idx=class_idx)
        if mode == "attention_rollout":
            return self.attention_rollout_maps(uploads, class_idx=class_idx)
        raise ValueError(f"Unknown explainability mode: {mode!r}")

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = GRADCAM_ALPHA,
        cmap: int = 2,  # unused when using matplotlib; kept for API compat
    ) -> np.ndarray:
        """Blend a [0,1] heatmap onto an [0,1] RGB image (jet colormap)."""
        import matplotlib

        try:
            colormap = matplotlib.colormaps[GRADCAM_COLORMAP]
        except (AttributeError, KeyError):  # pragma: no cover
            from matplotlib import pyplot as plt
            colormap = plt.get_cmap(GRADCAM_COLORMAP)

        base = np.clip(image, 0.0, 1.0).astype(np.float32)
        hm = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
        # Shape normalisation: guarantee a 3-channel (H, W, 3) base so the
        # blend always broadcasts (uploads may arrive as (H, W) grayscale).
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        elif base.ndim == 3 and base.shape[-1] == 1:
            base = np.repeat(base, 3, axis=-1)
        elif base.ndim == 3 and base.shape[-1] > 3:
            base = base[..., :3]
        if hm.shape != base.shape[:2]:
            hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize(
                (base.shape[1], base.shape[0]))
            hm = np.asarray(hm_pil, dtype=np.float32) / 255.0
        colored = colormap(hm)[..., :3].astype(np.float32)
        return np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)


def _staging_key(predicted_name: str) -> str:
    """Map a CLASSES display name onto the staging engine's profile keys."""
    lowered = predicted_name.lower().replace("_", "").replace("-", "")
    mapping = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
        "notumor": "notumor",
    }
    return mapping.get(lowered, lowered)


def build_engine_model(
    device: torch.device,
    pretrained: bool = False,
) -> FreeBuffV4:
    """Construct the FreeBuffV4 model for the inference engine (no download)."""
    from model import build_freebuff_v4

    return build_freebuff_v4(num_classes=NUM_CLASSES, device=device,
                             pretrained=pretrained)


# ==============================================================================
# MODULE SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    set_seeds = __import__("config").set_all_seeds
    set_seeds(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_engine_model(device=device, pretrained=False)
    model.eval()

    dummy = np.random.rand(144, 144, 3).astype(np.float32)
    uploads = {"t1ce": dummy, "t2": dummy, "flair": dummy}

    t0 = time.time()
    result = FreeBuffInference(model, device).predict_uploads(uploads, use_tta=True)
    elapsed = time.time() - t0

    print("=" * 64)
    print("FreeBuff v4 inference self-test")
    print("=" * 64)
    print(f"Predicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence'] * 100:.1f}%")
    print(f"Band            : {result['confidence_band']}")
    print(f"Modalities used : {result['modalities']}")
    print(f"Elapsed         : {elapsed:.2f}s (TTA)")
    print("Probabilities   :")
    for cls, p in result["probabilities"].items():
        print(f"  {cls:<12} {p * 100:6.2f}%")
    print("=" * 64)
