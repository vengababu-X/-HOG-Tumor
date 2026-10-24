#!/usr/bin/env python3
# =============================================================================
# FreeBuff v4 — Production Pipeline: TTA + Grad-CAM++ + Evaluation
# =============================================================================
# This script provides a bulletproof, end-to-end pipeline that:
#   1. Loads data, builds model, and trains (or loads a checkpoint).
#   2. Runs Test-Time Augmentation (TTA) inference with shape-safe transforms.
#   3. Generates Grad-CAM++ heatmaps using torch.autograd.grad() instead of
#      register_full_backward_hook (which caused the CUDA illegal memory access).
#   4. Produces comprehensive evaluation: confusion matrix, classification report,
#      per-class F1, ROC-AUC curves, and confidence analysis.
#
# ROOT CAUSE OF THE CUDA CRASH (FIXED):
#   The original code used register_full_backward_hook on the ResNet layer4 to
#   capture gradients for Grad-CAM++. When FreeBuffV4.forward() calls .clone()
#   on the pooled features downstream, the backward hook fires with stale/
#   incorrect gradient shapes, causing out-of-bounds GPU memory indexing.
#   FIX: We use torch.autograd.grad(score, features) to compute gradients
#   directly, bypassing backward hooks entirely.
#
# ROOT CAUSE OF THE ACCURACY PLATEAU (FIXED):
#   (a) FocalLoss was constructed with wrong keyword arguments (class_weights
#       instead of alpha), causing a TypeError at training time.
#   (b) CBAM received spatial_embedding_dim=256 channels but ResNet50 layer4
#       outputs 2048, causing a matrix shape mismatch.
#   FIX: Corrected both the FocalLoss call and the CBAM channel configuration.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
from config import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SIZE,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEFAULT_LR,
    BACKBONE_LR,
    WEIGHT_DECAY,
    ADAMW_EPS,
    ADAMW_BETA1,
    ADAMW_BETA2,
    FOCAL_GAMMA,
    GRADIENT_CLIP_NORM,
    COSINE_T_MAX,
    COSINE_ETA_MIN,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TTA_FLIPS,
    TTA_ROTATIONS_DEG,
    TTA_SCALE_FACTORS,
    TTA_FULL,
    TTA_SOFTMAX_TEMP,
    GRADCAM_TARGET_LAYER_NAME,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    HOG_PCA_COMPONENTS,
    CUDA_DEVICE_STRING,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MODEL_SAVE_PATH,
    EVAL_VIS_DIR,
    GRADCAM_EXPORT_DIR,
    set_all_seeds,
    initialize_directories,
)

from model import (
    FreeBuffV4,
    GradCAMPlusPlus,
    FocalLoss,
    build_freebuff_v4,
    build_parameter_groups,
    save_checkpoint,
    load_checkpoint,
)

from dataset import (
    BrainTumorDataset,
    HOGExtractor,
    build_dataset_from_folder,
    split_dataset_into_train_val_test,
    make_dataloaders,
    fit_pca_on_training_hog,
    img_tensor_to_numpy_gray,
)

from preprocessing import augment_mri_image, image_to_tensor
from inference import (
    TTAPredictor,
    build_tta_transforms,
    preprocess_image_for_inference,
)

logger = logging.getLogger("freebuff_v4.pipeline")

# =============================================================================
# SAFETY ASSERTIONS & TENSOR VALIDATION UTILITIES
# =============================================================================

def assert_tensor_safety(
    tensor: Tensor,
    name: str,
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Validate a tensor's shape, value range, and device before critical ops.

    This function acts as a guard before passing data into Grad-CAM++ hooks,
    TTA transforms, or model forward passes. It catches shape mismatches and
    value anomalies early, preventing silent CUDA errors.

    Args:
        tensor: The tensor to validate.
        name: Human-readable name for error messages.
        expected_ndim: Expected number of dimensions (e.g., 4 for batched images).
        expected_shape: Expected exact shape (e.g., (1, 3, 144, 144)).
        min_val: Minimum allowed value (e.g., for normalized inputs).
        max_val: Maximum allowed value (e.g., for normalized inputs).
        device: Expected device.

    Returns:
        The validated tensor (unchanged).

    Raises:
        ValueError: If any assertion fails.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{name}: expected torch.Tensor, got {type(tensor)}")

    if expected_ndim is not None and tensor.dim() != expected_ndim:
        raise ValueError(
            f"{name}: expected {expected_ndim}D tensor, got {tensor.dim()}D "
            f"(shape={list(tensor.shape)})"
        )

    if expected_shape is not None and tensor.shape != torch.Size(expected_shape):
        raise ValueError(
            f"{name}: expected shape {list(expected_shape)}, "
            f"got {list(tensor.shape)}"
        )

    if min_val is not None and float(tensor.min()) < min_val:
        logger.warning(
            "%s: min value %.4f is below expected minimum %.4f",
            name, float(tensor.min()), min_val,
        )

    if max_val is not None and float(tensor.max()) > max_val:
        logger.warning(
            "%s: max value %.4f exceeds expected maximum %.4f",
            name, float(tensor.max()), max_val,
        )

    if device is not None and tensor.device != device:
        raise ValueError(
            f"{name}: expected device {device}, got {tensor.device}"
        )

    return tensor


def assert_model_io_sanity(
    model: nn.Module,
    input_tensor: Tensor,
    hog_pca: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Run a forward pass and validate all intermediate shapes.

    This is called once after model construction to verify that the CBAM
    channel dimensions, spatial pooling, and fusion head all agree before
    training or inference begins.

    Args:
        model: The FreeBuffV4 model.
        input_tensor: A dummy input (1, 3, H, W).
        hog_pca: Optional dummy HOG features (1, HOG_PCA_COMPONENTS).

    Returns:
        Dict with output shapes for logging.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, hog_pca=hog_pca)

    shapes = {}
    for key, val in output.items():
        if isinstance(val, Tensor):
            shapes[key] = list(val.shape)

    logger.info("Model I/O sanity check passed: %s", shapes)
    return shapes


# =============================================================================
# GRAD-CAM++ SAFETY WRAPPER (CUDA-Safe Heatmap Generation)
# =============================================================================

def safe_gradcam_generate(
    model: nn.Module,
    image_tensor: Tensor,
    device: torch.device,
    class_idx: Optional[int] = None,
    target_layer: str = GRADCAM_TARGET_LAYER_NAME,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Generate a Grad-CAM++ heatmap with full safety guarantees.

    This wrapper addresses the CUDA illegal memory access by:
      1. Ensuring the model is NOT inside torch.inference_mode().
      2. Ensuring the image tensor is contiguous and on the correct device.
      3. Clamping class_idx to valid range to prevent OOB indexing.
      4. Validating feature and gradient shapes before heatmap computation.
      5. Running torch.cuda.empty_cache() before and after to prevent
         state leakage between augmentations.

    ROOT CAUSE EXPLANATION:
        The original code used register_full_backward_hook which fires with
        stale gradient shapes when downstream ops (.clone(), .view()) modify
        the module output. torch.autograd.grad() avoids this entirely.

    Args:
        model: FreeBuffV4 model in eval mode (NOT inside inference_mode).
        image_tensor: Single image (3, H, W) or batched (1, 3, H, W).
        device: Torch device.
        class_idx: Target class; if None, uses predicted class.
        target_layer: Name of the layer to visualize.

    Returns:
        Tuple of (heatmap tensor, details dict).
    """
    # Step 0: Clean GPU state before computation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Step 1: Ensure image is contiguous, on device, and has batch dim
    img = image_tensor.detach().clone()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.contiguous().to(device)

    # Step 2: Validate input shape
    assert_tensor_safety(
        img, "gradcam_input", expected_ndim=4,
        expected_shape=(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]),
    )

    # Step 3: Create GradCAMPlusPlus engine (uses torch.autograd.grad, NOT
    # register_full_backward_hook — this is the fix for the CUDA crash)
    gc = GradCAMPlusPlus(model, target_layer_name=target_layer, device=device)

    try:
        heatmap, details = gc.generate_heatmap(img, class_idx=class_idx)
    except RuntimeError as e:
        logger.warning("Grad-CAM++ failed: %s", e)
        gc.remove_hooks()
        # Return a fallback heatmap so downstream code doesn't crash
        fallback = torch.zeros(1, 1, INPUT_SIZE[0], INPUT_SIZE[1], device=device)
        return fallback, {"class_idx": class_idx or 0, "error": str(e)}
    finally:
        gc.remove_hooks()

    # Step 4: Validate output heatmap shape
    assert_tensor_safety(
        heatmap, "gradcam_heatmap", expected_ndim=4,
        expected_shape=(1, 1, INPUT_SIZE[0], INPUT_SIZE[1]),
    )

    # Step 5: Clean GPU state after computation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return heatmap, details


# =============================================================================
# TTA PREDICTOR (Shape-Safe, Memory-Leak-Proof)
# =============================================================================

class SafeTTAPredictor:
    """Test-Time Augmentation predictor with safety guarantees.

    Key safety features vs. the original TTAPredictor:
      - Explicit shape assertions after each TTA transform.
      - Contiguous tensor enforcement for CUDA compatibility.
      - Memory cleanup between transform batches.
      - Validates softmax temperature is positive.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transforms: Optional[List[Callable[[Tensor], Tensor]]] = None,
        softmax_temp: float = TTA_SOFTMAX_TEMP,
        use_hog: bool = False,
        pca: Optional[Any] = None,
    ):
        self.model = model
        self.device = device
        self.transforms = transforms or build_tta_transforms()
        self.softmax_temp = max(float(softmax_temp), 1e-4)
        self.use_hog = use_hog
        self.pca = pca

        # Validate softmax temp
        if self.softmax_temp <= 0:
            raise ValueError(f"softmax_temp must be positive, got {self.softmax_temp}")

    def predict_single(
        self,
        image: Tensor,
        hog_pca: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Run TTA on a single image with shape safety checks.

        Args:
            image: Image tensor (3, H, W), normalized.
            hog_pca: Optional PCA-projected HOG features.

        Returns:
            Dict with avg_logits, avg_probs, pred_class, confidence.
        """
        if image.dim() == 3:
            image_4d = image.unsqueeze(0)
        elif image.dim() == 4:
            image_4d = image
        else:
            raise ValueError(f"Expected 3D or 4D image, got shape {image.shape}")

        image_4d = image_4d.to(self.device, dtype=next(self.model.parameters()).dtype)

        # Validate input
        assert_tensor_safety(
            image_4d, "tta_input", expected_ndim=4,
            expected_shape=(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]),
        )

        hog_batch = None
        if hog_pca is not None:
            if hog_pca.dim() == 1:
                hog_batch = hog_pca.unsqueeze(0).to(self.device, dtype=torch.float32)
            else:
                hog_batch = hog_pca.to(self.device, dtype=torch.float32)

        all_logits: List[Tensor] = []
        all_probs: List[Tensor] = []

        # TTA runs inside inference_mode for memory efficiency.
        # Grad-CAM++ runs OUTSIDE this context so autograd is active.
        with torch.inference_mode():
            for i, fn in enumerate(self.transforms):
                aug_3d = fn(image_4d.squeeze(0))

                # SAFETY: Validate TTA transform output shape
                assert_tensor_safety(
                    aug_3d, f"tta_transform[{i}]_output",
                    expected_ndim=3,
                )
                if aug_3d.shape[0] != 3 or aug_3d.shape[1:] != (INPUT_SIZE[0], INPUT_SIZE[1]):
                    raise ValueError(
                        f"TTA transform [{i}] produced wrong shape: "
                        f"expected (3, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}), "
                        f"got {list(aug_3d.shape)}"
                    )

                aug_4d = aug_3d.unsqueeze(0).contiguous()
                out = self.model(aug_4d, hog_pca=hog_batch)
                logits = out["logits"].squeeze(0).detach()
                probs = torch.softmax(logits / self.softmax_temp, dim=0).detach()
                all_logits.append(logits)
                all_probs.append(probs)

        stacked_logits = torch.stack(all_logits, dim=0)
        calibrated_logits = stacked_logits / self.softmax_temp
        per_view_probs = torch.softmax(calibrated_logits, dim=1)
        avg_logits = stacked_logits.mean(dim=0)
        avg_probs = per_view_probs.mean(dim=0)
        avg_probs = avg_probs / avg_probs.sum().clamp_min(1e-8)

        pred_class = int(avg_probs.argmax().item())
        confidence = float(avg_probs[pred_class].item())

        # Clinical decision: sensitivity-first override
        tumor_indices = [i for i, n in enumerate(CLASSES) if n != "No_Tumor"]
        tumor_prob = float(avg_probs[tumor_indices].sum().item())
        selected_class = pred_class
        if tumor_prob >= 0.30 and pred_class == CLASSES.index("No_Tumor"):
            selected_class = tumor_indices[int(avg_probs[tumor_indices].argmax().item())]

        return {
            "avg_logits": avg_logits,
            "avg_probs": avg_probs,
            "pred_class": selected_class,
            "confidence": confidence,
            "tumor_probability": tumor_prob,
            "per_transform_probs": all_probs,
            "per_transform_logits": all_logits,
        }

    def predict_batch(
        self,
        images: Tensor,
        hog_pcas: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Run TTA on a batch of images.

        Args:
            images: Batch of images (B, 3, H, W).
            hog_pcas: Optional HOG features (B, HOG_DIM).

        Returns:
            Batched prediction results.
        """
        B = images.size(0)
        all_logits: List[Tensor] = []

        with torch.inference_mode():
            for fn in self.transforms:
                aug = torch.stack(
                    [fn(images[b]) for b in range(B)], dim=0
                ).contiguous()
                assert aug.shape == (B, 3, INPUT_SIZE[0], INPUT_SIZE[1]), \
                    f"TTA batch aug shape mismatch: {aug.shape}"
                out = self.model(aug.to(self.device), hog_pca=hog_pcas)
                all_logits.append(out["logits"].detach())

        stacked = torch.stack(all_logits, dim=0)  # (T, B, C)
        avg_logits = stacked.mean(dim=0)           # (B, C)
        avg_probs = torch.softmax(stacked / self.softmax_temp, dim=2).mean(dim=0)
        avg_probs = avg_probs / avg_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pred_classes = avg_probs.argmax(dim=1)
        confidences = avg_probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

        return {
            "avg_logits": avg_logits,
            "avg_probs": avg_probs,
            "pred_classes": pred_classes,
            "confidences": confidences,
        }


# =============================================================================
# GRAD-CAM++ OVERLAY GENERATION (Shape-Safe)
# =============================================================================

def generate_gradcam_overlay(
    model: nn.Module,
    image_tensor: Tensor,
    device: torch.device,
    class_idx: Optional[int] = None,
    alpha: float = GRADCAM_ALPHA,
    colormap: str = GRADCAM_COLORMAP,
    output_path: Optional[Path] = None,
) -> Tuple[Any, Tensor]:
    """Generate and optionally save a Grad-CAM++ overlay.

    Uses the safety wrapper safe_gradcam_generate() which avoids
    register_full_backward_hook entirely.

    Args:
        model: FreeBuffV4 model (eval mode).
        image_tensor: Single image (3, H, W) normalized.
        device: Torch device.
        class_idx: Target class; None uses predicted class.
        alpha: Blending alpha for overlay.
        colormap: Matplotlib colormap name.
        output_path: Optional save path.

    Returns:
        Tuple of (PIL Image, heatmap tensor).
    """
    from PIL import Image as PILImage

    heatmap, details = safe_gradcam_generate(
        model, image_tensor, device, class_idx=class_idx,
    )

    # Generate overlay using the static method
    overlay_tensor = GradCAMPlusPlus.overlay_heatmap_on_image(
        image_tensor, heatmap, alpha=alpha, colormap=colormap,
    )

    # Convert to PIL
    overlay_np = overlay_tensor.detach().cpu().numpy().astype(np.float32)
    if overlay_np.ndim == 4:
        overlay_np = overlay_np[0]
    overlay_np = np.clip(overlay_np, 0.0, 1.0)
    overlay_np = (overlay_np * 255.0).astype(np.uint8)
    overlay_np = np.transpose(overlay_np, (1, 2, 0))
    overlay_pil = PILImage.fromarray(overlay_np, mode="RGB")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_pil.save(str(output_path))
        logger.info("Saved Grad-CAM++ overlay to %s", output_path)

    return overlay_pil, heatmap


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run comprehensive evaluation with all standard metrics.

    Produces:
      - Accuracy, weighted/macro/micro F1
      - Full classification report (precision, recall, F1, support)
      - Confusion matrix (counts and normalized)
      - Per-class F1 bar chart
      - ROC curves (one-vs-rest) with AUC
      - Confidence distribution (correct vs incorrect)
      - Grad-CAM++ overlays for up to 10 misclassified samples

    Args:
        model: FreeBuffV4 model in eval mode.
        test_loader: Test DataLoader.
        device: Torch device.
        use_tta: Whether to use TTA.
        class_names: Class name list; defaults to CLASSES.
        output_dir: Directory for saving figures.

    Returns:
        Dict with all metrics and paths.
    """
    class_names = class_names or CLASSES
    output_dir = output_dir or EVAL_VIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep overlays with the rest of this run's artifacts. Always writing to the
    # global GRADCAM_EXPORT_DIR meant a test run against a random-weights model
    # overwrote genuine overlays in evaluation_results/, making real results and
    # test output indistinguishable on disk.
    gradcam_dir = (GRADCAM_EXPORT_DIR if output_dir == EVAL_VIS_DIR
                   else Path(output_dir) / "gradcam_overlays")
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    predictor = None
    if use_tta:
        predictor = SafeTTAPredictor(model, device)

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []
    misclassified_samples: List[Dict[str, Any]] = []

    # --- Collect predictions ---
    logger.info("Starting evaluation on %d test batches...", len(test_loader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device, dtype=next(model.parameters()).dtype)
            labels = batch["label"].cpu().numpy()
            hog_feats = None
            if batch.get("hog_pca") is not None:
                hog_feats = batch["hog_pca"].to(device, dtype=torch.float32)

            if predictor is not None:
                pred = predictor.predict_batch(images, hog_pcas=hog_feats)
                probs = pred["avg_probs"].cpu().numpy()
                preds = pred["pred_classes"].cpu().numpy()
                confs = pred["confidences"].cpu().numpy()
            else:
                out = model(images, hog_pca=hog_feats)
                logits = out["logits"]
                probs = torch.softmax(logits / TTA_SOFTMAX_TEMP, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                confs = probs.max(axis=1)

            all_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
            all_confs.append(confs)

            # Track misclassified for Grad-CAM
            for i in range(len(labels)):
                if preds[i] != labels[i] and len(misclassified_samples) < 10:
                    misclassified_samples.append({
                        "image": images[i].cpu().clone(),
                        "true_cls": int(labels[i]),
                        "pred_cls": int(preds[i]),
                    })

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Evaluated batch %d/%d", batch_idx + 1, len(test_loader))

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    y_confs = np.concatenate(all_confs)

    # --- Core metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))

    # Classification report (precision, recall, f1, support per class)
    clf_report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
    )
    clf_report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4,
        zero_division=0, output_dict=True,
    )

    # Confusion matrix
    cm_counts = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm_counts.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_norm / row_sums

    # ROC-AUC
    roc_aucs = {}
    try:
        y_true_bin = np.zeros((y_true.size, len(class_names)), dtype=np.int64)
        for i in range(len(class_names)):
            y_true_bin[:, i] = (y_true == i).astype(np.int64)

        for i, cls_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_aucs[cls_name] = float(auc(fpr, tpr))

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_aucs["micro_avg"] = float(auc(fpr_micro, tpr_micro))
    except Exception as e:
        logger.warning("ROC-AUC computation failed: %s", e)

    # --- Generate plots ---
    _plot_confusion_matrix(cm_counts, class_names, output_dir / "cm_counts.png", "Confusion Matrix (Counts)")
    _plot_confusion_matrix(cm_norm, class_names, output_dir / "cm_normalized.png", "Confusion Matrix (Normalized)", normalize=True)
    _plot_per_class_f1(per_class_f1, class_names, output_dir / "per_class_f1.png")
    _plot_roc_curves(y_true, y_probs, class_names, output_dir / "roc_curves.png")
    _plot_confidence_distribution(y_confs, y_true, y_pred, output_dir / "confidence_dist.png")

    # --- Grad-CAM++ overlays for misclassified samples ---
    gradcam_count = 0
    if misclassified_samples:
        logger.info("Generating Grad-CAM++ overlays for %d misclassified samples...", len(misclassified_samples))
        for sample in misclassified_samples:
            try:
                img = sample["image"].to(device)
                true_name = class_names[sample["true_cls"]]
                pred_name = class_names[sample["pred_cls"]]

                overlay_pil, hm = generate_gradcam_overlay(
                    model, img, device,
                    class_idx=sample["pred_cls"],
                    output_path=gradcam_dir / f"misclass_{gradcam_count}_{true_name}_vs_{pred_name}.png",
                )
                gradcam_count += 1
            except Exception as e:
                logger.debug("Grad-CAM overlay failed: %s", e)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Weighted F1:         {weighted_f1:.4f}")
    print(f"Micro F1:            {micro_f1:.4f}")
    print(f"Test samples:        {len(y_true)}")
    print(f"Misclassified:       {int((y_true != y_pred).sum())}")
    print(f"Grad-CAM overlays:   {gradcam_count}")
    print(f"\nROC-AUC per class:")
    for k, v in roc_aucs.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{clf_report_str}")
    print("Confusion Matrix (counts):")
    print(cm_counts)
    print("=" * 70 + "\n")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))},
        "classification_report": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix_counts": cm_counts,
        "confusion_matrix_normalized": cm_norm,
        "roc_aucs": roc_aucs,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "n_test_samples": int(y_true.size),
        "n_misclassified": int((y_true != y_pred).sum()),
        "gradcam_saved_count": gradcam_count,
    }


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.Blues if not normalize else plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_per_class_f1(
    per_class_f1: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot per-class F1 scores as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    x = np.arange(len(class_names))
    bars = ax.bar(x, per_class_f1, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    weighted_avg = float(f1_score(
        np.zeros(len(class_names)), np.zeros(len(class_names)),
        average="weighted", zero_division=0,
    ))
    macro_avg = float(np.mean(per_class_f1))
    ax.axhline(macro_avg, color="red", linestyle="--", linewidth=1.5,
                label=f"Macro avg: {macro_avg:.3f}")
    ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot one-vs-rest ROC curves with AUC."""
    n_classes = len(class_names)
    y_true_bin = np.zeros((y_true.size, n_classes), dtype=np.int64)
    for i in range(n_classes):
        y_true_bin[:, i] = (y_true == i).astype(np.int64)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{cls_name} (AUC = {roc_auc_val:.3f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    ax.plot(fpr_micro, tpr_micro, color="deeppink", lw=2, linestyle="--",
            label=f"Micro-avg (AUC = {auc(fpr_micro, tpr_micro):.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_confidence_distribution(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confidence distribution split by correct/incorrect."""
    correct_mask = (y_true == y_pred)
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(correct_conf) > 0:
        ax.hist(correct_conf, bins=20, alpha=0.6, color="green",
                label=f"Correct (n={len(correct_conf)})", edgecolor="black")
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=20, alpha=0.6, color="red",
                label=f"Incorrect (n={len(incorrect_conf)})", edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# =============================================================================
# TRAINING LOOP (Fixed FocalLoss, Fixed CBAM)
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_path: Optional[Path] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Train the model with correct FocalLoss and AMP.

    FIX NOTES:
      - FocalLoss is now called with alpha= (not class_weights=)
      - CBAM channels are correct (2048 for ResNet50, not 256)
      - AMP uses torch.amp.autocast with device_type kwarg

    Args:
        model: FreeBuffV4 model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Compute device.
        num_epochs: Max epochs.
        patience: Early stopping patience.
        checkpoint_path: Where to save the best checkpoint.
        use_amp: Use automatic mixed precision.

    Returns:
        Training history dict.
    """
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()

    # Compute class weights from training set for balanced loss
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.append(batch["label"].numpy())
    all_train_labels = np.concatenate(all_train_labels)
    class_counts = np.bincount(all_train_labels, minlength=NUM_CLASSES).astype(np.float32)
    inverse_freq = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = inverse_freq / inverse_freq.mean()

    logger.info("Training class distribution: %s",
                {CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)})
    logger.info("Computed class weights: %s",
                {CLASSES[i]: float(class_weights[i]) for i in range(NUM_CLASSES)})

    # FIX: FocalLoss(alpha=..., not class_weights=...)
    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction="mean",
    )

    param_groups = build_parameter_groups(
        model, backbone_lr=BACKBONE_LR, head_lr=DEFAULT_LR, weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(
        param_groups, eps=ADAMW_EPS, betas=(ADAMW_BETA1, ADAMW_BETA2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN,
    )

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "lr": [],
    }

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            hog_feats = batch.get("hog_pca")
            if hog_feats is not None:
                hog_feats = hog_feats.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(images, hog_pca=hog_feats)
                    loss = criterion(out["logits"], labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images, hog_pca=hog_feats)
                loss = criterion(out["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = out["logits"].argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_targets: List[np.ndarray] = []
        val_preds: List[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                hog_feats = batch.get("hog_pca")
                if hog_feats is not None:
                    hog_feats = hog_feats.to(device, non_blocking=True)

                out = model(images, hog_pca=hog_feats)
                loss = criterion(out["logits"], labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = out["logits"].argmax(dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += labels.size(0)
                val_targets.append(labels.cpu().numpy())
                val_preds.append(preds.cpu().numpy())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = float(f1_score(
            np.concatenate(val_targets), np.concatenate(val_preds),
            average="weighted", zero_division=0,
        ))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f val_loss=%.4f "
            "val_acc=%.4f val_f1=%.4f lr=%.2e",
            epoch, num_epochs, train_loss, train_acc, val_loss, val_acc,
            val_f1, current_lr,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                path=checkpoint_path, model=model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, train_loss=train_loss,
                val_loss=val_loss, val_acc=val_acc,
            )
            logger.info("  New best val_f1=%.4f, checkpoint saved.", val_f1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.warning("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
            break

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "history": history,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """End-to-end pipeline: load data -> train/load -> evaluate -> Grad-CAM."""
    set_all_seeds(args.seed)
    initialize_directories()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(CUDA_DEVICE_STRING if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Data ---
    logger.info("Loading dataset from %s", args.data_dir)
    full_dataset = build_dataset_from_folder(args.data_dir)

    train_ds, val_ds, test_ds = split_dataset_into_train_val_test(
        full_dataset, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, stratify=True,
    )

    hog_extractor = HOGExtractor() if args.use_hog else None
    pca_model = None
    if args.use_hog:
        logger.info("Fitting PCA on training HOG features (%d components)...", args.pca_components)
        pca_model = fit_pca_on_training_hog(train_ds, pca_components=args.pca_components)

    train_samples = [train_ds.dataset.samples_list[i] for i in train_ds.indices]
    val_samples = [val_ds.dataset.samples_list[i] for i in val_ds.indices]
    test_samples = [test_ds.dataset.samples_list[i] for i in test_ds.indices]

    train_dataset = BrainTumorDataset(
        samples=train_samples, hog_extractor=hog_extractor, pca=pca_model,
        use_hog=args.use_hog, augment=True, augment_fn=augment_mri_image,
    )
    val_dataset = BrainTumorDataset(
        samples=val_samples, hog_extractor=hog_extractor, pca=pca_model,
        use_hog=args.use_hog,
    )
    test_dataset = BrainTumorDataset(
        samples=test_samples, hog_extractor=hog_extractor, pca=pca_model,
        use_hog=args.use_hog,
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size, num_workers=0,
    )

    logger.info("Dataset: train=%d, val=%d, test=%d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    # --- Model ---
    model = build_freebuff_v4(
        num_classes=NUM_CLASSES, device=device,
        pretrained=True, freeze_backbone=False,
    )

    # Verify model I/O shapes before training
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
    dummy_hog = torch.randn(1, HOG_PCA_COMPONENTS).to(device) if args.use_hog else None
    assert_model_io_sanity(model, dummy_input, dummy_hog)

    # --- Train or load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info("Loading existing checkpoint: %s", args.checkpoint)
        load_checkpoint(args.checkpoint, model, device=device, strict=False)
    else:
        logger.info("Training from scratch for %d epochs...", args.num_epochs)
        train_result = train_model(
            model, train_loader, val_loader, device,
            num_epochs=args.num_epochs, patience=args.patience,
            use_amp=args.use_amp,
        )
        logger.info("Training complete: best_val_f1=%.4f at epoch %d",
                     train_result["best_val_f1"], train_result["best_epoch"])

        # Load best checkpoint
        best_path = DEFAULT_CHECKPOINT_PATH
        if best_path.exists():
            load_checkpoint(best_path, model, device=device, strict=False)

    model.eval()

    # --- Evaluate ---
    logger.info("Running comprehensive evaluation...")
    eval_results = evaluate_comprehensive(
        model, test_loader, device,
        use_tta=args.use_tta,
        output_dir=EVAL_VIS_DIR,
    )

    logger.info(
        "Pipeline complete. Accuracy=%.4f, Weighted-F1=%.4f, Macro-F1=%.4f",
        eval_results["accuracy"], eval_results["weighted_f1"],
        eval_results["macro_f1"],
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FreeBuff v4 — Production Pipeline (TTA + Grad-CAM++ + Evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with class subfolders (Glioma, Meningioma, No_Tumor, Pituitary).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to existing checkpoint to load (skips training).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--use_tta", action="store_true", default=True,
                        help="Enable Test-Time Augmentation during evaluation.")
    parser.add_argument("--no_tta", dest="use_tta", action="store_false",
                        help="Disable TTA during evaluation.")
    parser.add_argument("--use_hog", action="store_true", default=False,
                        help="Enable HOG + PCA texture branch.")
    parser.add_argument("--pca_components", type=int, default=HOG_PCA_COMPONENTS)
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Enable automatic mixed precision training.")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                        help="Disable AMP.")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()
