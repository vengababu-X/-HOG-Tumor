#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Comprehensive Evaluation & Clinical Analytics
# ==============================================================================
# Compiles test-split performance figures: confusion matrices (counts and
# normalized proportions), one-vs-rest ROC curves with individual and
# micro-averaged AUC, per-class F1-score bar charts with weighted average
# thresholds, prediction confidence distribution histograms (correct vs.
# incorrect splits), and mean confidence per class bar charts.
# ==============================================================================
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode

from config import (
    CLASSES,
    NUM_CLASSES,
    EVAL_VIS_DIR,
    GRADCAM_EXPORT_DIR,
    PDF_REPORT_DIR,
    TTA_FULL,
    TTA_NUM_SAMPLES,
    TTA_ROTATIONS_DEG,
    TTA_FLIPS,
    TTA_SCALE_FACTORS,
    TTA_SOFTMAX_TEMP,
    GRADCAM_TARGET_LAYER_NAME,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
)

from dataset import (
    BrainTumorDataset,
    HOGExtractor,
    build_dataset_from_folder,
    split_dataset_into_train_val_test,
    make_dataloaders,
    fit_pca_on_training_hog,
)

from model import (
    FreeBuffV4,
    build_freebuff_v4,
    GradCAMPlusPlus,
    load_checkpoint,
)

from inference import (
    TTAPredictor,
    build_tta_transforms,
    preprocess_image_for_inference,
    generate_diagnostic_summary,
    lookup_who_stage,
    is_low_confidence,
)

logger = logging.getLogger("freebuff_v4.evaluate")


# ==============================================================================
# FIGURE SAVING UTILITY
# ==============================================================================
def save_figure(fig: plt.Figure, name: str, directory: Optional[Path] = None) -> Path:
    """
    Save a matplotlib figure to the evaluation visualizations directory.

    Args:
        fig: Matplotlib figure.
        name: Filename stem (without extension).
        directory: Target directory; defaults to EVAL_VIS_DIR.

    Returns:
        Path to the saved figure.
    """
    if directory is None:
        directory = EVAL_VIS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved evaluation figure: %s", out_path)
    return out_path


# ==============================================================================
# CONFUSION MATRIX PLOTTING
# ==============================================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: Optional[Any] = None,
    figsize: Tuple[int, int] = (10, 8),
    output_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and plot a confusion matrix (counts or normalized).

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        class_names: List of class name strings for axis tick labels.
        normalize: If True, normalize each row to sum to 1 (recall-oriented).
        title: Plot title.
        cmap: Matplotlib colormap; defaults to 'Blues'.
        figsize: Figure size.
        output_name: Stem for saved figure; if None, no file is saved.

    Returns:
        Tuple of (cm_array, cm_normalized_or_counts_array).
    """
    cmatrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        row_sums = cmatrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cmatrix_norm = cmatrix.astype(np.float64) / row_sums
        disp = cmatrix_norm
        label_suffix = " (normalized)"
    else:
        disp = cmatrix
        label_suffix = ""

    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = cmap if cmap is not None else plt.cm.Blues
    im = ax.imshow(disp, interpolation="nearest", cmap=cmap_obj)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(f"{title}{label_suffix}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fmt = ".2f" if normalize else "d"
    thresh = disp.max() / 2.0
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            ax.text(j, i, format(disp[i, j], fmt),
                    ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black")

    fig.tight_layout()
    if output_name:
        save_figure(fig, output_name)
    return cmatrix, disp


# ==============================================================================
# ONE-VS-REST ROC CURVES AND AUC
# ==============================================================================
def plot_one_vs_rest_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    title: str = "One-vs-Rest ROC Curves",
    micro_avg: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    output_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Plot one-vs-rest ROC curves for each class and macro/micro averages.

    Args:
        y_true: Ground truth labels (N,).
        y_score: Predicted probability matrix (N, C).
        class_names: List of class names.
        title: Plot title.
        micro_avg: Whether to plot micro-averaged ROC.
        figsize: Figure size.
        output_name: Stem for saved figure.

    Returns:
        Dict mapping class name -> AUC, plus 'micro_avg' if enabled.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_classes = len(class_names)
    if y_score.ndim != 2 or y_score.shape[1] != n_classes:
        raise ValueError(f"y_score must have shape (N, {n_classes}), got {y_score.shape}")

    # Binarize ground truth for one-vs-rest
    y_bin = np.zeros((y_true.size, n_classes), dtype=np.int64)
    for i, cls in enumerate(range(n_classes)):
        y_bin[:, i] = (y_true == cls).astype(np.int64)

    fpr: Dict[str, np.ndarray] = {}
    tpr: Dict[str, np.ndarray] = {}
    roc_auc: Dict[str, float] = {}

    for i, cls_name in enumerate(class_names):
        fpr[cls_name], tpr[cls_name], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[cls_name] = auc(fpr[cls_name], tpr[cls_name])

    if micro_avg:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc["micro_avg"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average: aggregate all TPRs across thresholds
    if n_classes > 1:
        all_tpr = []
        mean_fpr = np.linspace(0, 1, 1000)
        for i in range(n_classes):
            interp_tpr = np.interp(mean_fpr, fpr[class_names[i]], tpr[class_names[i]])
            interp_tpr[0] = 0.0
            all_tpr.append(interp_tpr)
        mean_tpr = np.mean(all_tpr, axis=0)
        mean_tpr[-1] = 1.0
        roc_auc["macro_avg"] = auc(mean_fpr, mean_tpr)
        fpr["macro"] = mean_fpr
        tpr["macro"] = mean_tpr

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes + 2))
    for i, cls_name in enumerate(class_names):
        ax.plot(fpr[cls_name], tpr[cls_name], color=colors[i],
                lw=2, label=f"{cls_name} (AUC = {roc_auc[cls_name]:0.3f})")
    if micro_avg and "micro" in fpr:
        ax.plot(fpr["micro"], tpr["micro"], color="deeppink", lw=2,
                label=f"Micro-average (AUC = {roc_auc['micro_avg']:0.3f})", linestyle="--")
    if "macro" in fpr:
        ax.plot(fpr["macro"], tpr["macro"], color="navy", lw=2,
                label=f"Macro-average (AUC = {roc_auc['macro_avg']:0.3f})", linestyle=":")

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance (AUC=0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    if output_name:
        save_figure(fig, output_name)
    return roc_auc


# ==============================================================================
# PER-CLASS F1-SCORE BAR CHART
# ==============================================================================
def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    average_types: Optional[List[str]] = None,
    title: str = "Per-Class F1-Scores",
    figsize: Tuple[int, int] = (10, 6),
    output_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute and plot per-class F1 scores plus weighted average.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Class name list.
        average_types: Which averaging strategies to also display as horizontal
            reference lines (e.g., ['micro', 'macro', 'weighted']).
        title: Plot title.
        figsize: Figure size.
        output_name: Stem for saved figure.

    Returns:
        Dict with per-class F1 and selected averages.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    results: Dict[str, float] = {}
    for i, cls in enumerate(class_names):
        results[f"{cls}_F1"] = float(per_class[i])

    if average_types:
        for avg in average_types:
            val = float(f1_score(y_true, y_pred, average=avg, labels=range(len(class_names))))
            results[f"{avg}_avg_F1"] = val

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(class_names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(x, per_class, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)

    # Annotate bars with value
    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    if average_types:
        y_max = max(1.0, per_class.max() + 0.05)
        for avg_name in average_types:
            val = results[f"{avg_name}_avg_F1"]
            label = f"{avg_name.capitalize()} average: {val:.3f}"
            ax.axhline(val, color="red", linestyle="--", linewidth=1.5, label=label)
        ax.legend(fontsize="small")

    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    if output_name:
        save_figure(fig, output_name)
    return results


# ==============================================================================
# PREDICTION CONFIDENCE DISTRIBUTION HISTOGRAM
# ==============================================================================
def plot_confidence_distribution(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Prediction Confidence Distribution",
    correct_label: str = "Correct",
    incorrect_label: str = "Incorrect",
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot a histogram of prediction confidences split by correctness.

    Args:
        confidences: Confidence values (N,) in [0, 1].
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Class names (for overall report, not per-class here).
        title: Plot title.
        correct_label: Legend label for correct predictions.
        incorrect_label: Legend label for incorrect predictions.
        bins: Number of histogram bins.
        figsize: Figure size.
        output_name: Stem for saved figure.

    Returns:
        Dict with summary statistics.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    correct_mask = (y_true == y_pred)
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    summary = {
        "n_total": int(confidences.size),
        "n_correct": int(correct_mask.sum()),
        "n_incorrect": int((~correct_mask).sum()),
        "accuracy": float(correct_mask.mean()),
        "mean_confidence_correct": float(correct_conf.mean()) if correct_conf.size else 0.0,
        "mean_confidence_incorrect": float(incorrect_conf.mean()) if incorrect_conf.size else 0.0,
        "median_confidence_correct": float(np.median(correct_conf)) if correct_conf.size else 0.0,
        "median_confidence_incorrect": float(np.median(incorrect_conf)) if incorrect_conf.size else 0.0,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(correct_conf, bins=bins, alpha=0.6, color="green", label=correct_label, edgecolor="black")
    ax1.hist(incorrect_conf, bins=bins, alpha=0.6, color="red", label=incorrect_label, edgecolor="black")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title}\nCorrect: {summary['n_correct']}, Incorrect: {summary['n_incorrect']}")
    ax1.axvline(summary["mean_confidence_correct"], color="green", linestyle="--", lw=2,
                label=f"Mean correct ({summary['mean_confidence_correct']:.2f})")
    ax1.axvline(summary["mean_confidence_incorrect"], color="red", linestyle="--", lw=2,
                label=f"Mean incorrect ({summary['mean_confidence_incorrect']:.2f})")
    ax1.legend(fontsize="small")

    # Cumulative distribution also shown in second panel for clarity
    ax2.hist(correct_conf, bins=bins, cumulative=True, density=True, alpha=0.5,
             color="green", label=correct_label, edgecolor="black")
    ax2.hist(incorrect_conf, bins=bins, cumulative=True, density=True, alpha=0.5,
             color="red", label=incorrect_label, edgecolor="black")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Confidence Distribution")
    ax2.legend(fontsize="small")

    fig.tight_layout()
    if output_name:
        save_figure(fig, output_name)
    return summary


# ==============================================================================
# MEAN CONFIDENCE PER CLASS BAR CHART
# ==============================================================================
def plot_mean_confidence_per_class(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Mean Confidence per Class",
    figsize: Tuple[int, int] = (10, 6),
    output_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Plot mean prediction confidence for each class (over all samples where
    that class was the true label, including both correct and incorrect
    predictions for that true class).

    Args:
        confidences: Confidence values (N,).
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Class names.
        title: Plot title.
        figsize: Figure size.
        output_name: Stem for saved figure.

    Returns:
        Dict class_name -> mean_confidence, plus per-class accuracy.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    results: Dict[str, float] = {}
    class_ids = np.arange(len(class_names))
    mean_confs = []
    accuracies = []

    for cid in class_ids:
        mask = (y_true == cid)
        if mask.sum() == 0:
            mean_confs.append(0.0)
            accuracies.append(0.0)
            results[class_names[cid]] = 0.0
            continue
        conf_cls = confidences[mask]
        correct_cls = (y_pred[mask] == cid)
        mean_confs.append(float(conf_cls.mean()))
        accuracies.append(float(correct_cls.mean()))
        results[class_names[cid]] = float(conf_cls.mean())

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(class_names))
    width = 0.35
    ax.bar(x - width/2, mean_confs, width, label="Mean Confidence", color="steelblue", edgecolor="black")
    ax.bar(x + width/2, accuracies, width, label="Accuracy", color="seagreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.set_ylim(0.0, 1.05)

    # Annotate bars
    for i, (mc, ac) in enumerate(zip(mean_confs, accuracies)):
        ax.text(i - width/2, mc + 0.01, f"{mc:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, ac + 0.01, f"{ac:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    if output_name:
        save_figure(fig, output_name)
    return results


# ==============================================================================
# COMPREHENSIVE EVALUATION RUNNER
# ==============================================================================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    use_tta: bool = True,
    tta_transforms: Optional[List[Callable[[Tensor], Tensor]]] = None,
    hog_pca_for_test: Optional[Any] = None,
    use_hog: bool = False,
    output_dir: Optional[Path] = None,
    save_gradcam_for_misclassifications: bool = True,
    max_gradcam_saves: int = 10,
) -> Dict[str, Any]:
    """
    Run a full evaluation pass on the test loader and generate all standard
    visual analytics.

    Args:
        model: FreeBuffV4 model in eval mode.
        test_loader: DataLoader over the test split.
        device: Torch device.
        class_names: Class names list; defaults to config.CLASSES.
        use_tta: Whether to use TTA for test-set evaluation.
        tta_transforms: Optional custom TTA transform list.
        hog_pca_for_test: Optional PCA model used to project HOG features for
            the test set if use_hog is True.
        use_hog: Whether the model expects HOG inputs.
        output_dir: Directory for saving figures; defaults to EVAL_VIS_DIR.
        save_gradcam_for_misclassifications: Whether to save Grad-CAM overlays
            for a subset of misclassified test samples.
        max_gradcam_saves: Maximum number of Grad-CAM overlays to save.

    Returns:
        Dict with metrics, arrays, and paths to saved figures.
    """
    class_names = class_names or CLASSES
    output_dir = output_dir or EVAL_VIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    criterion = None  # Not needed for evaluation-only pass

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_confidences: List[np.ndarray] = []
    all_paths: List[str] = []
    misclassified_count = 0

    # TTA setup
    if use_tta:
        transforms = tta_transforms or build_tta_transforms()
        predictor = TTAPredictor(model, device, transforms=transforms)
    else:
        predictor = None

    gradcam_engine = None
    if save_gradcam_for_misclassifications:
        try:
            gradcam_engine = GradCAMPlusPlus(model, target_layer_name=GRADCAM_TARGET_LAYER_NAME, device=device)
        except Exception as e:
            logger.warning("Grad-CAM++ engine initialization failed: %s; skipping Grad-CAM exports.", e)
            gradcam_engine = None

    # Collect prediction batches (inside no_grad for efficiency)
    misclassified_samples: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, dtype=next(model.parameters()).dtype)
            labels = batch["label"].cpu().numpy()
            paths = batch.get("path", [f"sample_{i}" for i in range(len(labels))])

            hog_feats = None
            if use_hog and batch.get("hog_pca") is not None:
                hog_feats = batch["hog_pca"].to(device, dtype=torch.float32)

            if use_tta and predictor is not None:
                pred = predictor.predict_batch(images, hog_pcas=hog_feats)
                probs = pred["avg_probs"].cpu().numpy()
                preds = pred["pred_classes"].cpu().numpy()
                confs = pred["confidences"].cpu().numpy()
            else:
                images_b = images
                out = model(images_b, hog_pca=hog_feats)
                logits = out["logits"]
                probs = torch.softmax(logits / TTA_SOFTMAX_TEMP, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                confs = probs.max(axis=1)

            all_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
            all_confidences.append(confs)
            all_paths.extend(paths)

            # Track misclassifications for optional Grad-CAM.
            # FIX: Store the misclassified images for later Grad-CAM computation
            # OUTSIDE torch.no_grad(), because the new GradCAMPlusPlus uses
            # torch.autograd.grad() which requires gradient tracking.
            if save_gradcam_for_misclassifications:
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        misclassified_count += 1
                        if misclassified_count <= max_gradcam_saves:
                            misclassified_samples.append({
                                "image": images[i].clone().cpu(),
                                "true_cls": int(labels[i]),
                                "pred_cls": int(preds[i]),
                                "path": str(paths[i]),
                            })

    # FIX: Run Grad-CAM on misclassified samples OUTSIDE torch.no_grad().
    # The new GradCAMPlusPlus.generate_heatmap() uses torch.autograd.grad()
    # which requires the autograd engine to be active.
    if save_gradcam_for_misclassifications and gradcam_engine is not None:
        for sample_info in misclassified_samples:
            try:
                img_for_cam = sample_info["image"].to(device)
                true_cls = sample_info["true_cls"]
                pred_cls = sample_info["pred_cls"]
                img_path = sample_info["path"]

                # Ensure image requires grad for Grad-CAM
                img_for_cam = img_for_cam.detach().requires_grad_(True)

                hm, _ = gradcam_engine.generate_heatmap(
                    img_for_cam.unsqueeze(0), class_idx=pred_cls,
                )
                overlay = GradCAMPlusPlus.overlay_heatmap_on_image(
                    img_for_cam, hm, alpha=GRADCAM_ALPHA,
                    colormap=GRADCAM_COLORMAP,
                )
                overlay_np = overlay.cpu().numpy().astype(np.float32)
                overlay_np = (np.clip(overlay_np, 0, 1) * 255).astype(np.uint8)
                overlay_np = overlay_np.transpose(1, 2, 0)
                from PIL import Image as PILImage
                overlay_img = PILImage.fromarray(overlay_np, mode="RGB")
                save_path = GRADCAM_EXPORT_DIR / f"misclass_{true_cls}_true_{class_names[true_cls]}_pred_{class_names[pred_cls]}_{os.path.basename(img_path)}"
                overlay_img.save(str(save_path))
                logger.info("Saved Grad-CAM overlay for misclassified sample: %s", save_path)
            except Exception as e:
                logger.debug("Failed to save Grad-CAM overlay: %s", e)
        gradcam_engine.remove_hooks()

    # Concatenate
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    y_confs = np.concatenate(all_confidences)

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=range(len(class_names))))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=range(len(class_names))))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", labels=range(len(class_names))))

    # ROC AUCs
    try:
        roc_aucs = plot_one_vs_rest_roc(
            y_true, y_probs, class_names,
            title="One-vs-Rest ROC Curves — Test Split",
            micro_avg=True,
            output_name="roc_one_vs_rest",
        )
    except Exception as e:
        logger.warning("ROC computation failed: %s; returning empty dict.", e)
        roc_aucs = {}

    # Confusion matrices
    cm_counts, cm_norm = plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=False,
        title="Confusion Matrix — Counts",
        output_name="confusion_matrix_counts",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=True,
        title="Confusion Matrix — Normalized",
        output_name="confusion_matrix_normalized",
    )

    # Per-class F1
    f1_results = plot_per_class_f1(
        y_true, y_pred, class_names,
        average_types=["micro", "macro", "weighted"],
        title="Per-Class F1-Scores (Test Split)",
        output_name="per_class_f1",
    )

    # Confidence distribution
    conf_summary = plot_confidence_distribution(
        y_confs, y_true, y_pred, class_names,
        title="Prediction Confidence Distribution (Test Split)",
        output_name="confidence_distribution",
    )

    # Mean confidence per class
    mean_conf_per_class = plot_mean_confidence_per_class(
        y_confs, y_true, y_pred, class_names,
        title="Mean Confidence per True Class (Test Split)",
        output_name="mean_confidence_per_class",
    )

    # FIX: Generate a full classification report (precision, recall, f1 per class)
    # This is essential for measuring true performance rather than blind accuracy,
    # especially with class imbalance.
    clf_report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
    )
    clf_report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
        output_dict=True,
    )
    logger.info("Classification Report:\n%s", clf_report_str)

    results = {
        "accuracy": accuracy,
        "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "roc_aucs": roc_aucs,
        "classification_report_str": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix_counts": cm_counts,
        "confusion_matrix_normalized": cm_norm,
        "confidence_summary": conf_summary,
        "mean_confidence_per_class": mean_conf_per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "y_confidences": y_confs,
        "n_test_samples": int(y_true.size),
        "n_misclassified": int((y_true != y_pred).sum()),
        "output_dir": output_dir,
        "gradcam_saved_count": misclassified_count if gradcam_engine else 0,
    }
    return results


# ==============================================================================
# EVALUATION + VISUALIZATION COORDINATOR (WITH MODEL LOADING)
# ==============================================================================
def run_evaluation(
    checkpoint_path: Union[str, Path],
    test_data_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    use_tta: bool = True,
    use_hog: bool = False,
    pca_path: Optional[Union[str, Path]] = None,
    device_str: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end evaluation entry point: loads a checkpoint, builds the test set
    (if a data directory is given), runs evaluation, and generates all figures.

    Args:
        checkpoint_path: Path to trained model checkpoint (.pt).
        test_data_dir: Optional directory containing test-class subfolders.
        batch_size: Evaluation batch size.
        use_tta: Enable TTA during evaluation.
        use_hog: Whether to feed HOG features into the model.
        pca_path: Path to fitted PCA pickle (if use_hog).
        device_str: Override torch device string.
        output_dir: Directory for output figures.

    Returns:
        Evaluation results dict.
    """
    from sklearn.decomposition import PCA as SklearnPCA
    import pickle

    if device_str:
        requested = torch.device(device_str)
        device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = CLASSES

    logger.info("Loading checkpoint from %s", checkpoint_path)
    # Load model first
    model = build_freebuff_v4(num_classes=NUM_CLASSES, device=device, pretrained=False)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device, strict=True)

    # Optionally load PCA model for HOG
    pca_model = None
    if use_hog and pca_path is not None:
        pca_path = Path(pca_path)
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_model = pickle.load(f)
            logger.info("Loaded PCA model from %s", pca_path)
        else:
            logger.warning("PCA path not found: %s; continuing without HOG features.", pca_path)

    # Test set construction
    if test_data_dir is not None:
        test_data_dir = Path(test_data_dir)
        full_dataset = build_dataset_from_folder(test_data_dir)
        # Build dataset with HOG pca
        hog_extractor = HOGExtractor() if use_hog else None
        test_dataset = BrainTumorDataset(
            samples=full_dataset.samples_list,
            hog_extractor=hog_extractor,
            pca=pca_model,
            use_hog=use_hog,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: {
                "image": torch.stack([x["image"] for x in b]),
                "label": torch.stack([x["label"] for x in b]),
                "hog_pca": torch.stack([x["hog_pca"] for x in b]) if all(x["hog_pca"] is not None for x in b) else None,
                "path": [x["path"] for x in b],
                "metadata": [x["metadata"] for x in b],
            },
        )
        logger.info("Built test loader with %d samples.", len(test_dataset))
    else:
        raise ValueError("test_data_dir must be provided for evaluation.")

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        use_tta=use_tta,
        output_dir=output_dir,
        save_gradcam_for_misclassifications=True,
        max_gradcam_saves=10,
        hog_pca_for_test=pca_model,
        use_hog=use_hog,
    )

    logger.info(
        "Evaluation complete: accuracy=%.4f, macro_f1=%.4f, weighted_f1=%.4f, "
        "n_misclassified=%d / %d.",
        results["accuracy"], results["macro_f1"], results["weighted_f1"],
        results["n_misclassified"], results["n_test_samples"],
    )
    return results


# ==============================================================================
# BULK RESULT PRINTOUT
# ==============================================================================
def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Pretty-print key evaluation metrics to the console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:            {results['accuracy']:.4f}")
    print(f"Macro F1:            {results['macro_f1']:.4f}")
    print(f"Weighted F1:         {results['weighted_f1']:.4f}")
    print(f"Micro F1:            {results['micro_f1']:.4f}")
    print(f"Test samples:        {results['n_test_samples']}")
    print(f"Misclassified:       {results['n_misclassified']}")
    print(f"Mean confidence:     {results['confidence_summary']['mean_confidence_correct']:.3f} (correct), "
          f"{results['confidence_summary']['mean_confidence_incorrect']:.3f} (incorrect)")
    # FIX: Print the full classification report (precision, recall, f1, support)
    # This is essential for measuring true performance rather than blind accuracy.
    if 'classification_report_str' in results:
        print("\nClassification Report:")
        print(results['classification_report_str'])
    print("ROC AUCs:")
    for k, v in results['roc_aucs'].items():
        print(f"  {k}: {v:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in results['per_class_f1'].items():
        print(f"  {cls}: {f1:.4f}")
    print("\nConfusion Matrix (counts):")
    print(results['confusion_matrix_counts'].astype(int))
    print("\nConfusion Matrix (normalized):")
    print(np.array2string(results['confusion_matrix_normalized'], precision=3, suppress_small=True))
    print("=" * 60 + "\n")


# ==============================================================================
# MODULE ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="FreeBuff v4 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory with test class subfolders.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation during evaluation.")
    parser.add_argument("--use_hog", action="store_true", help="Enable HOG branch during evaluation (requires PCA).")
    parser.add_argument("--pca_path", type=str, default=None, help="Path to fitted PCA pickle.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device string.")

    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        test_data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
        use_hog=args.use_hog,
        pca_path=args.pca_path,
        device_str=args.device,
        output_dir=EVAL_VIS_DIR,
    )

    print_evaluation_summary(results)
