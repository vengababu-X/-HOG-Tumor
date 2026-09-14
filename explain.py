"""
explain.py -- FreeBuff v4 Explainability CLI (Grad-CAM + Overlay)
==================================================================

Standalone script that runs a FreeBuffalo inference and produces a
publication-ready visual explanation for the evaluation panel.

What it does
------------
1. Loads the ``FreeBuffInference`` engine from ``saved_models/``.
2. Preprocesses the provided T1c / T2 / FLAIR MRI slices (DICOM-aware,
   144x144x3, [0,1] float32 -- exactly the same contract as the CLI and
   API).
3. Runs a prediction through the full hybrid pipeline (deep extractor ->
   HOG/PCA -> scaler -> SVM / super-learner) to get the predicted class
   and confidence.
4. Computes per-modality Grad-CAM heatmaps by hooking the last convolutional
   layer of each modality's EfficientNet+CBAM branch (``{m}_cbam_top``) and
   backpropagating the predicted class logit.
5. Blends each heatmap onto the corresponding MRI slice with a Jet colormap
   and an alpha blend.
6. Draws a bounding box around the region of maximum activation (the region
   the model is "looking at").
7. Saves a 3-panel figure (``prediction_explanation.png``) -- one panel per
   modality -- plus a machine-readable JSON sidecar with the prediction and
   heatmap metadata.

This is intentionally standalone: it does not depend on Streamlit, the API
server, or the evaluation harness.  It imports ``FreeBuffInference`` and
uses the explainability methods that already live in ``inference.py`` --
``gradcam_maps``, ``explain_maps``, and ``overlay`` -- so there is a single
source of truth for the explanation algorithm.

The script already exists as a capability inside ``inference.py`` (the
``FreeBuffInference.gradcam_maps``, ``FreeBuffInference.explain_maps`` and
``FreeBuffInference.overlay`` methods).  This script is the dedicated
panel-facing CLI that consumes that capability and writes the figure to disk.

Usage
-----
    python explain.py path/to/t1c.png path/to/t2.png path/to/flair.png
    python explain.py path/to/t1c.png                  # single-modality (broadcast)
    python explain.py --output my_explain.png t1c.png t2.png flair.png
    python explain.py --mode grad_cam t1c.png         # force Grad-CAM
    python explain.py --mode score_cam t1c.png        # Score-CAM++
    python explain.py --mode attention_rollout t1c.png # transformer attention

Requirements
------------
    numpy, opencv-python, matplotlib, tensorflow, pillow
    (everything else is already in the project deps)

Author: FreeBuff capstone team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("explain")

# ---------------------------------------------------------------------------
# Constants -- kept in sync with config.py / inference.py
# ---------------------------------------------------------------------------

MODALITIES = ("t1ce", "t2", "flair")
MODALITY_LABELS = {
    "t1ce": "T1c (contrast-enhanced)",
    "t2": "T2-weighted",
    "flair": "FLAIR",
}
IMG_SIZE = 144


# ---------------------------------------------------------------------------
# Dataclasses for the JSON sidecar
# ---------------------------------------------------------------------------


@dataclass
class ExplainabilityResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: Any
    modalities_uploaded: List[str]
    heatmaps: Dict[str, Dict[str, Any]]
    bounding_boxes: Dict[str, Optional[List[int]]]
    fig_path: str


# ---------------------------------------------------------------------------
# Grad-CAM + overlay pipeline
# ---------------------------------------------------------------------------


def load_engine(models_dir: str) -> Any:
    """Load the FreeBuff inference engine (same as app_master / app.py)."""
    from inference import FreeBuffInference

    engine = FreeBuffInference.from_dir(models_dir)
    logger.info(
        "Engine loaded — classes=%s  version=v%s",
        engine.class_names,
        engine.manifest.get("artifact_version", "?"),
    )
    return engine


def preprocess_files(paths: Dict[str, Optional[str]]) -> Dict[str, Optional[np.ndarray]]:
    """Preprocess each provided MRI file through the engine's own contract."""
    from inference import FreeBuffInference

    uploads: Dict[str, Optional[np.ndarray]] = {m: None for m in MODALITIES}
    for modality, path in paths.items():
        if path is None or not os.path.isfile(path):
            continue
        try:
            uploads[modality] = FreeBuffInference.preprocess_path(path)
            logger.info("Preprocessed %s (%s) -> %s", modality, os.path.basename(path), uploads[modality].shape)
        except Exception as exc:
            logger.error("Failed to preprocess %s (%s): %s", modality, path, exc)
            uploads[modality] = None
    return uploads


def run_prediction(
    engine: Any, uploads: Dict[str, Optional[np.ndarray]], use_tta: bool = True
) -> Tuple[str, float, Dict[str, float], Any]:
    """Run the full inference pipeline and return (class, conf, probs, version)."""
    result = engine.predict_uploads(uploads, use_tta=use_tta)
    return (
        result["predicted_class"],
        result["confidence"],
        result["probabilities"],
        result.get("model_version"),
    )


def compute_heatmaps(
    engine: Any,
    uploads: Dict[str, Optional[np.ndarray]],
    class_idx: Optional[int] = None,
    mode: str = "auto",
) -> Dict[str, np.ndarray]:
    """Compute per-modality explanation heatmaps.

    ``mode``: "auto" (attention_rollout -> score_cam -> grad_cam, first
    success), "grad_cam", "score_cam", or "attention_rollout".
    """
    if mode == "auto":
        maps = engine.explain_maps(uploads, class_idx=class_idx, mode="auto")
    elif mode == "grad_cam":
        maps = engine.gradcam_maps(uploads, class_idx=class_idx)
    elif mode == "score_cam":
        maps = engine.score_cam_maps(uploads, class_idx=class_idx)
    elif mode == "attention_rollout":
        maps = engine.attention_rollout_maps(uploads, class_idx=class_idx)
    else:
        raise ValueError(f"Unknown explainability mode: {mode!r}")
    return maps


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a [0,1] heatmap onto an [0,1] RGB image with a colormap.

    This mirrors ``FreeBuffInference.overlay`` but is exposed here as a
    standalone helper so the script does not require the class instance for
    pure visualisation tasks.
    """
    base = np.clip(image, 0.0, 1.0).astype(np.float32)
    heat_u8 = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cmap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colored
    return np.clip(blended, 0.0, 1.0)


def draw_bounding_box_around_roi(
    image: np.ndarray,
    heatmap: np.ndarray,
    bbox_threshold: float = 0.3,
    margin: int = 8,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle around the region of highest heatmap activation.

    The bounding box is computed from the contour of the thresholded heatmap,
    so it adapts to the shape of the activated region rather than assuming a
    fixed box.  If no region exceeds the threshold, the box is placed around
    the single hottest pixel (with a small margin).
    """
    vis = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = heatmap.shape[:2]

    mask = (heatmap >= bbox_threshold).astype(np.uint8) * 255
    if mask.max() == 0:
        # Fallback: box around the single hottest pixel.
        ys, xs = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x1 = max(0, xs - margin)
        y1 = max(0, ys - margin)
        x2 = min(w, xs + margin)
        y2 = min(h, ys + margin)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x1 = max(0, xs - margin)
        y1 = max(0, ys - margin)
        x2 = min(w, xs + margin)
        y2 = min(h, ys + margin)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def _heatmap_bounding_box(heatmap: np.ndarray, threshold: float = 0.25, margin: int = 8) -> Optional[List[int]]:
    """Compute a bounding box [x1, y1, x2, y2] around the thresholded heatmap.

    Returns None if the heatmap is empty or below threshold everywhere.
    """
    mask = (heatmap >= threshold).astype(np.uint8)
    if mask.max() == 0:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(heatmap.shape[1], int(xs.max()) + margin)
    y2 = min(heatmap.shape[0], int(ys.max()) + margin)
    return [x1, y1, x2, y2]


def build_figure(
    uploads: Dict[str, Optional[np.ndarray]],
    heatmaps: Dict[str, np.ndarray],
    pred_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    modality_labels: Dict[str, str],
    box_threshold: float = 0.25,
    overlay_alpha: float = 0.5,
) -> Tuple[plt.Figure, Dict[str, Optional[List[int]]]]:
    """Build the 3-panel figure: one per modality, with overlay + bbox."""
    n_mods = len([m for m in MODALITIES if uploads.get(m) is not None])
    if n_mods == 0:
        raise ValueError("No modalities available for figure.")

    fig, axes = plt.subplots(1, n_mods, figsize=(5.2 * n_mods, 5.2), dpi=150)
    if n_mods == 1:
        axes = [axes]
    fig.suptitle(
        f"FreeBuff v4 — Explainability  |  Prediction: {pred_class.upper()} "
        f"({confidence * 100:.1f}% confidence)",
        fontsize=13,
        fontweight="bold",
    )

    bounding_boxes: Dict[str, Optional[List[int]]] = {}
    for ax, modality in zip(axes, MODALITIES):
        image = uploads.get(modality)
        heatmap = heatmaps.get(modality)
        if image is None:
            ax.axis("off")
            ax.set_title(f"{modality_labels.get(modality, modality)} — not provided")
            continue

        # Base image
        base = np.clip(image, 0.0, 1.0)
        ax.imshow(base)
        ax.set_title(f"{modality_labels.get(modality, modality)}", fontsize=11)

        if heatmap is not None and heatmap.size > 0:
            overlay = overlay_heatmap(base, heatmap, alpha=overlay_alpha)
            ax.imshow(overlay)

            # Bounding box + label
            bbox = _heatmap_bounding_box(heatmap, threshold=box_threshold)
            bounding_boxes[modality] = bbox
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="white",
                    linewidth=2.5,
                    linestyle="--",
                )
                ax.add_patch(rect)
                hot_pixel_val = float(np.max(heatmap))
                ax.text(
                    x1 + 4,
                    y1 + 12,
                    f"ROI focus  max={hot_pixel_val:.2f}",
                    color="white",
                    fontsize=8,
                    backgroundcolor="black",
                    alpha=0.7,
                )

        ax.axis("off")

    # Add a small legend / caption strip at the bottom.
    prob_str = "  |  ".join(f"{k}: {v * 100:.1f}%" for k, v in probabilities.items())
    fig.text(
        0.5,
        0.02,
        f"Class probabilities: {prob_str}",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.12)
    return fig, bounding_boxes


def save_figure(fig: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved explanation figure -> %s", path)
    return path


def save_sidecar(result: ExplainabilityResult, sidecar_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(sidecar_path)) or ".", exist_ok=True)
    payload = asdict(result)
    # Make bounding boxes JSON-serialisable (they already are, but be explicit).
    payload["bounding_boxes"] = {
        k: (v if v is None else [int(x) for x in v]) for k, v in payload["bounding_boxes"].items()
    }
    with open(sidecar_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved JSON sidecar -> %s", sidecar_path)
    return sidecar_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="FreeBuff v4 explainability CLI — Grad-CAM + overlay for panel demo.",
        epilog=(
            "At least one MRI file is required. Missing modalities are "
            "automatically replicated from a present one (same behaviour as "
            "the inference CLI and API)."
        ),
    )
    ap.add_argument(
        "images",
        nargs="+",
        help="One or more MRI files. Use 't1ce:file t2:file flair:file' to tag, "
        "or bare paths (first path -> T1c, or broadcast if only one provided).",
    )
    ap.add_argument("--models_dir", default="saved_models")
    ap.add_argument("--output", default="prediction_explanation.png")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "grad_cam", "score_cam", "attention_rollout"],
        help="Explainability method (default: auto = attention_rollout -> score_cam -> grad_cam).",
    )
    ap.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation for the prediction.")
    ap.add_argument("--box_threshold", type=float, default=0.25, help="Heatmap value threshold for the ROI bounding box.")
    ap.add_argument("--overlay_alpha", type=float, default=0.5, help="Overlay blend alpha (0=mri only, 1=heatmap only).")
    ap.add_argument("--sidecar", default=None, help="JSON sidecar path (default: same name as --output with .json extension).")
    return ap


def parse_image_args(ap: argparse.ArgumentParser, raw: List[str]) -> Dict[str, Optional[str]]:
    """Parse CLI image args into {modality: path}."""
    uploads: Dict[str, Optional[str]] = {m: None for m in MODALITIES}
    for item in raw:
        if ":" in item:
            mod, path = item.split(":", 1)
            mod = mod.strip().lower()
            path = path.strip()
        else:
            mod, path = MODALITIES[0], item.strip()
        if mod not in MODALITIES:
            ap.error(f"Unknown modality '{mod}'. Use one of: {', '.join(MODALITIES)}")
        uploads[mod] = path if path else None
    return uploads


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    paths = parse_image_args(ap, args.images)

    provided = [m for m, p in paths.items() if p and os.path.isfile(p)]
    if not provided:
        ap.error("No valid MRI files provided. Supply at least one of: " + ", ".join(MODALITIES))

    logger.info("Loading FreeBuff v4 engine from %s ...", args.models_dir)
    try:
        engine = load_engine(args.models_dir)
    except Exception as exc:
        logger.exception("Failed to load engine")
        sys.exit(f"[error] Failed to load FreeBuff engine: {exc}")

    logger.info("Preprocessing uploads: %s", provided)
    uploads = preprocess_files(paths)
    if not any(uploads.values()):
        sys.exit("[error] No uploads could be preprocessed.")

    pred_class, confidence, probabilities, model_version = run_prediction(
        engine, uploads, use_tta=not args.no_tta
    )
    logger.info("Prediction: %s (%.1f%%)", pred_class, confidence * 100)

    # Determine class index for the heatmap, then compute heatmaps.
    class_idx = engine.class_names.index(pred_class) if pred_class in engine.class_names else None
    logger.info("Explainability mode: %s", args.mode)
    heatmaps = compute_heatmaps(engine, uploads, class_idx=class_idx, mode=args.mode)

    fig, bounding_boxes = build_figure(
        uploads,
        heatmaps,
        pred_class,
        confidence,
        probabilities,
        MODALITY_LABELS,
        box_threshold=args.box_threshold,
        overlay_alpha=args.overlay_alpha,
    )
    fig_path = save_figure(fig, args.output)

    sidecar_path = args.sidecar or os.path.splitext(args.output)[0] + ".json"
    modalities_uploaded = [m for m in MODALITIES if uploads.get(m) is not None]
    result = ExplainabilityResult(
        predicted_class=pred_class,
        confidence=float(confidence),
        probabilities={k: float(v) for k, v in probabilities.items()},
        model_version=model_version,
        modalities_uploaded=modalities_uploaded,
        heatmaps={
            m: {
                "shape": list(map(int, hm.shape)) if hm is not None else None,
                "max": float(np.max(hm)) if hm is not None else None,
                "mean": float(np.mean(hm)) if hm is not None else None,
            }
            for m, hm in heatmaps.items()
        },
        bounding_boxes=bounding_boxes,
        fig_path=os.path.abspath(fig_path),
    )
    save_sidecar(result, sidecar_path)

    # Console summary for the live demo.
    print()
    print("=" * 70)
    print("              FreeBuff v4 — Explainability Output                ")
    print("=" * 70)
    print()
    print(f"  Prediction      : {pred_class.upper()}")
    print(f"  Confidence      : {confidence * 100:.1f}%")
    print()
    print("  Class probabilities:")
    for cls in engine.class_names:
        bar = "#" * int(probabilities[cls] * 40)
        print(f"    {cls:12s} {probabilities[cls] * 100:6.1f}%  {bar}")
    print()
    print("  Explainability heatmaps (saved per modality):")
    for modality in MODALITIES:
        hm = heatmaps.get(modality)
        if hm is None:
            print(f"    {modality}: (not available)")
        else:
            bbox = bounding_boxes.get(modality)
            bbox_str = f"  bbox={bbox}" if bbox else "  bbox=N/A"
            print(f"    {modality}: max={hm.max():.3f}  mean={hm.mean():.3f}{bbox_str}")
    print()
    print(f"  Figure saved       : {fig_path}")
    print(f"  JSON sidecar saved : {sidecar_path}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
