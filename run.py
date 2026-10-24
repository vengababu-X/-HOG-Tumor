#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Unified Entry Point
# ==============================================================================
# Single command to run the entire system:
#
#   python run.py train          — Train the model with full augmentation
#   python run.py evaluate       — Generate all analytical outputs
#   python run.py infer IMAGE    — Classify a single MRI scan
#   python run.py dashboard      — Launch the Streamlit web dashboard
#   python run.py all            — Train + Evaluate + Generate reports
#   python run.py demo           — Quick smoke-test (no data needed)
#
# All outputs are saved automatically:
#   evaluation_results/visualizations/   — Confusion matrices, ROC, F1, histograms
#   evaluation_results/gradcam_overlays/ — Grad-CAM++ heatmaps
#   clinical_reports/pdf_reports/        — Hospital-grade PDF reports
# ==============================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # torch is imported lazily so `run.py --help` stays fast,
    import torch   # but the annotation on _get_device still needs the name.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freebuff_v4.run")


# ==============================================================================
# DEVICE HELPER
# ==============================================================================
def _resolve_data_dir(given: Optional[str], split: str) -> Path:
    """Resolve a class-folder directory for `split` ("Training" / "Testing").

    `DATA_DIR` points at `data/`, which holds `Training/` and `Testing/`
    subfolders rather than class folders — so defaulting straight to it made
    `run.py train` and `run.py evaluate` fail with "No supported labeled
    images found". Fall through to the split subfolder when it exists.
    """
    from config import DATA_DIR

    if given:
        return Path(given)
    candidate = Path(DATA_DIR) / split
    return candidate if candidate.is_dir() else Path(DATA_DIR)


def _get_device(device_str: Optional[str] = None) -> "torch.device":
    """Resolve the best available torch device."""
    import torch
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. TRAIN
# ==============================================================================
def cmd_train(args: argparse.Namespace) -> None:
    """Full training pipeline: data loading, model build, train loop, checkpoint save."""
    from config import (
        CLASSES, NUM_CLASSES, INPUT_SIZE, CUDA_DEVICE_STRING,
        DEFAULT_MODEL_SAVE_PATH, SAVED_MODELS_DIR, DATA_DIR,
        EVAL_VIS_DIR, GRADCAM_EXPORT_DIR, PDF_REPORT_DIR,
        FOCAL_GAMMA, HEAD_LR, BACKBONE_LR, WEIGHT_DECAY,
        COSINE_T_MAX, COSINE_ETA_MIN, EARLY_STOPPING_PATIENCE,
        NUM_EPOCHS, BATCH_SIZE,
        initialize_directories, set_all_seeds,
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler
    from dataset import BrainTumorDataset, HOGExtractor
    from model import build_freebuff_v4, build_parameter_groups, save_checkpoint, FocalLoss

    device = _get_device(args.device)
    set_all_seeds(42)
    initialize_directories()

    logger.info("Device: %s", device)
    logger.info("Loading training data from: %s", args.data_dir)

    # --- Data ---
    data_dir = _resolve_data_dir(args.data_dir, "Training")
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.info("Expected class subfolders: %s", "/".join(CLASSES))
        sys.exit(1)

    train_ds = BrainTumorDataset(data_dir, split="train", img_size=INPUT_SIZE[0])
    val_ds = BrainTumorDataset(data_dir, split="val", img_size=INPUT_SIZE[0])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # --- Model ---
    model = build_freebuff_v4(num_classes=NUM_CLASSES, device=device,
                              pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    param_groups = build_parameter_groups(model, backbone_lr=BACKBONE_LR, head_lr=HEAD_LR,
                                          weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    scaler = GradScaler()

    # --- Training loop ---
    best_val_acc = 0.0
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_MODEL_SAVE_PATH

    logger.info("Starting training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                out = model(images)
                loss = criterion(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device)
                with torch.amp.autocast(device_type=device.type):
                    out = model(images)
                    loss = criterion(out["logits"], labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = out["logits"].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e | %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_acc, lr_now, elapsed,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # --- Checkpointing ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch,
                            train_loss, val_loss, val_acc)
            logger.info("  -> New best! Saved checkpoint to %s", ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, EARLY_STOPPING_PATIENCE)
            break

    # Save last epoch
    last_path = ckpt_path.parent / (ckpt_path.stem + "_last.pt")
    save_checkpoint(last_path, model, optimizer, scheduler, epoch,
                    train_loss, val_loss, val_acc)
    logger.info("Training complete. Best val_acc=%.4f | Saved: %s", best_val_acc, ckpt_path)


# ==============================================================================
# 2. EVALUATE — Generates ALL analytical outputs
# ==============================================================================
def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run full evaluation and generate all visualizations + PDF report."""
    from config import (
        CLASSES, NUM_CLASSES, DATA_DIR, EVAL_VIS_DIR,
        DEFAULT_MODEL_SAVE_PATH,
        initialize_directories,
    )
    from evaluate import run_evaluation, print_evaluation_summary

    device = _get_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_MODEL_SAVE_PATH
    data_dir = _resolve_data_dir(args.test_data_dir, "Testing")

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    logger.info("Running evaluation...")
    logger.info("  Checkpoint : %s", ckpt_path)
    logger.info("  Test data  : %s", data_dir)
    logger.info("  Output dir : %s", EVAL_VIS_DIR)
    logger.info("  TTA        : %s", not args.no_tta)
    logger.info("  Device     : %s", device)

    # Resolve the HOG branch: on unless --no_hog, and only when the PCA
    # artifact is actually present.
    from config import SAVED_MODELS_DIR
    pca_path = Path(args.pca_path) if args.pca_path else SAVED_MODELS_DIR / "freebuff_pca.pkl"
    use_hog = (not args.no_hog) and pca_path.exists()
    if not use_hog:
        logger.warning("HOG branch disabled (no_hog=%s, pca exists=%s) - "
                       "scores will differ from the documented run.",
                       args.no_hog, pca_path.exists())
        pca_path = None

    results = run_evaluation(
        checkpoint_path=str(ckpt_path),
        test_data_dir=str(data_dir) if data_dir.exists() else None,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
        use_hog=use_hog,
        pca_path=str(pca_path) if pca_path else None,
        device_str=str(device),
        output_dir=EVAL_VIS_DIR,
    )

    print_evaluation_summary(results)
    logger.info("All evaluation outputs saved to: %s", EVAL_VIS_DIR)


# ==============================================================================
# 3. INFER — Classify a single MRI scan with full pipeline
# ==============================================================================
def cmd_infer(args: argparse.Namespace) -> None:
    """Run TTA inference on a single image, generate Grad-CAM++, and produce a PDF report."""
    from config import (
        CLASSES, NUM_CLASSES, INPUT_SIZE,
        CUDA_DEVICE_STRING, DEFAULT_MODEL_SAVE_PATH,
        GRADCAM_EXPORT_DIR, PDF_REPORT_DIR,
    )
    import torch
    from PIL import Image as PILImage
    from inference import run_inference_on_image, lookup_who_stage, confidence_band
    from utils import generate_pdf_report, compute_report_checksum
    from model import build_freebuff_v4, load_checkpoint

    device = _get_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_MODEL_SAVE_PATH

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    # Load model
    model = build_freebuff_v4(num_classes=NUM_CLASSES, device=device, pretrained=False)
    load_checkpoint(ckpt_path, model, device=device, strict=False)

    # A NIfTI file is a whole study, not a slice. Route it to the volumetric
    # path so every informative slice contributes, instead of silently
    # classifying the middle slice and discarding the rest of the scan.
    is_volume = (image_path.suffix.lower() in (".nii", ".nifti")
                 or image_path.name.lower().endswith(".nii.gz"))
    if is_volume and not getattr(args, "middle_slice_only", False):
        from inference import FreeBuffInference

        logger.info("Volume detected — scoring all informative slices: %s", image_path)
        engine = FreeBuffInference.from_dir(Path(ckpt_path).parent)
        t0 = time.time()
        vol = engine.predict_volume(
            image_path,
            use_tta=not args.no_tta,
            patient_id=getattr(args, "patient_id", None),
            aggregation=getattr(args, "volume_aggregation", "mean"),
        )
        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print(f"  Volume       : {image_path.name}")
        print(f"  Slices scored: {vol['n_slices_scored']} / {vol['n_slices_total']}")
        print(f"  Aggregation  : {vol['aggregation']}")
        print(f"  Prediction   : {vol['predicted_class']}")
        print(f"  Confidence   : {vol['confidence']:.2%} ({vol['confidence_band']})")
        print(f"  Tumour prob. : {vol['tumor_probability']:.2%}")
        if vol.get("low_confidence"):
            print("  [!] LOW CONFIDENCE — provisional, secondary review required.")
        if vol.get("safety_alert"):
            print(f"  [!] {vol['safety_alert']}")
        print("\n  Per-class probabilities:")
        for cls, p in vol["probabilities"].items():
            print(f"    {cls:<13} {p:.4f}  {'#' * int(p * 30)}")
        print("\n  Per-slice predictions:")
        for sl in vol["slice_predictions"]:
            print(f"    slice {sl['slice_index']:>3}: {sl['predicted_class']:<11} "
                  f"{sl['confidence']:.3f}")
        print(f"\n  Time         : {elapsed:.2f}s")
        print("=" * 60)
        print("  FOR RESEARCH USE ONLY. NOT FOR CLINICAL DIAGNOSTIC USE.")
        print("=" * 60 + "\n")
        return

    # Run inference
    logger.info("Running inference on: %s", image_path)
    t0 = time.time()
    result = run_inference_on_image(
        model=model,
        image_source=image_path,
        device=device,
        use_tta=not args.no_tta,
        use_hog=args.use_hog,
        gradcam=True,
        gradcam_output_path=GRADCAM_EXPORT_DIR / f"gradcam_{image_path.stem}.png",
    )
    elapsed = time.time() - t0

    # Print results
    pred = result["prediction"]
    conf = result["confidence"]
    probs = result["probabilities"]
    stage = lookup_who_stage(pred)
    band = confidence_band(conf)

    print("\n" + "=" * 60)
    print("  FreeBuff v4 — Classification Result")
    print("=" * 60)
    print(f"  Image        : {image_path.name}")
    print(f"  Prediction   : {pred}")
    print(f"  Confidence   : {conf:.2%} ({band})")
    print(f"  TTA          : {'Yes' if result.get('tta_used') else 'No'}")
    print(f"  Time         : {elapsed:.2f}s")
    print()
    print("  Per-class probabilities:")
    for cls, p in zip(CLASSES, probs):
        bar = "#" * int(p * 40)
        print(f"    {cls:12s}  {p:.4f}  {bar}")
    print()
    print(f"  WHO Grade    : {stage.get('grade_label', 'N/A')}")
    print(f"  Malignancy   : {stage.get('malignancy', 'N/A')}")
    print()

    # Grad-CAM overlay saved
    overlay = result.get("gradcam_overlay")
    if overlay is not None:
        gradcam_path = GRADCAM_EXPORT_DIR / f"gradcam_{image_path.stem}.png"
        print(f"  Grad-CAM++   : {gradcam_path}")

    # Generate PDF report
    if not args.no_pdf:
        pdf_path = generate_pdf_report(
            inference_results={
                "prediction": pred,
                "confidence": conf,
                "probabilities": probs,
                "logits": result.get("logits", []),
                "pred_class_idx": result.get("pred_class_idx", 0),
                "image_source": str(image_path),
                "tta_used": result.get("tta_used", False),
                "grading_text": stage,
                "reasoning_text": result.get("reasoning_text", ""),
            },
            stage_info=stage,
            gradcam_image=overlay,
            patient_id=args.patient_id or f"infer_{image_path.stem}",
            patient_name=args.patient_name or "De-identified",
            exam_date=datetime.now().strftime("%Y-%m-%d"),
            output_dir=PDF_REPORT_DIR,
        )
        checksum = compute_report_checksum(pdf_path)
        print(f"  PDF Report   : {pdf_path}")
        print(f"  Checksum     : {checksum[:16]}...")

    print("=" * 60)
    print("  FOR RESEARCH USE ONLY. NOT FOR CLINICAL DIAGNOSTIC USE.")
    print("=" * 60 + "\n")


# ==============================================================================
# 4. DASHBOARD — Launch Streamlit web UI
# ==============================================================================
def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit interactive clinical dashboard."""
    import subprocess
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        logger.error("app.py not found at: %s", app_path)
        sys.exit(1)

    logger.info("Launching FreeBuff v4 Streamlit Dashboard...")
    logger.info("  URL: http://localhost:8501")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.headless", "false",
           "--server.port", str(args.port)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped.")


# ==============================================================================
# 5. DEMO — Quick smoke test (no data needed)
# ==============================================================================
def cmd_demo(args: argparse.Namespace) -> None:
    """Quick smoke test: builds model, runs dummy inference, verifies pipeline."""
    from config import CLASSES, NUM_CLASSES, INPUT_SIZE, CUDA_DEVICE_STRING
    import torch
    from model import build_freebuff_v4
    from inference import build_tta_transforms, TTAPredictor, preprocess_image_for_inference

    device = _get_device(args.device)
    logger.info("FreeBuff v4 — Quick Demo / Smoke Test")
    logger.info("Device: %s", device)

    # Build model
    model = build_freebuff_v4(num_classes=NUM_CLASSES, device=device, pretrained=True)
    logger.info("Model built successfully. Parameters: %d", sum(p.numel() for p in model.parameters()))

    # Dummy inference with random image
    dummy_img = torch.randn(3, *INPUT_SIZE).to(device)
    logger.info("Running dummy inference...")
    t0 = time.time()
    with torch.no_grad():
        out = model(dummy_img.unsqueeze(0))
    elapsed = time.time() - t0
    logits = out["logits"]
    probs = torch.softmax(logits, dim=1)

    print("\n" + "=" * 60)
    print("  FreeBuff v4 — Smoke Test PASSED")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Inference    : {elapsed:.3f}s")
    print(f"  Output shape : {logits.shape}")
    print(f"  Predicted    : {CLASSES[probs.argmax().item()]} ({probs.max().item():.2%})")
    print("  Classes      :", ", ".join(CLASSES))
    print()
    for cls, p in zip(CLASSES, probs.squeeze().tolist()):
        bar = "#" * int(p * 40)
        print(f"    {cls:12s}  {p:.4f}  {bar}")
    print("=" * 60 + "\n")

    # Verify TTA
    transforms = build_tta_transforms()
    logger.info("TTA transforms: %d augmentations", len(transforms))

    # Verify Grad-CAM
    from model import GradCAMPlusPlus
    gc = GradCAMPlusPlus(model, device=device)
    hm, details = gc.generate_heatmap(dummy_img.unsqueeze(0))
    gc.remove_hooks()
    logger.info("Grad-CAM++ heatmap generated: shape=%s", tuple(hm.shape))

    logger.info("All pipeline components verified successfully!")


# ==============================================================================
# 6. ALL — Train + Evaluate + Generate PDF reports
# ==============================================================================
def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline: train -> evaluate -> generate reports."""
    from config import (
        SAVED_MODELS_DIR, EVAL_VIS_DIR, GRADCAM_EXPORT_DIR, PDF_REPORT_DIR,
    )
    logger.info("=" * 60)
    logger.info("  FreeBuff v4 — Full Pipeline")
    logger.info("=" * 60)

    logger.info("\n[STEP 1/3] Training...")
    cmd_train(args)

    logger.info("\n[STEP 2/3] Evaluation & Visualizations...")
    cmd_evaluate(args)

    logger.info("\n[STEP 3/3] Complete!")
    logger.info("All outputs saved in:")
    logger.info("  Models        : %s", SAVED_MODELS_DIR)
    logger.info("  Visualizations: %s", EVAL_VIS_DIR)
    logger.info("  Grad-CAM++    : %s", GRADCAM_EXPORT_DIR)
    logger.info("  PDF Reports   : %s", PDF_REPORT_DIR)


# ==============================================================================
# CLI
# ==============================================================================
# Defaults used by argparse (avoid importing config at module level)
_DEFAULT_EPOCHS = 60
_DEFAULT_BATCH_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FreeBuff v4 — Multi-Modal Brain Tumor Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py demo                          Quick smoke test (no data)
  python run.py train --data_dir data/Training   Train the model
  python run.py evaluate --checkpoint saved_models/freebuff_v4.pt
  python run.py infer path/to/scan.png --checkpoint saved_models/freebuff_v4.pt
  python run.py dashboard                     Launch Streamlit UI
  python run.py all --data_dir data/Training    Full pipeline
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # --- train ---
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data_dir", type=str, default=None, help="Data root with class subfolders")
    p_train.add_argument("--checkpoint", type=str, default=None, help="Checkpoint save path")
    p_train.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p_train.add_argument("--batch_size", type=int, default=_DEFAULT_BATCH_SIZE)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--pretrained", action="store_true", default=True)
    p_train.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    p_train.add_argument("--freeze_backbone", action="store_true", default=False)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate and generate all visualizations")
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    p_eval.add_argument("--test_data_dir", type=str, default=None, help="Test data directory")
    p_eval.add_argument("--batch_size", type=int, default=_DEFAULT_BATCH_SIZE)
    p_eval.add_argument("--no_tta", action="store_true", help="Disable TTA")
    # The shipped model HAS a HOG branch, and its PCA artifact sits in
    # saved_models/. Defaulting HOG off made this command silently evaluate
    # the deep branch alone and disagree with the documented evaluate.py run.
    p_eval.add_argument("--no_hog", action="store_true",
                        help="Disable the HOG branch (on by default when a PCA artifact exists).")
    p_eval.add_argument("--use_hog", action="store_true",
                        help="Deprecated: HOG is on by default; kept so old commands still run.")
    p_eval.add_argument("--pca_path", type=str, default=None,
                        help="PCA artifact; defaults to saved_models/freebuff_pca.pkl.")
    p_eval.add_argument("--device", type=str, default=None)

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Classify a single MRI scan")
    p_infer.add_argument("image", type=str,
                         help="Path to an MRI slice, or a NIfTI volume (.nii/.nii.gz) "
                              "which is scored slice-by-slice and aggregated.")
    p_infer.add_argument("--volume_aggregation", choices=["mean", "max_tumor"], default="mean",
                         help="How to combine per-slice predictions for a volume. "
                              "'mean' is stable; 'max_tumor' is sensitivity-first for a "
                              "lesion visible on only a few slices.")
    p_infer.add_argument("--middle_slice_only", action="store_true",
                         help="Treat a NIfTI volume as a single middle slice (legacy behaviour).")
    p_infer.add_argument("--checkpoint", type=str, default=None)
    p_infer.add_argument("--no_tta", action="store_true")
    p_infer.add_argument("--use_hog", action="store_true")
    p_infer.add_argument("--no_pdf", action="store_true", help="Skip PDF report")
    p_infer.add_argument("--patient_id", type=str, default=None)
    p_infer.add_argument("--patient_name", type=str, default=None)
    p_infer.add_argument("--device", type=str, default=None)

    # --- dashboard ---
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    # --- all ---
    p_all = sub.add_parser("all", help="Full pipeline: train + evaluate + reports")
    p_all.add_argument("--data_dir", type=str, default=None)
    p_all.add_argument("--checkpoint", type=str, default=None)
    p_all.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p_all.add_argument("--batch_size", type=int, default=_DEFAULT_BATCH_SIZE)
    p_all.add_argument("--device", type=str, default=None)
    p_all.add_argument("--pretrained", action="store_true", default=True)
    p_all.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    p_all.add_argument("--freeze_backbone", action="store_true", default=False)
    p_all.add_argument("--no_tta", action="store_true")

    # --- demo ---
    p_demo = sub.add_parser("demo", help="Quick smoke test (no data needed)")
    p_demo.add_argument("--device", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Lazy init so --help works without torch/numpy installed
    try:
        from config import initialize_directories, set_all_seeds
        initialize_directories()
        set_all_seeds(42)
    except ImportError:
        pass

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "dashboard": cmd_dashboard,
        "all": cmd_all,
        "demo": cmd_demo,
    }
    func = commands.get(args.command)
    if func is None:
        parser.print_help()
        sys.exit(1)

    t0 = time.time()
    try:
        func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Command '%s' failed: %s", args.command, e)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("Completed '%s' in %.1fs.", args.command, elapsed)


if __name__ == "__main__":
    main()
