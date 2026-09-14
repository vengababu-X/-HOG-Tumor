#!/usr/bin/env python3
"""End-to-end smoke test for run_pipeline.py using synthetic data.

Creates a fake dataset, builds a model, runs training, TTA, Grad-CAM++,
and evaluation — all on CPU with tiny data to verify no crashes occur.
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Suppress matplotlib GUI
import matplotlib
matplotlib.use("Agg")

from config import set_all_seeds, CLASSES, NUM_CLASSES, INPUT_SIZE, HOG_PCA_COMPONENTS
from model import build_freebuff_v4, GradCAMPlusPlus
from dataset import BrainTumorDataset, HOGExtractor, make_dataloaders
from run_pipeline import (
    assert_tensor_safety,
    assert_model_io_sanity,
    safe_gradcam_generate,
    SafeTTAPredictor,
    generate_gradcam_overlay,
    evaluate_comprehensive,
    train_model,
)

set_all_seeds(42)
device = torch.device("cpu")

# ---- Step 1: Create synthetic dataset ----
print("=" * 60)
print("STEP 1: Creating synthetic dataset...")
tmpdir = Path(tempfile.mkdtemp())
for cls_name in CLASSES:
    cls_dir = tmpdir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        # Create a tiny random PNG image
        from PIL import Image
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.png")

print(f"  Created {len(CLASSES) * 20} synthetic images in {tmpdir}")

# ---- Step 2: Build model and verify I/O shapes ----
print("\nSTEP 2: Building model and verifying I/O shapes...")
model = build_freebuff_v4(num_classes=NUM_CLASSES, device=device, pretrained=False)
dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
shapes = assert_model_io_sanity(model, dummy_input)
print(f"  Model I/O: {shapes}")

# ---- Step 3: Test tensor safety assertions ----
print("\nSTEP 3: Testing tensor safety assertions...")
t = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
assert_tensor_safety(t, "test_tensor", expected_ndim=4)
print("  assert_tensor_safety: PASS")

try:
    bad_tensor = torch.randn(1, 5)
    assert_tensor_safety(bad_tensor, "bad_tensor", expected_ndim=4)
    print("  Should have raised ValueError!")
    sys.exit(1)
except ValueError as e:
    print(f"  Correctly caught shape mismatch: {e}")

# ---- Step 4: Test Grad-CAM++ (safe wrapper) ----
print("\nSTEP 4: Testing Grad-CAM++ safe wrapper...")
model.eval()
heatmap, details = safe_gradcam_generate(model, dummy_input.squeeze(0), device, class_idx=0)
print(f"  Heatmap shape: {list(heatmap.shape)}")
print(f"  Details: class_idx={details['class_idx']}")
assert heatmap.shape == (1, 1, INPUT_SIZE[0], INPUT_SIZE[1]), f"Wrong heatmap shape: {heatmap.shape}"
print("  Grad-CAM++ safe wrapper: PASS")

# ---- Step 5: Test Grad-CAM++ overlay ----
print("\nSTEP 5: Testing Grad-CAM++ overlay generation...")
overlay_pil, hm = generate_gradcam_overlay(
    model, dummy_input.squeeze(0), device, class_idx=0,
    output_path=tmpdir / "test_overlay.png",
)
print(f"  Overlay PIL size: {overlay_pil.size}")
assert overlay_pil.size == (INPUT_SIZE[1], INPUT_SIZE[0])
print("  Overlay generation: PASS")

# ---- Step 6: Test SafeTTAPredictor ----
print("\nSTEP 6: Testing SafeTTAPredictor...")
predictor = SafeTTAPredictor(model, device)
result = predictor.predict_single(dummy_input.squeeze(0))
print(f"  Predicted class: {result['pred_class']} ({CLASSES[result['pred_class']]})")
print(f"  Confidence: {result['confidence']:.4f}")
print(f"  TTA transforms used: {len(predictor.transforms)}")
print(f"  Probs shape: {list(result['avg_probs'].shape)}")
assert result["avg_probs"].shape == (NUM_CLASSES,)
assert 0 <= result["confidence"] <= 1.0
print("  SafeTTAPredictor: PASS")

# ---- Step 7: Test batch TTA prediction ----
print("\nSTEP 7: Testing batch TTA prediction...")
batch_input = torch.randn(4, 3, INPUT_SIZE[0], INPUT_SIZE[1])
batch_result = predictor.predict_batch(batch_input)
print(f"  Batch pred_classes shape: {list(batch_result['pred_classes'].shape)}")
assert batch_result["pred_classes"].shape == (4,)
assert batch_result["avg_probs"].shape == (4, NUM_CLASSES)
print("  Batch TTA: PASS")

# ---- Step 8: Create DataLoader-based dataset for evaluation ----
print("\nSTEP 8: Building DataLoader from synthetic dataset...")
from dataset import build_dataset_from_folder, split_dataset_into_train_val_test
full_ds = build_dataset_from_folder(tmpdir)
train_ds, val_ds, test_ds = split_dataset_into_train_val_test(full_ds, val_frac=0.2, test_frac=0.2, seed=42)

train_samples = [train_ds.dataset.samples_list[i] for i in train_ds.indices]
val_samples = [val_ds.dataset.samples_list[i] for i in val_ds.indices]
test_samples = [test_ds.dataset.samples_list[i] for i in test_ds.indices]

train_dataset = BrainTumorDataset(samples=train_samples, use_hog=False)
val_dataset = BrainTumorDataset(samples=val_samples, use_hog=False)
test_dataset = BrainTumorDataset(samples=test_samples, use_hog=False)

train_loader, val_loader, test_loader = make_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=0,
)
print(f"  Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ---- Step 9: Quick training (2 epochs) ----
print("\nSTEP 9: Running 2-epoch training...")
train_result = train_model(
    model, train_loader, val_loader, device,
    num_epochs=2, patience=3, use_amp=False,
)
print(f"  Best val_f1: {train_result['best_val_f1']:.4f}")
print(f"  Total epochs: {train_result['total_epochs']}")
print("  Training: PASS")

# ---- Step 10: Full evaluation ----
print("\nSTEP 10: Running comprehensive evaluation...")
eval_output_dir = tmpdir / "eval_output"
eval_results = evaluate_comprehensive(
    model, test_loader, device,
    use_tta=True,
    output_dir=eval_output_dir,
)
print(f"  Accuracy: {eval_results['accuracy']:.4f}")
print(f"  Weighted F1: {eval_results['weighted_f1']:.4f}")
print(f"  Macro F1: {eval_results['macro_f1']:.4f}")
print(f"  Confusion matrix shape: {eval_results['confusion_matrix_counts'].shape}")
print(f"  ROC-AUCs: {eval_results['roc_aucs']}")
print(f"  Grad-CAM overlays saved: {eval_results['gradcam_saved_count']}")

# Verify output files exist
expected_files = [
    "cm_counts.png", "cm_normalized.png", "per_class_f1.png",
    "roc_curves.png", "confidence_dist.png",
]
for fname in expected_files:
    fpath = eval_output_dir / fname
    assert fpath.exists(), f"Missing output file: {fpath}"
    print(f"  {fname}: EXISTS ({fpath.stat().st_size} bytes)")

print("  Evaluation: PASS")

# ---- Cleanup ----
shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nComponents verified:")
print("  [OK] Tensor safety assertions")
print("  [OK] Model I/O shape validation")
print("  [OK] Grad-CAM++ safe wrapper (torch.autograd.grad)")
print("  [OK] Grad-CAM++ overlay generation")
print("  [OK] SafeTTAPredictor (single image)")
print("  [OK] SafeTTAPredictor (batch)")
print("  [OK] Training loop (fixed FocalLoss)")
print("  [OK] Comprehensive evaluation")
print("  [OK] Confusion matrix, ROC, F1 plots")
print("  [OK] Grad-CAM overlays for misclassifications")
