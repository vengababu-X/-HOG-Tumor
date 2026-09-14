#!/usr/bin/env python3
"""End-to-end test for the FreeBuff v4 FastAPI backend.

Tests every endpoint with real HTTP requests using FastAPI's TestClient.
Creates a synthetic model checkpoint and dummy images for testing.
"""
import sys
import os
import io
import json
import tempfile
import shutil
from pathlib import Path

# Suppress matplotlib GUI
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import set_all_seeds, NUM_CLASSES, INPUT_SIZE, CLASSES
from model import build_freebuff_v4, save_checkpoint

# ---- Step 1: Create a dummy checkpoint (SAFEGUARDED) ----
# FIX: this test previously overwrote saved_models/freebuff_v4.pt — the REAL
# trained model — with random weights. It now saves a dummy checkpoint to a
# temp directory and points the app's config paths at it BEFORE api.main is
# imported, so the production artifact is never touched.
print("=" * 60)
print("STEP 1: Creating dummy model checkpoint (temp dir — real model untouched)...")
set_all_seeds(42)

dummy_ckpt_dir = Path(tempfile.mkdtemp(prefix="freebuff_test_api_"))
dummy_ckpt_path = dummy_ckpt_dir / "freebuff_v4.pt"

model = build_freebuff_v4(num_classes=NUM_CLASSES, device=torch.device("cpu"), pretrained=False)
save_checkpoint(path=dummy_ckpt_path, model=model, epoch=1, val_acc=0.5)
print(f"  Saved dummy checkpoint: {dummy_ckpt_path}")

# Redirect the app's checkpoint resolution to the dummy BEFORE api imports.
import config as _config
_config.DEFAULT_MODEL_SAVE_PATH = dummy_ckpt_path
_config.DEFAULT_CHECKPOINT_PATH = dummy_ckpt_path
print("  Config checkpoint paths redirected to temp dir for this test run.")

# ---- Step 2: Create test images ----
print("\nSTEP 2: Creating test images...")
test_dir = Path(tempfile.mkdtemp())

# Valid MRI-like image (random noise with some structure)
valid_img_path = test_dir / "valid_scan.png"
arr = np.random.randint(10, 245, (200, 200, 3), dtype=np.uint8)
Image.fromarray(arr).save(valid_img_path)
print(f"  Valid image: {valid_img_path}")

# Grayscale image (should be auto-converted to RGB)
gray_img_path = test_dir / "gray_scan.png"
gray_arr = np.random.randint(10, 245, (150, 150), dtype=np.uint8)
Image.fromarray(gray_arr, mode="L").save(gray_img_path)
print(f"  Grayscale image: {gray_img_path}")

# All-black image (should be rejected)
black_img_path = test_dir / "black.png"
black_arr = np.zeros((100, 100, 3), dtype=np.uint8)
Image.fromarray(black_arr).save(black_img_path)
print(f"  Black image (should fail validation): {black_img_path}")

# ---- Step 3: Start test client ----
print("\nSTEP 3: Initializing FastAPI test client...")
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app, raise_server_exceptions=False)
print("  TestClient initialized")

# ---- Step 4: Test GET /health ----
print("\nSTEP 4: Testing GET /health...")
response = client.get("/health")
print(f"  Status: {response.status_code}")
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
data = response.json()
print(f"  Response: {json.dumps(data, indent=2)}")
assert data["status"] in ("healthy", "degraded"), f"Unexpected status: {data['status']}"
assert "model_loaded" in data
assert "device" in data
assert "gpu_available" in data
assert "classes" in data
assert data["classes"] == CLASSES
print("  GET /health: PASS")

# ---- Step 5: Test POST /predict with valid image (TTA + GradCAM) ----
print("\nSTEP 5: Testing POST /predict (valid image, TTA + Grad-CAM)...")
with open(valid_img_path, "rb") as f:
    response = client.post(
        "/predict",
        files={"file": ("scan.png", f, "image/png")},
        data={"use_tta": "true", "generate_gradcam": "true"},
    )
print(f"  Status: {response.status_code}")
assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
data = response.json()
print(f"  predicted_class: {data['predicted_class']}")
print(f"  confidence: {data['confidence']:.4f}")
print(f"  tumor_probability: {data['tumor_probability']:.4f}")
print(f"  potential_tumor: {data['potential_tumor']}")
print(f"  tta_used: {data['tta_used']}")
print(f"  num_tta_transforms: {data['num_tta_transforms']}")
print(f"  processing_time_ms: {data['processing_time_ms']}")
print(f"  gradcam_heatmap_base64 length: {len(data['gradcam_heatmap_base64'])} chars")
assert data["success"] is True
assert data["predicted_class"] in CLASSES
assert 0 <= data["confidence"] <= 1.0
assert len(data["probabilities"]) == NUM_CLASSES
assert len(data["gradcam_heatmap_base64"]) > 0, "Grad-CAM base64 should not be empty"
assert data["num_tta_transforms"] > 1, "TTA should use multiple transforms"
print("  POST /predict (TTA + GradCAM): PASS")

# ---- Step 6: Test POST /predict without TTA ----
print("\nSTEP 6: Testing POST /predict (no TTA, no GradCAM)...")
with open(valid_img_path, "rb") as f:
    response = client.post(
        "/predict",
        files={"file": ("scan.png", f, "image/png")},
        data={"use_tta": "false", "generate_gradcam": "false"},
    )
print(f"  Status: {response.status_code}")
assert response.status_code == 200
data = response.json()
assert data["tta_used"] is False
assert data["num_tta_transforms"] == 1
assert data["gradcam_heatmap_base64"] == "", "GradCAM should be empty when disabled"
print("  POST /predict (no TTA): PASS")

# ---- Step 7: Test POST /predict with grayscale image ----
print("\nSTEP 7: Testing POST /predict (grayscale input)...")
with open(gray_img_path, "rb") as f:
    response = client.post(
        "/predict",
        files={"file": ("gray.png", f, "image/png")},
        data={"use_tta": "false", "generate_gradcam": "false"},
    )
print(f"  Status: {response.status_code}")
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
data = response.json()
assert data["success"] is True
print("  POST /predict (grayscale): PASS")

# ---- Step 8: Test POST /predict with invalid image ----
print("\nSTEP 8: Testing POST /predict (all-black image — should fail)...")
with open(black_img_path, "rb") as f:
    response = client.post(
        "/predict",
        files={"file": ("black.png", f, "image/png")},
        data={"use_tta": "false", "generate_gradcam": "false"},
    )
print(f"  Status: {response.status_code}")
assert response.status_code == 422, f"Expected 422, got {response.status_code}"
data = response.json()
print(f"  Error: {data.get('detail', data.get('error', 'N/A'))}")
print("  POST /predict (black image): PASS (correctly rejected)")

# ---- Step 9: Test POST /predict with corrupt file ----
print("\nSTEP 9: Testing POST /predict (corrupt file)...")
corrupt_bytes = b"this is not an image file at all"
response = client.post(
    "/predict",
    files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
    data={"use_tta": "false", "generate_gradcam": "false"},
)
print(f"  Status: {response.status_code}")
assert response.status_code == 422, f"Expected 422, got {response.status_code}"
print("  POST /predict (corrupt file): PASS (correctly rejected)")

# ---- Step 10: Test POST /predict with wrong content-type ----
print("\nSTEP 10: Testing POST /predict (wrong content-type)...")
response = client.post(
    "/predict",
    files={"file": ("data.bin", b"binary data", "application/octet-stream")},
    data={"use_tta": "false", "generate_gradcam": "false"},
)
print(f"  Status: {response.status_code}")
assert response.status_code == 422, f"Expected 422, got {response.status_code}"
print("  POST /predict (wrong content-type): PASS (correctly rejected)")

# ---- Step 11: Test response schema validation ----
print("\nSTEP 11: Testing response schema validation...")
with open(valid_img_path, "rb") as f:
    response = client.post(
        "/predict",
        files={"file": ("scan.png", f, "image/png")},
        data={"use_tta": "false", "generate_gradcam": "false"},
    )
data = response.json()
# Verify all required fields exist
required_fields = [
    "success", "predicted_class", "predicted_class_index", "confidence",
    "probabilities", "tumor_probability", "potential_tumor", "safety_alert",
    "gradcam_heatmap_base64", "tta_used", "num_tta_transforms",
    "processing_time_ms", "model_version",
]
for field in required_fields:
    assert field in data, f"Missing required field: {field}"
    print(f"  {field}: {type(data[field]).__name__} = {str(data[field])[:60]}")

# Verify probabilities structure
for prob in data["probabilities"]:
    assert "class_name" in prob
    assert "probability" in prob
    assert 0 <= prob["probability"] <= 1.0
print("  Schema validation: PASS")

# ---- Cleanup ----
shutil.rmtree(test_dir, ignore_errors=True)
shutil.rmtree(dummy_ckpt_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("ALL API TESTS PASSED!")
print("=" * 60)
print("\nEndpoints verified:")
print("  [OK] GET  /health        — System health check")
print("  [OK] POST /predict        — Valid image with TTA + Grad-CAM")
print("  [OK] POST /predict        — Valid image without TTA")
print("  [OK] POST /predict        — Grayscale image (auto-converted)")
print("  [OK] POST /predict        — All-black image (rejected 422)")
print("  [OK] POST /predict        — Corrupt file (rejected 422)")
print("  [OK] POST /predict        — Wrong content-type (rejected 422)")
print("  [OK] Response schema      — All required fields validated")
print("\nTo start the server:")
print("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
