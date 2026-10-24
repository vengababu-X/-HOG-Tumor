#!/usr/bin/env python3
"""Integration test for frontend.py helper functions.

Tests that the frontend's HTTP client functions correctly communicate
with the FastAPI backend.
"""
import sys
import json
import base64
import tempfile
import shutil
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")

import httpx
import requests
import numpy as np
import torch
from PIL import Image

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import set_all_seeds, NUM_CLASSES, CLASSES, INPUT_SIZE
from model import build_freebuff_v4, save_checkpoint

# ---- Step 1: Create dummy checkpoint (SAFEGUARDED) ----
# FIX: this test previously overwrote saved_models/freebuff_v4.pt — the REAL
# trained model — with random weights. Now it uses a temp checkpoint so the
# production artifact is never touched.
print("=" * 60)
print("STEP 1: Creating dummy checkpoint (temp dir — real model untouched)...")
set_all_seeds(42)
tmp_ckpt_dir = Path(tempfile.mkdtemp(prefix="freebuff_test_ckpt_"))
ckpt_path = tmp_ckpt_dir / "freebuff_v4.pt"
model = build_freebuff_v4(num_classes=NUM_CLASSES, device=torch.device("cpu"), pretrained=False)
save_checkpoint(path=ckpt_path, model=model, epoch=1, val_acc=0.5)
print(f"  Checkpoint: {ckpt_path}  (temporary, not the production model)")

# ---- Step 2: Import frontend helper functions ----
print("\nSTEP 2: Importing frontend helper functions...")
from frontend import (
    check_api_health,
    call_predict_api,
    decode_gradcam_image,
    get_confidence_color,
    get_confidence_label,
)
print("  All imports OK")

# ---- Step 3: Create test images ----
print("\nSTEP 3: Creating test images...")
test_dir = Path(tempfile.mkdtemp())

# Valid image
valid_img = Image.fromarray(np.random.randint(10, 245, (200, 200, 3), dtype=np.uint8))
valid_path = test_dir / "test_scan.png"
valid_img.save(valid_path)

# Grayscale image
gray_img = Image.fromarray(np.random.randint(10, 245, (150, 150), dtype=np.uint8), mode="L")
gray_path = test_dir / "gray_scan.png"
gray_img.save(gray_path)
print(f"  Valid: {valid_path}")
print(f"  Grayscale: {gray_path}")

# ---- Step 4: Test helper functions (no API needed) ----
print("\nSTEP 4: Testing helper functions...")
assert get_confidence_color(0.95) == "#2ecc71"
assert get_confidence_color(0.80) == "#3498db"
assert get_confidence_color(0.65) == "#f39c12"
assert get_confidence_color(0.40) == "#e74c3c"
assert get_confidence_label(0.95) == "High"
assert get_confidence_label(0.80) == "Moderate"
assert get_confidence_label(0.65) == "Low"
assert get_confidence_label(0.40) == "Very Low"
print("  get_confidence_color: PASS")
print("  get_confidence_label: PASS")

# Test base64 decode
test_b64 = base64.b64encode(b"fake PNG data").decode()
result = decode_gradcam_image(test_b64)
# This will return None because the data isn't a valid PNG, but shouldn't crash
print(f"  decode_gradcam_image (invalid PNG): returns {result} (expected None)")

# Test with a real small PNG
buf = BytesIO()
Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(buf, format="PNG")
buf.seek(0)
real_b64 = base64.b64encode(buf.read()).decode()
decoded = decode_gradcam_image(real_b64)
assert decoded is not None
assert decoded.size == (10, 10)
print(f"  decode_gradcam_image (valid PNG): {decoded.size} OK")

# ---- Step 5: Test API communication ----
print("\nSTEP 5: Testing API communication (requires running server)...")
# Check if server is already running
api_url = "http://localhost:8000"
try:
    health = check_api_health(api_url)
    if "error" in health:
        print(f"  API not running at {api_url} — starting test server...")
        # Start server in background.
        # Ask the OS for a free port rather than hardcoding 8001: a previous
        # run leaves TIME_WAIT sockets on that port, so back-to-back runs (as
        # in the consolidated verification) could fail to bind and the whole
        # API leg would be reported as skipped.
        import socket
        import subprocess
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind(("127.0.0.1", 0))
            free_port = _s.getsockname()[1]
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--host", "127.0.0.1", "--port", str(free_port)],
            cwd=str(_project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        api_url = f"http://127.0.0.1:{free_port}"
        print(f"  test server on port {free_port}")
        # Poll for readiness instead of sleeping a fixed interval: loading the
        # checkpoint takes far longer than 5s on a cold cache or a busy
        # machine, and a fixed sleep silently skipped the whole API leg.
        deadline = time.time() + 120
        health = {"error": "not started"}
        while time.time() < deadline:
            if server_proc.poll() is not None:
                health = {"error": f"server exited early ({server_proc.returncode})"}
                break
            health = check_api_health(api_url)
            if "error" not in health:
                break
            time.sleep(2)
        server_started = True
    else:
        server_started = False
except Exception as e:
    print(f"  Could not connect to API: {e}")
    server_started = False
    api_url = None

if api_url and "error" not in health:
    api_tests_ran = True
    # Test health check
    print(f"  Health check: {json.dumps(health, indent=2)}")
    assert health.get("model_loaded", False), "Model should be loaded"
    print("  Health check: PASS")

    # Test prediction with valid image
    print("\n  Testing /predict with valid image (TTA + GradCAM)...")
    with open(valid_path, "rb") as f:
        result = call_predict_api(f.read(), "scan.png", use_tta=True, generate_gradcam=True, api_url=api_url)

    print(f"    predicted_class: {result['predicted_class']}")
    print(f"    confidence: {result['confidence']:.4f}")
    print(f"    tumor_probability: {result['tumor_probability']:.4f}")
    print(f"    num_tta_transforms: {result['num_tta_transforms']}")
    print(f"    gradcam length: {len(result['gradcam_heatmap_base64'])} chars")
    assert result["success"] is True
    assert result["predicted_class"] in CLASSES
    assert 0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == NUM_CLASSES
    assert len(result["gradcam_heatmap_base64"]) > 0
    print("    /predict (TTA + GradCAM): PASS")

    # Test prediction without TTA
    print("\n  Testing /predict without TTA...")
    with open(valid_path, "rb") as f:
        result2 = call_predict_api(f.read(), "scan.png", use_tta=False, generate_gradcam=False, api_url=api_url)
    assert result2["tta_used"] is False
    assert result2["num_tta_transforms"] == 1
    assert result2["gradcam_heatmap_base64"] == ""
    print("    /predict (no TTA): PASS")

    # Test GradCAM image decoding
    if result["gradcam_heatmap_base64"]:
        decoded = decode_gradcam_image(result["gradcam_heatmap_base64"])
        assert decoded is not None
        print(f"\n  GradCAM decoded: {decoded.size[0]}x{decoded.size[1]} pixels")
        print("  GradCAM decode: PASS")

    # Test error handling
    print("\n  Testing error handling...")
    try:
        with open(valid_path, "rb") as f:
            call_predict_api(b"corrupt", "bad.png", use_tta=False, generate_gradcam=False, api_url=api_url)
        print("    Should have raised HTTPStatusError!")
    except requests.exceptions.HTTPError as e:
        assert e.response.status_code == 422
        print(f"    Corrupt file correctly rejected (422): PASS")

    if server_started:
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("\n  Test server stopped.")
else:
    api_tests_ran = False
    print("  Skipping API tests (server not available)")
    print("  To test: start server with 'uvicorn api.main:app --port 8001'")

# Cleanup
shutil.rmtree(test_dir, ignore_errors=True)
shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("ALL FRONTEND TESTS PASSED!" if api_tests_ran
      else "LOCAL TESTS PASSED - API TESTS WERE SKIPPED")
print("=" * 60)
print("\nComponents verified:")
print("  [OK] Helper functions (confidence color/label)")
print("  [OK] Base64 image encoding/decoding")
_api = "[OK]" if api_tests_ran else "[SKIPPED]"
print(f"  {_api} API health check communication")
print(f"  {_api} API prediction with TTA + Grad-CAM")
print(f"  {_api} API prediction without TTA")
print(f"  {_api} Error handling for corrupt files")
print(f"  {_api} Grad-CAM heatmap decoding")

if not api_tests_ran:
    # A skipped leg must never look like a pass: exit non-zero so neither
    # CI nor a human reader mistakes "not run" for "verified".
    print("\nFAIL: the API leg never ran, so its results are unknown.")
    sys.exit(1)
