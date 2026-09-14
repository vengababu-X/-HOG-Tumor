"""
FreeBuff v4 — FastAPI Backend Server

Production-ready API for brain tumor classification with TTA inference
and Grad-CAM++ explainability.

Start the server:
    cd project_root
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    Or with auto-reload during development:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict  — Upload an MRI image, get classification + Grad-CAM overlay
    GET  /health   — System health check with GPU/CPU status
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path for local imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.schemas import (
    ClassProbability,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from api.model_loader import ModelManager
from api.utils import validate_and_load_image

logger = logging.getLogger("freebuff_v4.api")


# =============================================================================
# LIFESPAN (replaces deprecated on_startup / on_shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Context manager that handles server startup and shutdown.

    On startup: loads the model checkpoint so the API is ready immediately.
    On shutdown: releases GPU memory.
    """
    # --- Startup ---
    logger.info("=" * 60)
    logger.info("FreeBuff v4 API starting up...")
    logger.info("=" * 60)

    try:
        manager = ModelManager.get_instance()
        manager.load()
        logger.info("Model loaded successfully. API is ready to serve requests.")
    except FileNotFoundError as e:
        logger.error(
            "CRITICAL: Could not load model checkpoint. "
            "The /predict endpoint will return 503 until a valid checkpoint "
            "is placed at saved_models/freebuff_v4.pt. Error: %s", e,
        )
    except Exception as e:
        logger.error("CRITICAL: Unexpected error loading model: %s", e, exc_info=True)

    yield  # Server is running

    # --- Shutdown ---
    logger.info("Shutting down FreeBuff v4 API...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")


# =============================================================================
# APP CONFIGURATION
# =============================================================================
app = FastAPI(
    title="FreeBuff v4 — Brain Tumor Classification API",
    description=(
        "Multi-modal brain tumor classification with Test-Time Augmentation (TTA) "
        "and Grad-CAM++ explainability. Accepts MRI scan images and returns "
        "class predictions with confidence scores and visual explanations."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Convert ValueError to a 422 Unprocessable Entity response."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=str(exc),
            detail="The uploaded image could not be processed.",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


# =============================================================================
# ROUTES
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="System health check",
)
async def health_check() -> HealthResponse:
    """Return system health, GPU status, and model availability.

    This endpoint is lightweight and does NOT require the model to be loaded.
    Use it for load balancer health checks, monitoring, and debugging.
    """
    manager = ModelManager.get_instance()
    info = manager.get_health_info()

    return HealthResponse(
        status="healthy" if info["model_loaded"] else "degraded",
        version="4.0.0",
        model_loaded=info["model_loaded"],
        device=info["device"],
        gpu_available=info["gpu_available"],
        gpu_name=info["gpu_name"],
        gpu_memory_total_mb=info["gpu_memory_total_mb"],
        gpu_memory_used_mb=info["gpu_memory_used_mb"],
        tta_transforms_count=info["tta_transforms_count"],
        classes=["Glioma", "Meningioma", "No_Tumor", "Pituitary"],
        uptime_seconds=info["uptime_seconds"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify a brain MRI scan",
    responses={
        200: {"model": PredictionResponse, "description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Invalid image input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(
    file: UploadFile = File(
        ..., description="MRI image file (JPEG, PNG, BMP, TIFF)"
    ),
    use_tta: bool = Form(
        default=True,
        description="Enable Test-Time Augmentation (improves accuracy, ~48x slower)",
    ),
    generate_gradcam: bool = Form(
        default=True,
        description="Generate Grad-CAM++ explainability heatmap",
    ),
    class_idx_override: Optional[int] = Form(
        default=None,
        description="Override target class for Grad-CAM (None = use prediction)",
    ),
) -> PredictionResponse:
    """Upload a brain MRI scan and receive a classification with explainability.

    **Input:** A single image file (JPEG, PNG, BMP, TIFF).

    **Output:**
    - Predicted class label (Glioma, Meningioma, No_Tumor, Pituitary)
    - Confidence score and per-class probability distribution
    - Base64-encoded Grad-CAM++ heatmap overlay (if enabled)
    - Safety alerts for potential tumor detection

    **Processing pipeline:**
    1. Validate and load the uploaded image (format, corruption, dimensions)
    2. Preprocess: crop brain field, normalize intensity, resize to 144×144
    3. Run TTA inference: 48 augmented views aggregated for robust prediction
    4. Generate Grad-CAM++ heatmap using torch.autograd.grad() (CUDA-safe)
    5. Return structured JSON response with all results
    """
    # --- Step 1: Validate model is loaded ---
    manager = ModelManager.get_instance()
    if not manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a valid checkpoint exists "
                   "at saved_models/freebuff_v4.pt and restart the server.",
        )

    # --- Step 2: Read and validate the uploaded file ---
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Expected an image file, got content-type: {file.content_type}",
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read uploaded file: {e}",
        )

    try:
        pil_image = validate_and_load_image(file_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Step 3: Run inference ---
    try:
        result = manager.predict(
            image=pil_image,
            use_tta=use_tta,
            generate_gradcam=generate_gradcam,
            class_idx_override=class_idx_override,
        )
    except RuntimeError as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # --- Step 4: Build response ---
    class_probs = [
        ClassProbability(class_name=cp["class_name"], probability=cp["probability"])
        for cp in result["probabilities"]
    ]

    return PredictionResponse(
        success=result["success"],
        predicted_class=result["predicted_class"],
        predicted_class_index=result["predicted_class_index"],
        confidence=result["confidence"],
        probabilities=class_probs,
        tumor_probability=result["tumor_probability"],
        potential_tumor=result["potential_tumor"],
        safety_alert=result["safety_alert"],
        gradcam_heatmap_base64=result["gradcam_heatmap_base64"],
        tta_used=result["tta_used"],
        num_tta_transforms=result["num_tta_transforms"],
        processing_time_ms=result["processing_time_ms"],
        model_version=result["model_version"],
    )


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "Starting FreeBuff v4 API server — "
        "ALL inference runs locally (no external API keys required)."
    )

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
    )
