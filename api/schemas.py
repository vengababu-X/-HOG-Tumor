"""
FreeBuff v4 — API Schemas (Pydantic Models)

Request and response models for the FastAPI prediction and health endpoints.
These enforce strict type validation on all HTTP payloads.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Class label enum (mirrors config.CLASSES)
# ---------------------------------------------------------------------------
class TumorClass(str, Enum):
    """Valid brain tumor classification labels."""

    glioma = "Glioma"
    meningioma = "Meningioma"
    no_tumor = "No_Tumor"
    pituitary = "Pituitary"


# ---------------------------------------------------------------------------
# Prediction response
# ---------------------------------------------------------------------------
class ClassProbability(BaseModel):
    """Probability for a single class."""

    class_name: str = Field(..., description="Human-readable class label")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax probability"
    )


class WHOStaging(BaseModel):
    """WHO CNS grade reference for a predicted class.

    Mirrors the record returned by inference.lookup_who_stage. This is a
    reference lookup keyed on the predicted class, not a grade predicted by
    the model - the model classifies tumour type, it does not grade tumours.
    """

    grade_label: str = Field("", description="e.g. 'WHO Grade I (Pilocytic Astrocytoma)'")
    typical_imaging_features: str = Field("", description="Typical imaging presentation")
    malignancy: str = Field("", description="Malignancy profile")
    growth_pattern: str = Field("", description="Typical growth pattern")
    recommended_intervention_protocol: str = Field(
        "", description="Reference intervention protocol (not patient advice)"
    )


class PredictionResponse(BaseModel):
    """Response model for POST /predict."""

    success: bool = Field(True, description="Whether prediction succeeded")
    predicted_class: str = Field(
        ..., description="Predicted tumor class label"
    )
    predicted_class_index: int = Field(
        ..., ge=0, description="Integer index of predicted class"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the top prediction"
    )
    probabilities: List[ClassProbability] = Field(
        ..., description="Per-class probability distribution"
    )
    tumor_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate probability of any tumor class",
    )
    potential_tumor: bool = Field(
        ...,
        description="True if tumor_probability exceeds the safety threshold",
    )
    safety_alert: str = Field(
        "",
        description="Non-empty if a safety alert was triggered",
    )
    low_confidence: bool = Field(
        False,
        description=(
            "True when confidence falls below the low-confidence threshold. "
            "The result must then be treated as provisional and referred for "
            "secondary review."
        ),
    )
    confidence_band: str = Field(
        "",
        description="Reliability band: High / Moderate / Low / Very Low",
    )
    staging: Optional[WHOStaging] = Field(
        None,
        description="WHO CNS grade reference for the predicted class",
    )
    reasoning_text: str = Field(
        "",
        description="Human-readable clinical reasoning for the prediction",
    )
    gradcam_heatmap_base64: str = Field(
        "",
        description="Base64-encoded PNG of the Grad-CAM++ heatmap overlay",
    )
    tta_used: bool = Field(
        True, description="Whether Test-Time Augmentation was applied"
    )
    num_tta_transforms: int = Field(
        0, description="Number of TTA transforms applied"
    )
    processing_time_ms: float = Field(
        0.0, description="Total inference time in milliseconds"
    )
    model_version: str = Field("4.0.0", description="Model version string")


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = Field("healthy", description="Service status")
    version: str = Field("4.0.0", description="API version")
    model_loaded: bool = Field(
        ..., description="Whether the model checkpoint is loaded"
    )
    device: str = Field(..., description="Active compute device (cuda:N or cpu)")
    gpu_available: bool = Field(
        ..., description="Whether CUDA GPU is available"
    )
    gpu_name: Optional[str] = Field(
        None, description="GPU device name (if available)"
    )
    gpu_memory_total_mb: Optional[float] = Field(
        None, description="Total GPU VRAM in MB"
    )
    gpu_memory_used_mb: Optional[float] = Field(
        None, description="Currently allocated GPU VRAM in MB"
    )
    tta_transforms_count: int = Field(
        0, description="Number of TTA transforms registered"
    )
    classes: List[str] = Field(
        ..., description="Ordered list of class labels"
    )
    uptime_seconds: float = Field(
        ..., description="Seconds since server started"
    )
    timestamp: str = Field(
        ..., description="ISO-8601 timestamp of the health check"
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Standard error response for all failure modes."""

    success: bool = Field(False)
    error: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(
        None, description="Technical detail for debugging"
    )
