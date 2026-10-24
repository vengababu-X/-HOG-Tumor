"""
FreeBuff v4 — Model Loader (Thread-Safe Singleton)

Manages model lifecycle: loading checkpoints, providing thread-safe inference,
and handling GPU memory. Designed so concurrent FastAPI requests don't collide
on GPU tensors or cause memory leaks.

Key design decisions:
  - Model is loaded once at startup and shared across requests (read-only).
  - Each request creates its own temporary tensors, runs inference inside
    torch.no_grad(), and immediately releases GPU memory.
  - A threading.Lock serializes forward passes to prevent CUDA race conditions
    on shared model parameters.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

# Project imports — these resolve relative to the project root, not api/
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SIZE,
    GRADCAM_TARGET_LAYER_NAME,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_CHECKPOINT_PATH,
    TTA_SOFTMAX_TEMP,
    CUDA_DEVICE_STRING,
    set_all_seeds,
)
from model import build_freebuff_v4, load_checkpoint, GradCAMPlusPlus
from inference import (
    build_tta_transforms,
    confidence_band,
    generate_clinical_reasoning,
    is_low_confidence,
    lookup_who_stage,
)
from preprocessing import preprocess_pil_image, image_to_tensor

logger = logging.getLogger("freebuff_v4.api.model_loader")


class ModelManager:
    """Thread-safe singleton that owns the model and provides inference methods.

    Usage:
        manager = ModelManager.get_instance()
        manager.load(checkpoint_path)
        result = manager.predict(pil_image)
    """

    _instance: Optional["ModelManager"] = None
    _lock_class = threading.Lock()

    def __new__(cls) -> "ModelManager":
        """Ensure only one ModelManager exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock_class:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return the singleton ModelManager instance, creating it if needed.

        On first access, also attempts to auto-load the model checkpoint
        so the API is ready without requiring explicit load() calls.
        """
        instance = cls()
        if not instance.loaded:
            try:
                instance.load()
            except FileNotFoundError:
                logger.warning(
                    "Auto-load: no checkpoint found at default paths. "
                    "Model must be loaded manually via load()."
                )
        return instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Model state
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device(
            CUDA_DEVICE_STRING if torch.cuda.is_available() else "cpu"
        )
        self.loaded: bool = False
        self.model_version: str = "4.0.0"

        # TTA transforms (built once, reused across requests)
        self.tta_transforms = build_tta_transforms()
        self.softmax_temp: float = max(float(TTA_SOFTMAX_TEMP), 1e-4)

        # Inference lock — serializes forward passes to prevent CUDA race
        # conditions when multiple threads share model parameters.
        self._inference_lock = threading.Lock()

        # Server start time (for uptime calculation)
        self._start_time: float = time.time()

        logger.info(
            "ModelManager initialized: device=%s, tta_transforms=%d",
            self.device, len(self.tta_transforms),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint. If None, tries
                DEFAULT_MODEL_SAVE_PATH then DEFAULT_CHECKPOINT_PATH.

        Raises:
            FileNotFoundError: If no valid checkpoint is found.
        """
        if checkpoint_path is None:
            for candidate in [DEFAULT_MODEL_SAVE_PATH, DEFAULT_CHECKPOINT_PATH]:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found. Tried: {DEFAULT_MODEL_SAVE_PATH}, "
                f"{DEFAULT_CHECKPOINT_PATH}"
            )

        logger.info("Loading checkpoint: %s", checkpoint_path)

        # Build model architecture (pretrained=False since we load weights)
        self.model = build_freebuff_v4(
            num_classes=NUM_CLASSES,
            device=self.device,
            pretrained=False,
        )

        # Load weights
        load_checkpoint(checkpoint_path, self.model, device=self.device, strict=False)

        # Move to eval mode — disables dropout and batch norm training behavior
        self.model.eval()

        self.loaded = True
        logger.info(
            "Model loaded successfully on %s (%d parameters)",
            self.device,
            sum(p.numel() for p in self.model.parameters()),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        image: Image.Image,
        use_tta: bool = True,
        generate_gradcam: bool = True,
        class_idx_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run inference on a PIL image.

        This method is thread-safe: it acquires the inference lock, creates
        temporary tensors, runs the forward pass inside torch.no_grad(), and
        releases GPU memory before returning.

        Args:
            image: PIL Image (any mode — will be converted to RGB).
            use_tta: Whether to apply test-time augmentation.
            generate_gradcam: Whether to generate Grad-CAM++ heatmap.
            class_idx_override: Override target class for Grad-CAM.

        Returns:
            Dict with prediction results, probabilities, and Grad-CAM base64.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Acquire lock to serialize CUDA operations
        with self._inference_lock:
            result = self._predict_locked(
                image, use_tta, generate_gradcam, class_idx_override,
            )

        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
        return result

    def _predict_locked(
        self,
        image: Image.Image,
        use_tta: bool,
        generate_gradcam: bool,
        class_idx_override: Optional[int],
    ) -> Dict[str, Any]:
        """Core prediction logic (must be called while holding _inference_lock)."""
        assert self.model is not None

        # Step 1: Preprocess image to tensor
        # Force RGB conversion for model compatibility
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        image_tensor = image_to_tensor(image, size=INPUT_SIZE)

        # Step 2: TTA prediction
        all_logits: List[Tensor] = []
        num_transforms = 0

        with torch.no_grad():
            if use_tta:
                for fn in self.tta_transforms:
                    aug_3d = fn(image_tensor)
                    aug_4d = aug_3d.unsqueeze(0).contiguous().to(self.device)
                    out = self.model(aug_4d)
                    logits = out["logits"].squeeze(0)
                    all_logits.append(logits)
                    num_transforms += 1
            else:
                img_4d = image_tensor.unsqueeze(0).contiguous().to(self.device)
                out = self.model(img_4d)
                logits = out["logits"].squeeze(0)
                all_logits.append(logits)
                num_transforms = 1

        # Step 3: Aggregate TTA logits
        stacked = torch.stack(all_logits, dim=0)
        if use_tta and num_transforms > 1:
            calibrated = stacked / self.softmax_temp
            probs = torch.softmax(calibrated, dim=1).mean(dim=0)
        else:
            probs = torch.softmax(stacked.mean(dim=0) / self.softmax_temp, dim=0)

        probs = probs / probs.sum().clamp_min(1e-8)
        probs_np = probs.cpu().numpy().astype(float)

        pred_idx = int(probs.argmax().item())
        confidence = float(probs_np[pred_idx])

        # Tumor probability (sum of non-"No_Tumor" classes)
        no_tumor_idx = CLASSES.index("No_Tumor")
        tumor_indices = [i for i in range(NUM_CLASSES) if i != no_tumor_idx]
        tumor_prob = float(probs_np[tumor_indices].sum())
        potential_tumor = tumor_prob >= 0.30

        safety_alert = ""
        selected_class = pred_idx
        if potential_tumor and pred_idx == no_tumor_idx:
            # Override: pick the most likely tumor class
            tumor_sub = probs_np[tumor_indices]
            selected_class = tumor_indices[int(np.argmax(tumor_sub))]
            safety_alert = "Potential Tumor Detected: requires radiologist review."

        # Step 4: Grad-CAM++ heatmap
        gradcam_b64 = ""
        if generate_gradcam:
            try:
                gradcam_b64 = self._generate_gradcam_base64(
                    image_tensor, class_idx_override or selected_class,
                )
            except Exception as e:
                logger.warning("Grad-CAM++ generation failed: %s", e)
                # Don't fail the whole request — return prediction without heatmap

        # Build response
        class_probs = [
            {"class_name": CLASSES[i], "probability": float(probs_np[i])}
            for i in range(NUM_CLASSES)
        ]

        # Clinical decision-support fields. These mirror what the Streamlit
        # dashboard shows, so an API consumer gets the same safeguards rather
        # than a bare argmax: the <70% low-confidence flag, the reliability
        # band, the WHO grade reference and the reasoning narrative.
        selected_name = CLASSES[selected_class]
        reported_conf = float(probs_np[selected_class])
        low_conf = is_low_confidence(reported_conf)
        band = confidence_band(reported_conf)
        staging = lookup_who_stage(selected_name)
        try:
            reasoning = generate_clinical_reasoning(
                selected_name, reported_conf, probs_np,
            )
        except Exception as exc:  # narrative must never fail a prediction
            logger.warning("Clinical reasoning generation failed: %s", exc)
            reasoning = ""

        return {
            "success": True,
            "predicted_class": selected_name,
            "predicted_class_index": selected_class,
            "confidence": round(confidence, 6),
            "probabilities": class_probs,
            "tumor_probability": round(tumor_prob, 6),
            "potential_tumor": potential_tumor,
            "safety_alert": safety_alert,
            "low_confidence": low_conf,
            "confidence_band": band,
            "staging": staging,
            "reasoning_text": reasoning,
            "gradcam_heatmap_base64": gradcam_b64,
            "tta_used": use_tta,
            "num_tta_transforms": num_transforms,
            "model_version": self.model_version,
        }

    # ------------------------------------------------------------------
    # Grad-CAM++ heatmap generation
    # ------------------------------------------------------------------
    def _generate_gradcam_base64(
        self,
        image_tensor: Tensor,
        class_idx: int,
    ) -> str:
        """Generate Grad-CAM++ overlay and return as base64-encoded PNG.

        Uses torch.autograd.grad() instead of register_full_backward_hook
        to avoid CUDA illegal memory access errors.

        Args:
            image_tensor: Preprocessed image (3, H, W).
            class_idx: Target class index.

        Returns:
            Base64-encoded PNG string of the overlay image.
        """
        assert self.model is not None

        img = image_tensor.detach().clone().unsqueeze(0).contiguous().to(self.device)

        # Clean GPU cache before Grad-CAM (it needs gradient computation)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        gc = GradCAMPlusPlus(
            self.model,
            target_layer_name=GRADCAM_TARGET_LAYER_NAME,
            device=self.device,
        )

        try:
            heatmap, details = gc.generate_heatmap(img, class_idx=class_idx)
        finally:
            gc.remove_hooks()

        # Create overlay
        overlay_tensor = GradCAMPlusPlus.overlay_heatmap_on_image(
            image_tensor.to(self.device), heatmap,
            alpha=GRADCAM_ALPHA, colormap=GRADCAM_COLORMAP,
        )

        # Convert to PIL
        overlay_np = overlay_tensor.detach().cpu().numpy().astype(np.float32)
        if overlay_np.ndim == 4:
            overlay_np = overlay_np[0]
        overlay_np = np.clip(overlay_np, 0.0, 1.0)
        overlay_np = (overlay_np * 255.0).astype(np.uint8)
        overlay_np = np.transpose(overlay_np, (1, 2, 0))
        overlay_pil = Image.fromarray(overlay_np, mode="RGB")

        # Encode to base64 PNG
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        b64_string = base64.b64encode(buf.read()).decode("utf-8")

        # Clean up
        del heatmap, overlay_tensor, overlay_np, overlay_pil, buf
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return b64_string

    # ------------------------------------------------------------------
    # Health info
    # ------------------------------------------------------------------
    def get_health_info(self) -> Dict[str, Any]:
        """Return current model and system health information."""
        gpu_info: Dict[str, Any] = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_used_mb": None,
        }

        if torch.cuda.is_available():
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            # total_memory is the correct attribute; total_mem was deprecated
            total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            total = total_bytes / (1024 ** 2)
            used = torch.cuda.memory_allocated(0) / (1024 ** 2)
            gpu_info["gpu_memory_total_mb"] = round(total, 1)
            gpu_info["gpu_memory_used_mb"] = round(used, 1)

        return {
            "model_loaded": self.loaded,
            "device": str(self.device),
            "tta_transforms_count": len(self.tta_transforms),
            "uptime_seconds": round(time.time() - self._start_time, 1),
            **gpu_info,
        }
