"""
FreeBuff v4 — API Utilities

Image validation, preprocessing, and helper functions for the FastAPI
prediction endpoint. Handles corrupt images, wrong formats, and ensures
all inputs are safely converted to the format the model expects.
"""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

# Allow truncated images (common with medical scan exports)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("freebuff_v4.api.utils")

# Maximum allowed image dimensions (prevents memory bombs)
MAX_IMAGE_DIMENSION = 4096

# Supported input formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def validate_and_load_image(file_bytes: bytes, filename: Optional[str] = None) -> Image.Image:
    """Validate and load an image from raw bytes.

    Performs robust input validation:
      1. Checks file is not empty.
      2. Attempts PIL decode (catches corrupt files).
      3. Forces RGB conversion (grayscale, RGBA, palette all handled).
      4. Checks dimensions are within safe limits.
      5. Verifies the image contains real content (not all-black/white).

    Args:
        file_bytes: Raw image file bytes.
        filename: Original filename (for extension check).

    Returns:
        Validated PIL Image in RGB mode.

    Raises:
        ValueError: If the image is empty, corrupt, or unsupported.
        IOError: If PIL cannot decode the bytes.
    """
    # --- Check 1: Empty file ---
    if not file_bytes or len(file_bytes) < 100:
        raise ValueError("Uploaded file is empty or too small to be a valid image.")

    # --- Check 2: Extension hint ---
    if filename is not None:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    # --- Check 3: PIL decode ---
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # Force full decode to catch truncated/corrupt files
    except Exception as e:
        raise ValueError(
            f"Failed to decode image. The file may be corrupt or not an image. "
            f"PIL error: {e}"
        ) from e

    # --- Check 4: Convert to RGB ---
    # Medical images often arrive as grayscale (L), palette (P), or RGBA.
    # The model expects 3-channel RGB input.
    original_mode = img.mode
    if img.mode not in ("RGB",):
        if img.mode == "L":
            # Grayscale → RGB by replicating the channel
            img = img.convert("RGB")
        elif img.mode in ("RGBA", "LA"):
            # Composite alpha onto white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

    logger.debug(
        "Image loaded: %dx%d, mode=%s→%s, %d bytes",
        img.width, img.height, original_mode, img.mode, len(file_bytes),
    )

    # --- Check 5: Dimension sanity ---
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions {img.width}x{img.height} exceed maximum "
            f"allowed size of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}."
        )
    if img.width < 16 or img.height < 16:
        raise ValueError(
            f"Image dimensions {img.width}x{img.height} are too small. "
            f"Minimum is 16x16 pixels."
        )

    # --- Check 6: Not all-black or all-white ---
    arr = np.array(img)
    if arr.min() == arr.max():
        raise ValueError(
            "Image appears to be a solid color (all pixels identical). "
            "Please upload a valid MRI scan."
        )

    return img


def image_to_base64_png(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string.

    Args:
        img: PIL Image (any mode — will be converted to RGB).

    Returns:
        Base64-encoded PNG string.
    """
    import base64

    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def resize_for_display(
    img: Image.Image,
    max_width: int = 512,
    max_height: int = 512,
) -> Image.Image:
    """Resize an image for display, preserving aspect ratio.

    Args:
        img: Input PIL Image.
        max_width: Maximum output width.
        max_height: Maximum output height.

    Returns:
        Resized PIL Image (or original if already small enough).
    """
    if img.width <= max_width and img.height <= max_height:
        return img
    ratio = min(max_width / img.width, max_height / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.Resampling.LANCZOS)
