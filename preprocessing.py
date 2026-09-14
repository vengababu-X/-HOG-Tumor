"""MRI preprocessing shared by training, validation, and inference.

The pipeline removes uniform black padding, centers the brain field of view,
normalizes scanner intensity variation, and returns the same ImageNet-normalized
3-channel tensor to every model entry point.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import INPUT_SIZE, INTENSITY_CLIP_MAX, INTENSITY_CLIP_MIN, NORMALIZE_MEAN, NORMALIZE_STD


def crop_brain_field(image: Image.Image, margin_ratio: float = 0.06) -> Image.Image:
    """Crop uniform black padding and retain the largest foreground field.

    The mask is computed from a grayscale copy. A small adaptive threshold
    avoids treating scanner background as brain while preserving low-contrast
    parenchyma. If no reliable foreground is found, the original image is
    returned unchanged.
    """
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    if gray.size == 0 or float(gray.max()) <= 1.0:
        return rgb

    high = float(np.percentile(gray, 99.5))
    threshold = max(3.0, min(30.0, high * 0.04))
    mask = gray > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return rgb

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    height, width = gray.shape
    margin_y = max(1, int((y1 - y0) * margin_ratio))
    margin_x = max(1, int((x1 - x0) * margin_ratio))
    y0 = max(0, y0 - margin_y)
    y1 = min(height, y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(width, x1 + margin_x)
    return rgb.crop((x0, y0, x1, y1))


def normalize_mri_intensity(image: Image.Image) -> Image.Image:
    """Normalize an MRI image with robust percentile clipping to 8-bit RGB."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low = float(np.percentile(gray, 1.0))
    high = float(np.percentile(gray, 99.0))
    if high <= low:
        low = INTENSITY_CLIP_MIN
        high = INTENSITY_CLIP_MAX
    values = np.clip(gray, low, high)
    values = (values - low) / max(high - low, 1e-6)
    return Image.fromarray((values * 255.0).round().astype(np.uint8), mode="L").convert("RGB")


def preprocess_pil_image(image: Image.Image, size: Tuple[int, int] = INPUT_SIZE) -> Image.Image:
    """Crop, robustly normalize, and resize an MRI image for model input."""
    cropped = crop_brain_field(image)
    normalized = normalize_mri_intensity(cropped)
    return normalized.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)


def image_to_tensor(image: Image.Image, size: Tuple[int, int] = INPUT_SIZE) -> torch.Tensor:
    """Return a `(3,H,W)` ImageNet-normalized tensor."""
    prepared = preprocess_pil_image(image, size=size)
    tensor = TF.to_tensor(prepared)
    return TF.normalize(tensor, mean=list(NORMALIZE_MEAN), std=list(NORMALIZE_STD))


def augment_mri_image(image: Image.Image) -> Image.Image:
    """Apply conservative MRI augmentation after foreground cropping.

    Random resized crop supplies center zoom and scale variation; affine
    translation and rotation prevent reliance on fixed skull-edge coordinates.
    """
    image = crop_brain_field(image)
    width, height = image.size
    scale = float(torch.empty(1).uniform_(0.78, 1.0).item())
    ratio = float(torch.empty(1).uniform_(0.92, 1.08).item())
    crop_h = max(1, min(height, int(height * scale)))
    crop_w = max(1, min(width, int(width * scale / ratio)))
    top = int(torch.randint(0, max(1, height - crop_h + 1), (1,)).item())
    left = int(torch.randint(0, max(1, width - crop_w + 1), (1,)).item())
    image = TF.resized_crop(image, top, left, crop_h, crop_w, (height, width), InterpolationMode.BILINEAR)
    angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())
    translate = (int(width * 0.04), int(height * 0.04))
    image = TF.affine(image, angle=angle, translate=(
        int(torch.randint(-translate[0], translate[0] + 1, (1,)).item()),
        int(torch.randint(-translate[1], translate[1] + 1, (1,)).item()),
    ), scale=1.0, shear=(0.0, 0.0), interpolation=InterpolationMode.BILINEAR)
    if bool(torch.rand(1).item() < 0.5):
        image = TF.hflip(image)
    return ImageEnhance.Contrast(image).enhance(float(torch.empty(1).uniform_(0.9, 1.1).item()))
