"""Prepare a leakage-safe, cleaned MRI image tree for FreeBuff v4.

Usage:
    python prepare_dataset.py --input data --output data_clean

The script preserves class folders, removes black padding with the shared
foreground crop, applies robust percentile intensity normalization, and writes
224/144-compatible PNG files. It never mixes labels or creates augmented files;
augmentation remains training-only.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from preprocessing import preprocess_pil_image

LOGGER = logging.getLogger("freebuff_v4.prepare_dataset")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield supported images recursively in deterministic order."""
    yield from sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def prepare_dataset(input_root: Path, output_root: Path, overwrite: bool = False) -> int:
    """Clean every image under input_root and return the number written."""
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_root}")
    written = 0
    failures = 0
    for source in iter_images(input_root):
        relative = source.relative_to(input_root)
        destination = output_root / relative.with_suffix(".png")
        if destination.exists() and not overwrite:
            continue
        try:
            with Image.open(source) as image:
                cleaned = preprocess_pil_image(image.convert("RGB"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                cleaned.save(destination, format="PNG", optimize=True)
            written += 1
        except Exception:
            failures += 1
            LOGGER.exception("Failed to prepare %s", source)
    LOGGER.info("Prepared %d images; %d failures; output=%s", written, failures, output_root)
    if written == 0:
        raise RuntimeError(f"No images were prepared from {input_root}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare_dataset(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
