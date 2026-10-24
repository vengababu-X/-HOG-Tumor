#!/usr/bin/env python
"""
FreeBuff v4 — batch prediction over a folder, exported as CSV.

The single-image paths (dashboard, REST API, run.py infer) all handle one scan
at a time. Screening a study folder one upload at a time is not practical, and
`inference.run_batch_inference` existed but was not reachable from any entry
point. This exposes it.

Every row carries the full decision record, not just a label: the four class
probabilities, the aggregate tumour probability, the reliability band, both
safeguard flags and the WHO grade reference — so the CSV can be audited later
without re-running the model.

    python batch_predict.py --input_dir data/Testing/glioma --out predictions.csv
    python batch_predict.py --input_dir scans/ --out out.csv --gradcam_dir overlays/
    python batch_predict.py --input_dir data/Testing --recursive --out all.csv

If the folder is laid out as class subfolders (the dataset layout), pass
--labelled to also score accuracy against the folder names.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from config import CLASSES, DEFAULT_MODEL_SAVE_PATH, SAVED_MODELS_DIR
from inference import FreeBuffInference

logger = logging.getLogger("freebuff_v4.batch")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".nii"}


def collect_images(root: Path, recursive: bool) -> List[Path]:
    """Every supported image under `root`, sorted for reproducible output."""
    it = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in it
             if p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES
                                 or p.name.lower().endswith(".nii.gz"))]
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-classify a folder of MRI scans to CSV.")
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--models_dir", type=Path, default=SAVED_MODELS_DIR)
    ap.add_argument("--recursive", action="store_true",
                    help="Descend into subfolders (use for the dataset layout).")
    ap.add_argument("--labelled", action="store_true",
                    help="Treat the parent folder name as ground truth and report accuracy.")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--volume_aggregation", choices=["mean", "max_tumor"], default="mean",
                    help="How to combine per-slice predictions for NIfTI volumes.")
    ap.add_argument("--middle_slice_only", action="store_true",
                    help="Treat NIfTI volumes as a single middle slice (legacy behaviour).")
    ap.add_argument("--gradcam_dir", type=Path, default=None,
                    help="Also write a Grad-CAM++ overlay PNG per scan into this folder.")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N images.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.input_dir.exists():
        print(f"ERROR: input folder not found: {args.input_dir}", file=sys.stderr)
        return 2

    images = collect_images(args.input_dir, args.recursive)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        print(f"ERROR: no supported images found under {args.input_dir}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("FreeBuff v4 — batch prediction")
    print("=" * 70)
    print(f"input : {args.input_dir}  ({len(images)} images)")
    print(f"output: {args.out}")

    engine = FreeBuffInference.from_dir(args.models_dir)
    print(f"engine: {engine.checkpoint_path}   device={engine.device}   "
          f"TTA={'off' if args.no_tta else 'on'}\n")

    if args.gradcam_dir:
        args.gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Normalised folder name -> canonical class label, for --labelled scoring.
    folder_to_class = {c.lower().replace("_", ""): c for c in CLASSES}

    rows = []
    correct = total_labelled = 0
    failures = 0
    t0 = time.time()

    for i, path in enumerate(images, 1):
        try:
            # A NIfTI file is a whole study: score every informative slice
            # rather than silently keeping only the middle one.
            is_volume = (path.suffix.lower() in (".nii", ".nifti")
                         or path.name.lower().endswith(".nii.gz"))
            if is_volume and not args.middle_slice_only:
                result = engine.predict_volume(
                    path,
                    use_tta=not args.no_tta,
                    patient_id=path.stem,
                    aggregation=args.volume_aggregation,
                )
                arr = None
            else:
                arr = engine.preprocess_path(path)
                result = engine.predict_uploads(
                    {"t1ce": arr, "t2": None, "flair": None},
                    use_tta=not args.no_tta,
                    patient_id=path.stem,
                )
        except Exception as exc:                       # one bad file must not abort the run
            failures += 1
            logger.warning("Skipped %s: %s", path.name, exc)
            rows.append({"file": path.name, "path": str(path), "error": str(exc)})
            continue

        probs = result["probabilities"]
        staging = result.get("staging") or {}
        truth = ""
        if args.labelled:
            truth = folder_to_class.get(path.parent.name.lower().replace("_", ""), "")
            if truth:
                total_labelled += 1
                correct += int(truth == result["predicted_class"])

        row = {
            "file": path.name,
            "path": str(path),
            "predicted_class": result["predicted_class"],
            "confidence": round(float(result["confidence"]), 6),
            "confidence_band": result.get("confidence_band", ""),
            "low_confidence": bool(result.get("low_confidence", False)),
            "tumor_probability": round(float(result.get("tumor_probability", 0.0)), 6),
            "potential_tumor": bool(result.get("potential_tumor", False)),
            "safety_alert": result.get("safety_alert", ""),
            "who_grade": staging.get("typical_grade", "") or staging.get("grade_label", ""),
            "malignancy": (staging.get("malignancy_profile", "")
                           or staging.get("malignancy", ""))[:120],
            "error": "",
            "input_kind": "volume" if is_volume and not args.middle_slice_only else "slice",
            "slices_scored": result.get("n_slices_scored", 1),
        }
        for cls in CLASSES:
            row[f"p_{cls}"] = round(float(probs.get(cls, 0.0)), 6)
        if args.labelled:
            row["true_class"] = truth
            row["correct"] = (truth == result["predicted_class"]) if truth else ""
        rows.append(row)

        if args.gradcam_dir and arr is not None:
            try:
                maps = engine.gradcam_maps({"t1ce": arr, "t2": None, "flair": None})
                key = next(iter(maps))
                overlay = engine.overlay(arr, maps[key])
                from PIL import Image as PILImage
                import numpy as np
                a = (np.clip(np.asarray(overlay), 0, 1) * 255).astype("uint8")
                PILImage.fromarray(a).save(args.gradcam_dir / f"{path.stem}_gradcam.png")
            except Exception as exc:
                logger.warning("Grad-CAM failed for %s: %s", path.name, exc)

        if i % 25 == 0 or i == len(images):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(images)}  ({rate:.1f} img/s)")

    # union of keys, so a run with failures still produces a valid header
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\nwrote {args.out}  ({len(rows)} rows, {elapsed:.1f}s, "
          f"{len(images)/max(elapsed,1e-9):.1f} img/s)")
    if failures:
        print(f"  {failures} file(s) could not be processed — see the 'error' column.")

    n_low = sum(1 for r in rows if r.get("low_confidence") is True)
    n_alert = sum(1 for r in rows if r.get("potential_tumor") is True)
    print(f"  low-confidence flagged : {n_low}")
    print(f"  tumour alert raised    : {n_alert}")
    if args.labelled and total_labelled:
        print(f"  accuracy vs folder names: {correct}/{total_labelled} "
              f"= {correct/total_labelled:.4f}")
    if args.gradcam_dir:
        print(f"  Grad-CAM overlays      : {args.gradcam_dir}")
    print("\nFOR RESEARCH USE ONLY — every row requires verification by a qualified reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
