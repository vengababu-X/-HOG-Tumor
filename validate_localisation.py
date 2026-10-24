#!/usr/bin/env python
"""
FreeBuff v4 — is the explanation map good enough to measure the lesion?

WHY THIS EXISTS
---------------
The obvious next feature after "what type of tumour is this?" is "what WHO
grade is it?". Grade is, in principle, estimable from imaging: central
necrosis, enhancement heterogeneity, peritumoral oedema, margin definition and
lesion size are the criteria radiologists use before biopsy, and they are all
quantities you can compute from pixels.

Every one of those measurements needs the lesion to be *localised* first. This
project's explanation maps (Grad-CAM++, Score-CAM, CBAM attention rollout) look
like localisation, so the natural move is to threshold one and measure inside
it. This script tests whether that is actually valid — before anything is built
on top of it.

THE TEST
--------
Threshold the map to get a region of interest, then compare the image
statistics inside that ROI against a **size-matched random region of the same
brain**. If the map genuinely finds the lesion, the ROI must look different
from a random patch. If it does not, any radiomics computed inside it is
measuring noise.

A second check confirms the size problem: thresholding a map at the Nth
percentile yields a region of exactly (100-N)% of the brain, by construction.
If "lesion size" tracks the threshold rather than the image, it is a tautology,
not a measurement.

MEASURED RESULT ON THIS MODEL
-----------------------------
    method                        ROI mean   random mean   |diff|
    Grad-CAM++ (layer4)              0.446         0.421    0.025
    Score-CAM                        0.428         0.418    0.011
    CBAM attention rollout           0.409         0.419    0.010

None reach the 0.05 separation threshold. The backbone's layer4 feature map is
5x5 for a 144x144 input, so an upsampled map is far too coarse to delimit a
lesion boundary.

CONCLUSION: imaging-based WHO grading is NOT implementable on this system as
it stands. It needs either lesion segmentation masks (which this dataset does
not ship) or a segmentation model trained for the purpose. An estimator built
on these maps would produce confident-looking numbers with no measurable basis,
which in a clinical-adjacent tool is worse than returning nothing.

This is why the system reports WHO grade as a *reference lookup for the
predicted tumour type* and states plainly that it is not a patient-specific
grade. That wording is a measured conclusion, not caution for its own sake.

Usage
-----
    python validate_localisation.py
    python validate_localisation.py --n_per_class 20 --threshold_pct 85
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from config import CLASSES, SAVED_MODELS_DIR

logger = logging.getLogger("freebuff_v4.validate_localisation")

SEPARATION_THRESHOLD = 0.05   # |ROI mean - random mean| below this = no localisation


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    rng = float(a.max() - a.min())
    return (a - a.min()) / rng if rng > 0 else a


def evaluate_method(
    engine,
    map_fn: Callable[[np.ndarray], Optional[np.ndarray]],
    paths: List[str],
    threshold_pct: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Compare ROI statistics against size-matched random regions."""
    roi_means, rand_means, roi_hets, rand_hets, areas = [], [], [], [], []
    for p in paths:
        arr = engine.preprocess_path(p)
        hm = map_fn(arr)
        if hm is None:
            continue
        hm = _normalise(hm)
        img = _normalise(np.asarray(arr))
        brain = img > 0.10
        if brain.sum() < 100:
            continue
        thr = np.percentile(hm[brain], threshold_pct)
        roi = (hm >= thr) & brain
        n = int(roi.sum())
        if n < 20:
            continue

        roi_means.append(float(img[roi].mean()))
        roi_hets.append(float(img[roi].std() / max(img[roi].mean(), 1e-6)))
        areas.append(100.0 * n / float(brain.sum()))

        idx = np.flatnonzero(brain.ravel())
        pick = rng.choice(idx, n, replace=False)
        mask = np.zeros(img.size, dtype=bool)
        mask[pick] = True
        mask = mask.reshape(img.shape)
        rand_means.append(float(img[mask].mean()))
        rand_hets.append(float(img[mask].std() / max(img[mask].mean(), 1e-6)))

    if not roi_means:
        return {}
    return {
        "roi_mean": float(np.mean(roi_means)),
        "rand_mean": float(np.mean(rand_means)),
        "diff": float(abs(np.mean(roi_means) - np.mean(rand_means))),
        "roi_het": float(np.mean(roi_hets)),
        "rand_het": float(np.mean(rand_hets)),
        "area_pct": float(np.mean(areas)),
        "n": len(roi_means),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test whether explanation maps localise the lesion well enough for radiomics.")
    ap.add_argument("--models_dir", type=Path, default=SAVED_MODELS_DIR)
    ap.add_argument("--data_dir", type=Path, default=Path("data/Testing"))
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--threshold_pct", type=float, default=90.0,
                    help="Percentile of the map used as the ROI cut.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from inference import FreeBuffInference
    engine = FreeBuffInference.from_dir(args.models_dir)
    rng = np.random.RandomState(args.seed)

    paths: List[str] = []
    for cls_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        paths += sorted(str(p) for p in cls_dir.iterdir() if p.is_file())[:args.n_per_class]
    if not paths:
        print(f"ERROR: no images under {args.data_dir}")
        return 2

    print("=" * 74)
    print("FreeBuff v4 — lesion localisation validation")
    print("=" * 74)
    print(f"model  : {engine.checkpoint_path}")
    print(f"images : {len(paths)} from {args.data_dir}")
    print(f"ROI cut: top {100 - args.threshold_pct:.0f}% of each map\n")

    methods = [
        ("Grad-CAM++ (layer4)",
         lambda a: engine.gradcam_maps({"t1ce": a, "t2": None, "flair": None}).get("t1ce")),
        ("Score-CAM",
         lambda a: engine.score_cam_maps({"t1ce": a, "t2": None, "flair": None}).get("t1ce")),
        ("CBAM attention rollout",
         lambda a: engine.attention_rollout_maps({"t1ce": a, "t2": None, "flair": None}).get("t1ce")),
    ]

    print(f"{'method':<26}{'ROI mean':>10}{'rand mean':>11}{'|diff|':>9}"
          f"{'ROI area%':>11}{'verdict':>16}")
    print("-" * 84)
    any_pass = False
    for name, fn in methods:
        try:
            r = evaluate_method(engine, fn, paths, args.threshold_pct, rng)
        except Exception as exc:
            print(f"{name:<26}  failed: {exc}")
            continue
        if not r:
            print(f"{name:<26}  no measurable ROI")
            continue
        ok = r["diff"] > SEPARATION_THRESHOLD
        any_pass |= ok
        print(f"{name:<26}{r['roi_mean']:>10.3f}{r['rand_mean']:>11.3f}{r['diff']:>9.3f}"
              f"{r['area_pct']:>11.1f}{('LOCALISES' if ok else 'no separation'):>16}")

    print("-" * 84)
    print(f"\nSeparation threshold: |diff| > {SEPARATION_THRESHOLD}")
    if any_pass:
        print("\nAt least one map localises the lesion measurably. Radiomics-based")
        print("grading may be viable using that map — validate per class before use.")
    else:
        print("\nNo map separates the lesion from a random region of the same size.")
        print("Radiomics computed inside these ROIs would measure noise, so")
        print("imaging-based WHO grading is NOT implementable on this system.")
        print("\nWhy: the backbone's layer4 map is 5x5 for a 144x144 input. Upsampling")
        print("that to the image cannot delimit a lesion boundary. Grading would need")
        print("segmentation masks, which this dataset does not ship.")
        print("\nNote the ROI area column: thresholding at the Nth percentile always")
        print("yields (100-N)% of the brain, so 'lesion size' from these maps is a")
        print("property of the threshold, not of the image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
