#!/usr/bin/env python
"""
FreeBuff v4 — dataset de-duplication.

`audit_leakage.py` measures the problem; this script removes it.

Two defects exist in the public dataset as shipped:

  1. 114 images in data/Testing are byte-identical to images in data/Training
     (100 of them meningioma, i.e. 25% of that class's test set). A model that
     memorises those scores on them for free.
  2. data/Training contains 190 duplicate images of its own, which silently
     re-weights those samples during training.

The fix keeps the TEST set whole and removes the offending copies from the
TRAINING set instead. That way the held-out set stays at its full 1,600 images
(so results remain comparable to the pre-fix numbers) while every one of those
images becomes genuinely unseen during training.

Output is written to a NEW directory; the original data/ is never modified.

    python deduplicate_dataset.py                    # dry run, prints the plan
    python deduplicate_dataset.py --apply            # writes data_clean/

Then retrain against it:

    python train.py --data_dir data_clean/Training --checkpoint_dir checkpoints/clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from config import CLASSES
from dataset import build_dataset_from_folder


def pixel_hash(path: str | Path) -> str:
    """SHA-1 of the decoded greyscale pixel array (plus shape).

    Hashing decoded pixels rather than file bytes means two files that differ
    only in JPEG metadata still register as the same image.
    """
    arr = np.asarray(Image.open(path).convert("L"))
    return hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove duplicate leakage from the dataset.")
    ap.add_argument("--train_dir", type=Path, default=Path("data/Training"))
    ap.add_argument("--test_dir", type=Path, default=Path("data/Testing"))
    ap.add_argument("--out_dir", type=Path, default=Path("data_clean"))
    ap.add_argument("--near_threshold", type=float, default=0.999,
                    help="Also drop training images whose pixel correlation to any TEST "
                         "image reaches this. 0 disables. Default 0.999 — measured: "
                         "genuinely different scans sit near 0.91, so this removes "
                         "re-encoded/shifted copies without touching real variation.")
    ap.add_argument("--apply", action="store_true", help="Actually write the clean dataset.")
    args = ap.parse_args()

    train_samples = build_dataset_from_folder(args.train_dir).samples_list
    test_samples = build_dataset_from_folder(args.test_dir).samples_list

    print("=" * 72)
    print("FreeBuff v4 — dataset de-duplication")
    print("=" * 72)
    print(f"source train: {len(train_samples)} images")
    print(f"source test : {len(test_samples)} images\n")

    print("hashing test set ...")
    test_hashes = {pixel_hash(s["image_path"]) for s in test_samples}

    print("hashing training set ...")
    keep: List[Dict] = []
    dropped_leak = defaultdict(int)
    dropped_intra = defaultdict(int)
    seen: set[str] = set()

    for s in train_samples:
        h = pixel_hash(s["image_path"])
        cls = CLASSES[s["label"]]
        if h in test_hashes:                      # leaks into the held-out set
            dropped_leak[cls] += 1
            continue
        if h in seen:                             # duplicate within training
            dropped_intra[cls] += 1
            continue
        seen.add(h)
        keep.append(s)

    n_leak = sum(dropped_leak.values())
    n_intra = sum(dropped_intra.values())

    # ---- near-duplicate pass -------------------------------------------------
    # An exact hash misses a scan that was re-encoded or shifted by a pixel, and
    # those are still the same study. Pixel correlation on a 64x64 downsample
    # separates them cleanly: genuinely different scans sit around 0.91, while
    # re-encoded copies sit above 0.999.
    dropped_near = defaultdict(int)
    if args.near_threshold and args.near_threshold > 0 and keep:
        print(f"\nnear-duplicate pass (threshold {args.near_threshold}) ...")

        def vectorise(paths: List[str], n: int = 64) -> np.ndarray:
            out = np.zeros((len(paths), n * n), dtype=np.float32)
            for i, p in enumerate(paths):
                a = np.asarray(Image.open(p).convert("L").resize((n, n)),
                               dtype=np.float32).ravel()
                a -= a.mean()
                norm = float(np.linalg.norm(a))
                out[i] = a / norm if norm > 0 else a
            return out

        test_vecs = vectorise([s["image_path"] for s in test_samples])
        keep_paths = [s["image_path"] for s in keep]
        train_vecs = vectorise(keep_paths)

        max_sim = np.zeros(len(keep), dtype=np.float32)
        for i in range(0, len(keep), 256):
            sims = train_vecs[i:i + 256] @ test_vecs.T
            max_sim[i:i + 256] = sims.max(axis=1)

        survivors = []
        for s, sim in zip(keep, max_sim):
            if sim >= args.near_threshold:
                dropped_near[CLASSES[s["label"]]] += 1
            else:
                survivors.append(s)
        keep = survivors

    n_near = sum(dropped_near.values())

    print(f"\n{'class':<14}{'kept':>8}{'exact-leak':>12}{'intra-dup':>11}{'near-dup':>10}")
    print("-" * 56)
    for cls in CLASSES:
        kept_c = sum(1 for s in keep if CLASSES[s["label"]] == cls)
        print(f"{cls:<14}{kept_c:>8}{dropped_leak.get(cls,0):>12}"
              f"{dropped_intra.get(cls,0):>11}{dropped_near.get(cls,0):>10}")
    print("-" * 56)
    print(f"{'TOTAL':<14}{len(keep):>8}{n_leak:>12}{n_intra:>11}{n_near:>10}")
    print(f"\ntraining set: {len(train_samples)} -> {len(keep)} "
          f"({len(train_samples) - len(keep)} removed)")
    print(f"test set    : {len(test_samples)} -> {len(test_samples)} (unchanged, now fully unseen)")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write the clean dataset.")
        return 0

    out_train = args.out_dir / "Training"
    out_test = args.out_dir / "Testing"
    if args.out_dir.exists():
        print(f"\nERROR: {args.out_dir} already exists — remove it first so nothing is "
              f"silently merged into a previous run.")
        return 2

    print(f"\nwriting {args.out_dir} ...")
    # Mirror the SOURCE folder names exactly. Deriving them from CLASSES would
    # produce "no_tumor" where the dataset actually ships "notumor".
    for s in keep:
        src = Path(s["image_path"])
        dst_dir = out_train / src.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
    for s in test_samples:
        src = Path(s["image_path"])
        dst_dir = out_test / src.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)

    manifest = {
        "source_train": str(args.train_dir),
        "source_test": str(args.test_dir),
        "train_before": len(train_samples),
        "train_after": len(keep),
        "removed_leaking_into_test": n_leak,
        "removed_intra_train_duplicates": n_intra,
        "removed_near_duplicates": n_near,
        "near_threshold": float(args.near_threshold),
        "per_class_removed_near": dict(dropped_near),
        "test_count": len(test_samples),
        "per_class_removed_leak": dict(dropped_leak),
        "per_class_removed_intra": dict(dropped_intra),
        "note": ("Test set deliberately kept whole so results stay comparable; "
                 "the duplicated copies were removed from TRAINING instead."),
    }
    (args.out_dir / "dedup_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                      encoding="utf-8")
    print(f"wrote {args.out_dir / 'dedup_manifest.json'}")
    print(f"\nRetrain with:\n  python train.py --data_dir {out_train} "
          f"--checkpoint_dir checkpoints/clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
