#!/usr/bin/env python
"""
FreeBuff v4 — train/test duplicate-leakage audit.

The public brain-MRI dataset ships as loose slices with no patient identifiers,
so a patient-level train/test split is impossible: slices from one patient can
land on both sides. We cannot fix the split without those identifiers. What we
can do is measure the exposure — and on this dataset the exposure is real.

Measured result (data/Training vs data/Testing):

    114 of 1,600 test images (7.12%) are BYTE-IDENTICAL to a training image.
    100 of those are meningioma, i.e. 25% of that class's test set.
    data/Training additionally contains 190 duplicate images of its own.

Those images test memorisation, not generalisation, so the honest held-out
figure is the one computed after removing them. Run with --evaluate to get it:

    full test set   0.9556 accuracy / 0.9552 macro F1  (1,600 images)
    leak-free       0.9542 accuracy / 0.9518 macro F1  (1,486 images)

The gap is 0.14 accuracy points, so the headline result survives the audit —
but that is a measured conclusion now, not an assumption. Meningioma precision
is the metric that moves most (0.9040 -> 0.8758), which is exactly the class
carrying the duplicates.

Detection uses an exact hash of the decoded pixel array, so a match means the
two files decode to the same image — not merely a similar-looking one. This is
deliberately stricter than perceptual hashing, which on brain MRI flags far
more pairs than are genuinely duplicates (every scan is a bright ellipse on a
black field, so coarse hashes collide constantly).

Usage
-----
    python audit_leakage.py
    python audit_leakage.py --evaluate          # also score leak-free vs full
    python audit_leakage.py --list 20           # show example duplicate pairs
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from config import CLASSES, DEFAULT_MODEL_SAVE_PATH
from dataset import build_dataset_from_folder

logger = logging.getLogger("freebuff_v4.audit")


def pixel_hash(path: str | Path) -> str:
    """SHA-1 of the decoded greyscale pixel array (plus its shape).

    Hashing decoded pixels rather than file bytes means two files that differ
    only in JPEG metadata still register as the same image.
    """
    arr = np.asarray(Image.open(path).convert("L"))
    return hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def index_by_hash(samples: List[Dict]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        index[pixel_hash(s["image_path"])].append(s["image_path"])
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit train/test duplicate leakage.")
    ap.add_argument("--train_dir", type=Path, default=Path("data/Training"))
    ap.add_argument("--test_dir", type=Path, default=Path("data/Testing"))
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    ap.add_argument("--pca_path", type=Path, default=Path("saved_models/freebuff_pca.pkl"))
    ap.add_argument("--list", type=int, default=0, help="Print this many example duplicate pairs.")
    ap.add_argument("--near_threshold", type=float, default=0.999,
                    help="Also treat a test image as contaminated when its pixel "
                         "correlation to any training image reaches this. 0 disables. "
                         "An exact hash misses a scan that was merely re-encoded or "
                         "shifted a pixel, and those are still the same study.")
    ap.add_argument("--evaluate", action="store_true",
                    help="Also score the model on the full and leak-free test sets.")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    train_samples = build_dataset_from_folder(args.train_dir).samples_list
    test_samples = build_dataset_from_folder(args.test_dir).samples_list

    print("=" * 72)
    print("FreeBuff v4 — train/test duplicate-leakage audit")
    print("=" * 72)
    print(f"train: {len(train_samples)} images from {args.train_dir}")
    print(f"test : {len(test_samples)} images from {args.test_dir}\n")

    train_index = index_by_hash(train_samples)

    leaked, clean = [], []
    per_class = defaultdict(int)
    examples: List[tuple] = []
    for s in test_samples:
        h = pixel_hash(s["image_path"])
        if h in train_index:
            leaked.append(s)
            per_class[CLASSES[s["label"]]] += 1
            if len(examples) < max(args.list, 0):
                examples.append((s["image_path"], train_index[h][0]))
        else:
            clean.append(s)

    pct = 100.0 * len(leaked) / max(len(test_samples), 1)
    print(f"EXACT duplicates (test image also present in train): {len(leaked)} / "
          f"{len(test_samples)}  ({pct:.2f}%)")
    for cls in CLASSES:
        n = per_class.get(cls, 0)
        total = sum(1 for s in test_samples if CLASSES[s["label"]] == cls)
        bar = "#" * int(30 * n / max(total, 1))
        print(f"   {cls:<12} {n:4d} / {total:<4d} ({100*n/max(total,1):5.1f}%)  {bar}")

    intra = sum(len(v) - 1 for v in train_index.values() if len(v) > 1)
    print(f"\nduplicate images WITHIN {args.train_dir}: {intra}")

    # ---- near-duplicate pass -------------------------------------------------
    # Exact hashing misses a re-encoded or 1-pixel-shifted copy of the same
    # scan. Pixel correlation on a 64x64 downsample separates those cleanly:
    # genuinely different scans sit around 0.91, re-encoded copies above 0.999.
    near_flagged = 0
    if args.near_threshold and args.near_threshold > 0:
        def vectorise(items):
            out = np.zeros((len(items), 64 * 64), dtype=np.float32)
            for i, s in enumerate(items):
                a = np.asarray(Image.open(s["image_path"]).convert("L").resize((64, 64)),
                               dtype=np.float32).ravel()
                a -= a.mean()
                nrm = float(np.linalg.norm(a))
                out[i] = a / nrm if nrm > 0 else a
            return out

        tr_v, te_v = vectorise(train_samples), vectorise(test_samples)
        max_sim = np.zeros(len(test_samples), dtype=np.float32)
        for i in range(0, len(test_samples), 256):
            max_sim[i:i + 256] = (te_v[i:i + 256] @ tr_v.T).max(axis=1)

        leaked_paths = {s["image_path"] for s in leaked}
        still_clean, near_per_class = [], defaultdict(int)
        for s, sim in zip(test_samples, max_sim):
            if s["image_path"] in leaked_paths:
                continue
            if sim >= args.near_threshold:
                near_flagged += 1
                near_per_class[CLASSES[s["label"]]] += 1
                leaked.append(s)
            else:
                still_clean.append(s)
        clean = still_clean

        print(f"\nNEAR duplicates (correlation >= {args.near_threshold}, beyond the exact ones): "
              f"{near_flagged}")
        for cls in CLASSES:
            print(f"   {cls:<12} {near_per_class.get(cls, 0):>4}")

    total_contaminated = len(leaked)
    print(f"\nTOTAL contaminated test images: {total_contaminated} / {len(test_samples)} "
          f"({100 * total_contaminated / max(len(test_samples), 1):.1f}%)")
    print(f"UNCONTAMINATED evaluation set: {len(clean)} images")

    if examples:
        print(f"\nexample duplicate pairs:")
        for t, tr in examples:
            print(f"   {Path(t).name:<20} == {Path(tr).name}")

    if args.evaluate:
        import joblib, torch
        from torch.utils.data import DataLoader
        from dataset import BrainTumorDataset, HOGExtractor
        from evaluate import evaluate_model
        from model import build_freebuff_v4, load_checkpoint

        device = torch.device(args.device) if args.device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_freebuff_v4(pretrained=False).to(device)
        load_checkpoint(args.checkpoint, model, device=device, strict=False, verbose=False)
        pca = joblib.load(args.pca_path) if Path(args.pca_path).exists() else None

        def collate(b):
            return {"image": torch.stack([x["image"] for x in b]),
                    "label": torch.stack([x["label"] for x in b]),
                    "hog_pca": (torch.stack([x["hog_pca"] for x in b])
                                if all(x["hog_pca"] is not None for x in b) else None),
                    "path": [x["path"] for x in b],
                    "metadata": [x["metadata"] for x in b]}

        def score(samples, tag):
            ds = BrainTumorDataset(samples=samples,
                                   hog_extractor=HOGExtractor() if pca is not None else None,
                                   pca=pca, use_hog=pca is not None)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
            r = evaluate_model(model=model, test_loader=dl, device=device, class_names=CLASSES,
                               use_tta=True, hog_pca_for_test=pca, use_hog=pca is not None,
                               save_gradcam_for_misclassifications=False)
            print(f"\n{tag}  ({len(samples)} images)")
            print(f"   accuracy {r['accuracy']:.4f}   macro F1 {r['macro_f1']:.4f}   "
                  f"misclassified {r['n_misclassified']}")
            rep = r.get("classification_report_dict", {})
            for c in CLASSES:
                d = rep.get(c)
                if d:
                    print(f"     {c:<12} P {d['precision']:.4f}  R {d['recall']:.4f}  "
                          f"F1 {d['f1-score']:.4f}  n={int(d['support'])}")
            return r

        print("\n" + "=" * 72)
        full = score(test_samples, "FULL test set (as normally reported)")
        lf = score(clean, "UNCONTAMINATED subset (exact + near duplicates removed)")
        print("\n" + "=" * 72)
        print(f"reported accuracy  : {full['accuracy']:.4f}   macro F1 {full['macro_f1']:.4f}")
        print(f"uncontaminated acc : {lf['accuracy']:.4f}   macro F1 {lf['macro_f1']:.4f}")
        print(f"difference         : {lf['accuracy']-full['accuracy']:+.4f} accuracy, "
              f"{lf['macro_f1']-full['macro_f1']:+.4f} macro F1")
        print("=" * 72)
        print("The UNCONTAMINATED figure is the one to quote as a generalisation estimate.")
    else:
        print("\nRe-run with --evaluate to score the model on the leak-free subset.")

    print("\nNote: this measures duplicate FILES. Without patient identifiers it")
    print("cannot detect different slices from the same patient, so it is a lower")
    print("bound on the true overlap, not a clean bill of health.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
