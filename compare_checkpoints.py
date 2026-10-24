#!/usr/bin/env python
"""
FreeBuff v4 — Checkpoint A/B comparison on the held-out set.

Evaluates two checkpoints under identical conditions (same data, same TTA
policy, same HOG branch state) and prints a side-by-side comparison, so a
newly trained model is adopted on evidence rather than on hope.

Each checkpoint is paired with its OWN PCA artifact. That pairing matters:
the HOG branch is only meaningful when the PCA that projects the descriptor
is the one the checkpoint was trained with, and a mismatched pair silently
degrades the branch instead of failing loudly.

Usage
-----
Compare a freshly trained run against the deployed model::

    python compare_checkpoints.py \
        --baseline saved_models/freebuff_v4.pt \
        --baseline-pca saved_models/freebuff_pca.pkl \
        --candidate checkpoints/v5_aug/freebuff_v4.pt \
        --candidate-pca checkpoints/v5_aug/freebuff_pca.pkl

Add ``--promote`` to copy the candidate over the deployed artifacts *only if*
it wins on macro F1. The previous artifacts are backed up first.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from config import CLASSES
from evaluate import run_evaluation

logger = logging.getLogger("freebuff_v4.compare")


def evaluate_one(
    checkpoint: Path,
    pca: Optional[Path],
    test_dir: Path,
    batch_size: int,
    use_tta: bool,
    device_str: Optional[str],
) -> Dict[str, Any]:
    """Run the standard held-out evaluation for a single checkpoint."""
    return run_evaluation(
        checkpoint_path=checkpoint,
        test_data_dir=test_dir,
        batch_size=batch_size,
        use_tta=use_tta,
        use_hog=pca is not None,
        pca_path=pca,
        device_str=device_str,
    )


def _fmt_delta(new: float, old: float) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two FreeBuff v4 checkpoints on the held-out set.",
    )
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Currently deployed checkpoint.")
    parser.add_argument("--baseline-pca", type=Path, default=None,
                        help="PCA artifact paired with the baseline checkpoint.")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="Newly trained checkpoint to evaluate.")
    parser.add_argument("--candidate-pca", type=Path, default=None,
                        help="PCA artifact paired with the candidate checkpoint.")
    parser.add_argument("--test_data_dir", type=Path, default=Path("data/Testing"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_tta", action="store_true",
                        help="Evaluate both without TTA.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--promote", action="store_true",
                        help="If the candidate wins on macro F1, copy it (and its "
                             "PCA) over the deployed artifacts, backing up first.")
    parser.add_argument("--promote-to", type=Path, default=Path("saved_models"),
                        help="Deployment directory for --promote.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for p in (args.baseline, args.candidate):
        if not p.exists():
            print(f"ERROR: checkpoint not found: {p}", file=sys.stderr)
            return 2
    if not args.test_data_dir.exists():
        print(f"ERROR: test data not found: {args.test_data_dir}", file=sys.stderr)
        return 2

    use_tta = not args.no_tta
    print("=" * 70)
    print("FreeBuff v4 — checkpoint comparison")
    print("=" * 70)
    print(f"test data : {args.test_data_dir}")
    print(f"TTA       : {'on' if use_tta else 'off'}")
    print(f"device    : {args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print()

    print(f"[1/2] evaluating BASELINE  : {args.baseline}")
    base = evaluate_one(args.baseline, args.baseline_pca, args.test_data_dir,
                        args.batch_size, use_tta, args.device)
    print(f"[2/2] evaluating CANDIDATE : {args.candidate}")
    cand = evaluate_one(args.candidate, args.candidate_pca, args.test_data_dir,
                        args.batch_size, use_tta, args.device)

    # ---- side-by-side --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'metric':<22}{'baseline':>14}{'candidate':>14}{'delta':>14}")
    print("-" * 70)
    for key, label in (("accuracy", "accuracy"),
                       ("macro_f1", "macro F1"),
                       ("weighted_f1", "weighted F1")):
        b, c = float(base[key]), float(cand[key])
        print(f"{label:<22}{b:>14.4f}{c:>14.4f}{_fmt_delta(c, b):>14}")
    bm, cm = int(base["n_misclassified"]), int(cand["n_misclassified"])
    print(f"{'misclassified':<22}{bm:>14d}{cm:>14d}{cm - bm:>+14d}")

    print("\nper-class F1")
    print("-" * 70)
    b_rep = base.get("classification_report_dict", {})
    c_rep = cand.get("classification_report_dict", {})
    for cls in CLASSES:
        b = float(b_rep.get(cls, {}).get("f1-score", float("nan")))
        c = float(c_rep.get(cls, {}).get("f1-score", float("nan")))
        print(f"{cls:<22}{b:>14.4f}{c:>14.4f}{_fmt_delta(c, b):>14}")
    print("=" * 70)

    # ---- verdict -------------------------------------------------------------
    delta_f1 = float(cand["macro_f1"]) - float(base["macro_f1"])
    candidate_wins = delta_f1 > 0
    if candidate_wins:
        print(f"\nVERDICT: candidate WINS on macro F1 ({_fmt_delta(float(cand['macro_f1']), float(base['macro_f1']))}).")
    elif delta_f1 == 0:
        print("\nVERDICT: tie on macro F1 — keep the baseline (no reason to churn).")
    else:
        print(f"\nVERDICT: baseline WINS — keep it. Candidate is {abs(delta_f1):.4f} macro F1 worse.")

    if args.promote:
        if not candidate_wins:
            print("--promote given, but the candidate did not win. Nothing changed.")
            return 1
        dest = args.promote_to
        dest.mkdir(parents=True, exist_ok=True)
        backup = dest / "previous"
        backup.mkdir(parents=True, exist_ok=True)

        ckpt_dest = dest / "freebuff_v4.pt"
        if ckpt_dest.exists():
            shutil.copy2(ckpt_dest, backup / "freebuff_v4.pt")
        shutil.copy2(args.candidate, ckpt_dest)
        print(f"promoted checkpoint -> {ckpt_dest} (previous backed up in {backup})")

        if args.candidate_pca and Path(args.candidate_pca).exists():
            pca_dest = dest / "freebuff_pca.pkl"
            if pca_dest.exists():
                shutil.copy2(pca_dest, backup / "freebuff_pca.pkl")
            shutil.copy2(args.candidate_pca, pca_dest)
            print(f"promoted PCA        -> {pca_dest}")

        manifest = Path(args.candidate).parent / "manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, dest / "manifest.json")
            print(f"promoted manifest   -> {dest / 'manifest.json'}")

        print("\nRe-run evaluate.py to refresh evaluation_results/holdout_report.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
