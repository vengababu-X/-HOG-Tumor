#!/usr/bin/env python
"""
FreeBuff v4 — confidence calibration (temperature scaling).

A network's softmax score is not a probability: modern classifiers are
typically over-confident, so a reported "97%" does not mean the model is right
97% of the time. Temperature scaling divides the logits by a single scalar T
fitted by maximum likelihood on a held-out set. It is the standard fix
(Guo et al., 2017) and it has one property that matters clinically:

    dividing every logit by the same positive T cannot change which class is
    largest, so accuracy, F1 and the confusion matrix are provably unchanged.

Only the confidence numbers move — which is exactly what the reliability bands
and the low-confidence safeguard consume.

Measured finding on this project
--------------------------------
Where the calibration set comes from decides whether this works at all:

  * fitted on a held-out split of data/Training  -> T = 0.65, and the test-set
    ECE got WORSE (0.0214 -> 0.0369). The model is near-perfect on the training
    distribution, so the fit sharpens confidence further; that does not
    transfer across the domain gap to data/Testing.

  * fitted by 2-fold cross-validation inside data/Testing -> T ~= 1.24, and the
    held-out ECE improved (0.0238 -> 0.0165, a 31% relative reduction) with
    accuracy unchanged to the digit.

So the calibration set must come from the *deployment* distribution. That is
why no temperature is hardcoded in config.py: a site fits its own on its own
scans with this script.

Usage
-----
Fit on a calibration folder and report the effect on a separate evaluation
folder::

    python calibrate.py --calib_data_dir data/Calibration --eval_data_dir data/Testing

Fit and persist, so the inference engine picks it up automatically::

    python calibrate.py --calib_data_dir data/Calibration --apply

`--apply` writes saved_models/calibration.json. FreeBuffInference.from_dir
reads that file when present and uses the temperature for every prediction.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CLASSES, DEFAULT_MODEL_SAVE_PATH, SAVED_MODELS_DIR
from dataset import BrainTumorDataset, HOGExtractor, build_dataset_from_folder
from inference import fit_temperature
from model import build_freebuff_v4, load_checkpoint

logger = logging.getLogger("freebuff_v4.calibrate")

CALIBRATION_FILENAME = "calibration.json"


# ---------------------------------------------------------------- collation
def _collate(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        "hog_pca": (torch.stack([x["hog_pca"] for x in batch])
                    if all(x["hog_pca"] is not None for x in batch) else None),
    }


def collect_logits(
    model: torch.nn.Module,
    data_dir: Path,
    pca,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a class-folder directory and return (logits, labels)."""
    samples = build_dataset_from_folder(data_dir).samples_list
    ds = BrainTumorDataset(samples=samples, hog_extractor=HOGExtractor() if pca is not None else None,
                           pca=pca, use_hog=pca is not None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=_collate)
    logits_all, labels_all = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            hog = batch["hog_pca"].to(device).float() if batch["hog_pca"] is not None else None
            out = model(images, hog_pca=hog)
            logits_all.append(out["logits"].float().cpu())
            labels_all.append(batch["label"].cpu())
    return torch.cat(logits_all), torch.cat(labels_all).long()


# ---------------------------------------------------------------- metrics
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                               n_bins: int = 15) -> float:
    """Expected Calibration Error: mean |accuracy - confidence| across bins.

    Predictions are bucketed by their confidence. Within each bucket we compare
    how often the model was right against how confident it claimed to be; ECE
    is the sample-weighted mean of those gaps. 0 is perfect calibration.
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if in_bin.sum() > 0:
            weight = in_bin.float().mean()
            gap = (correct[in_bin].mean() - confidences[in_bin].mean()).abs()
            ece += float(weight * gap)
    return ece


def summarise(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    probs = torch.softmax(logits / temperature, dim=1)
    onehot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    return {
        "temperature": float(temperature),
        "ece": expected_calibration_error(probs, labels),
        "nll": float(torch.nn.functional.cross_entropy(logits / temperature, labels)),
        "brier": float(((probs - onehot) ** 2).sum(dim=1).mean()),
        "accuracy": float(probs.argmax(1).eq(labels).float().mean()),
        "mean_confidence": float(probs.max(1).values.mean()),
    }


def _print_row(name: str, before: dict, after: dict, key: str, lower_is_better: Optional[bool]) -> None:
    b, a = before[key], after[key]
    delta = a - b
    if lower_is_better is True:
        verdict = "better" if delta < 0 else ("worse" if delta > 0 else "same")
    elif lower_is_better is False:
        verdict = "unchanged" if abs(delta) < 1e-9 else "CHANGED (unexpected)"
    else:
        verdict = ""
    print(f"  {name:<20}{b:>12.4f}{a:>12.4f}{delta:>+12.4f}   {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a softmax temperature for FreeBuff v4.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--pca_path", type=Path, default=SAVED_MODELS_DIR / "freebuff_pca.pkl")
    parser.add_argument("--calib_data_dir", type=Path, required=True,
                        help="Class-folder directory used to FIT the temperature. "
                             "It must come from the deployment distribution.")
    parser.add_argument("--eval_data_dir", type=Path, default=None,
                        help="Optional separate directory for reporting the effect. "
                             "Never fit and report on the same images.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--apply", action="store_true",
                        help="Persist the fitted temperature to "
                             f"saved_models/{CALIBRATION_FILENAME} so the engine uses it.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 2
    if not args.calib_data_dir.exists():
        print(f"ERROR: calibration data not found: {args.calib_data_dir}")
        return 2

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_freebuff_v4(pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, device=device, strict=False, verbose=False)

    pca = None
    if args.pca_path and Path(args.pca_path).exists():
        import joblib
        pca = joblib.load(args.pca_path)

    print("=" * 70)
    print("FreeBuff v4 — temperature calibration")
    print("=" * 70)
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}")
    print(f"HOG branch : {'on' if pca is not None else 'off'}\n")

    calib_logits, calib_labels = collect_logits(model, args.calib_data_dir, pca, device, args.batch_size)
    print(f"calibration set: {len(calib_labels)} images from {args.calib_data_dir}")

    temperature = fit_temperature(calib_logits, calib_labels)
    print(f"fitted temperature: T = {temperature:.4f}")
    if temperature > 1.0:
        print("  T > 1 softens the distribution — the model was over-confident.")
    elif temperature < 1.0:
        print("  T < 1 sharpens the distribution — the model was under-confident.")
    print()

    eval_dir = args.eval_data_dir
    if eval_dir is not None:
        if not eval_dir.exists():
            print(f"ERROR: evaluation data not found: {eval_dir}")
            return 2
        eval_logits, eval_labels = collect_logits(model, eval_dir, pca, device, args.batch_size)
        print(f"evaluation set : {len(eval_labels)} images from {eval_dir}\n")
        before = summarise(eval_logits, eval_labels, 1.0)
        after = summarise(eval_logits, eval_labels, temperature)
        print(f"  {'metric':<20}{'T = 1.0':>12}{'T = ' + format(temperature, '.3f'):>12}{'change':>12}")
        print("  " + "-" * 62)
        _print_row("ECE", before, after, "ece", True)
        _print_row("NLL", before, after, "nll", True)
        _print_row("Brier", before, after, "brier", True)
        _print_row("accuracy", before, after, "accuracy", False)
        _print_row("mean confidence", before, after, "mean_confidence", None)
        print()
        if after["ece"] < before["ece"]:
            rel = 100.0 * (before["ece"] - after["ece"]) / max(before["ece"], 1e-12)
            print(f"  Calibration IMPROVED — ECE down {rel:.1f}% relative, accuracy unchanged.")
        else:
            print("  Calibration did NOT improve on this evaluation set.")
            print("  That usually means the calibration set came from a different")
            print("  distribution than the evaluation set. Fit on deployment data.")
        print()

    if args.apply:
        out = Path(args.checkpoint).parent / CALIBRATION_FILENAME
        payload = {
            "temperature": float(temperature),
            "fitted_on": str(args.calib_data_dir),
            "n_calibration_images": int(len(calib_labels)),
            "checkpoint": Path(args.checkpoint).name,
            "classes": list(CLASSES),
            "note": ("Temperature scaling. Divides logits by T before softmax. "
                     "Cannot change the predicted class, only its confidence."),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
        print("The inference engine will load this automatically on next start.")
    else:
        print("Not persisted. Re-run with --apply to write "
              f"saved_models/{CALIBRATION_FILENAME}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
