#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Training Pipeline (train.py)
# ==============================================================================
# Robust training loop for the RTX 3050 (4 GB):
#   * AdamW with decoupled backbone / head learning rates
#   * Focal Loss (gamma = 2.0) with inverse-frequency class weighting
#   * Cosine annealing LR schedule (ReduceLROnPlateau fallback)
#   * Automatic Mixed Precision (torch.amp autocast + GradScaler, fp16)
#   * Gradient clipping, per-epoch validation (loss / accuracy / weighted F1)
#   * Best-checkpoint persistence + early stopping + CSV history export
#
# Run:
#   python train.py --data_dir data --num_epochs 60
# ==============================================================================
from __future__ import annotations

import csv
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Modern torch.amp namespace (torch.cuda.amp is deprecated since torch 2.4 and
# this codebase runs torch 2.5.1). Falls back for very old torch versions.
try:  # torch >= 2.3
    from torch.amp import GradScaler, autocast

    def _autocast(enabled: bool):
        return autocast(device_type="cuda", enabled=enabled)

except ImportError:  # pragma: no cover — legacy torch
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

    def _autocast(enabled: bool):
        return autocast(enabled=enabled)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import f1_score

from config import (
    ADAMW_BETA1,
    ADAMW_BETA2,
    ADAMW_EPS,
    BACKBONE_LR,
    BATCH_SIZE,
    CLASSES,
    COSINE_ETA_MIN,
    COSINE_T_MAX,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LR,
    DEFAULT_MODEL_SAVE_PATH,
    EARLY_STOPPING_PATIENCE,
    EVALUATION_DIR,
    FOCAL_GAMMA,
    GRADIENT_CLIP_NORM,
    IMG_SIZE,
    INPUT_SIZE,
    LOG_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    WEIGHT_DECAY,
    set_all_seeds,
)
from dataset import (
    BrainTumorDataset,
    HOGExtractor,
    build_dataset_from_folder,
    fit_pca_on_training_hog,
    make_dataloaders,
    print_dataset_statistics,
    split_dataset_into_train_val_test,
)
from model import (
    FocalLoss,
    FreeBuffV4,
    build_freebuff_v4,
    build_parameter_groups,
    save_checkpoint,
)
from preprocessing import augment_mri_image

logger = logging.getLogger("freebuff_v4.train")


# ==============================================================================
# METERS & HISTORY
# ==============================================================================
class AccuracyMeter:
    """Running top-1 accuracy and mean loss over batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.count = 0

    def update(self, logits: Tensor, targets: Tensor, loss: float) -> None:
        preds = logits.argmax(dim=1)
        self.correct += int((preds == targets).sum().item())
        self.total += int(targets.size(0))
        self.loss_sum += float(loss)
        self.count += int(logits.size(0))

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / self.count if self.count else 0.0


class TrainHistory:
    """Per-epoch training / validation metrics with CSV export."""

    def __init__(self) -> None:
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []
        self.val_f1s: List[float] = []
        self.lrs: List[float] = []

    def record(self, train_loss: float, train_acc: float, val_loss: float,
               val_acc: float, lr: float, val_f1: Optional[float] = None) -> None:
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_f1s.append(val_acc if val_f1 is None else val_f1)
        self.lrs.append(lr)

    def best_val_idx(self) -> int:
        return int(np.argmax(self.val_accs)) if self.val_accs else -1

    def best_val_acc(self) -> float:
        return float(max(self.val_accs)) if self.val_accs else 0.0

    def save_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_loss", "train_acc",
                             "val_loss", "val_acc", "val_f1", "lr"])
            for i in range(len(self.train_losses)):
                writer.writerow([
                    i + 1,
                    f"{self.train_losses[i]:.6f}",
                    f"{self.train_accs[i]:.6f}",
                    f"{self.val_losses[i]:.6f}",
                    f"{self.val_accs[i]:.6f}",
                    f"{self.val_f1s[i]:.6f}",
                    f"{self.lrs[i]:.6e}",
                ])
        logger.info("Training history exported to %s", path)
        return path


# ==============================================================================
# ONE TRAINING EPOCH (AMP-enabled)
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    clip_norm: float = GRADIENT_CLIP_NORM,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch; returns (mean_loss, accuracy)."""
    model.train()
    meter = AccuracyMeter()
    start = time.time()
    amp_active = bool(use_amp and device.type == "cuda" and scaler is not None
                      and scaler.is_enabled())

    for batch_idx, batch in enumerate(loader):
        images = batch.get("image")
        labels = batch.get("label")
        if images is None or labels is None:
            raise KeyError("Training batch missing 'image'/'label' keys.")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        hog_feats = None
        if batch.get("hog_pca") is not None:
            hog_feats = batch["hog_pca"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if amp_active:
            with _autocast(True):
                out = model(images, hog_pca=hog_feats)
                logits = out["logits"] if isinstance(out, dict) else out
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()            # type: ignore[union-attr]
            scaler.unscale_(optimizer)               # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)                   # type: ignore[union-attr]
            scaler.update()                          # type: ignore[union-attr]
        else:
            out = model(images, hog_pca=hog_feats)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        meter.update(logits.detach(), labels, loss.item())

        if (batch_idx + 1) % 20 == 0 or batch_idx == len(loader) - 1:
            lr_now = optimizer.param_groups[0].get("lr", 0.0)
            logger.info(
                "Epoch %d batch %d/%d | loss=%.4f acc=%.4f lr=%.2e",
                epoch, batch_idx + 1, len(loader), loss.item(), meter.accuracy, lr_now,
            )

    logger.info(
        "Epoch %d train: loss=%.4f acc=%.4f (%.1fs)",
        epoch, meter.avg_loss, meter.accuracy, time.time() - start,
    )
    return meter.avg_loss, meter.accuracy


# ==============================================================================
# VALIDATION
# ==============================================================================
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate without gradients; returns (mean_loss, accuracy, weighted_f1)."""
    model.eval()
    meter = AccuracyMeter()
    targets_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch.get("image")
            labels = batch.get("label")
            if images is None or labels is None:
                raise KeyError("Validation batch missing 'image'/'label' keys.")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            hog_feats = None
            if batch.get("hog_pca") is not None:
                hog_feats = batch["hog_pca"].to(device, non_blocking=True).float()

            out = model(images, hog_pca=hog_feats)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, labels)
            meter.update(logits, labels, loss.item())
            targets_all.append(labels.cpu().numpy())
            preds_all.append(logits.argmax(dim=1).cpu().numpy())

    f1 = 0.0
    if targets_all:
        f1 = float(f1_score(np.concatenate(targets_all),
                            np.concatenate(preds_all),
                            average="weighted", zero_division=0))
    logger.info(
        "Validation: loss=%.4f acc=%.4f weighted_f1=%.4f",
        meter.avg_loss, meter.accuracy, f1,
    )
    return meter.avg_loss, meter.accuracy, f1


# ==============================================================================
# SCHEDULER FACTORY
# ==============================================================================
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = "cosine",
    t_max: int = COSINE_T_MAX,
    eta_min: float = COSINE_ETA_MIN,
    factor: float = 0.5,
    patience: int = 6,
    cooldown: int = 2,
) -> Any:
    """Cosine annealing (default) or ReduceLROnPlateau (fallback)."""
    if mode == "cosine":
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if mode == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor,
                                 patience=patience, cooldown=cooldown)
    raise ValueError(f"Unknown scheduler mode: {mode}")


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def run_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_path: Optional[Union[str, Path]] = None,
    use_amp: bool = True,
    start_epoch: int = 0,
    history: Optional[TrainHistory] = None,
    model_save_path: Optional[Union[str, Path]] = None,
) -> TrainHistory:
    """Full loop: train -> validate -> schedule -> checkpoint -> early-stop."""
    history = history if history is not None else TrainHistory()

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    model_save_path = Path(model_save_path) if model_save_path else DEFAULT_MODEL_SAVE_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    best_val_f1 = 0.0
    best_epoch = start_epoch
    stale_epochs = 0

    logger.info(
        "Training start: epochs=%d device=%s amp=%s patience=%d",
        num_epochs, device, use_amp and device.type == "cuda", patience,
    )
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB VRAM)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        logger.info("=" * 64)
        logger.info("EPOCH %d / %d", epoch, num_epochs)
        logger.info("=" * 64)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, epoch, GRADIENT_CLIP_NORM, use_amp,
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0].get("lr", 0.0)
        history.record(train_loss, train_acc, val_loss, val_acc, current_lr, val_f1)

        logger.info(
            "Epoch %d summary: train_loss=%.4f train_acc=%.4f val_loss=%.4f "
            "val_acc=%.4f val_f1=%.4f best_f1=%.4f",
            epoch, train_loss, train_acc, val_loss, val_acc, val_f1, best_val_f1,
        )

        # ---- scheduler step (API differs between variants) -----------------
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_f1)
        elif scheduler is not None:
            scheduler.step()

        # ---- best-checkpoint persistence ------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            logger.info("New best val weighted F1 %.4f at epoch %d — saving.",
                        val_f1, epoch)
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            # Keep the deployment copy in saved_models/ in sync with the best
            # checkpoint. This line was previously commented out ("same file"
            # fix), which left a stale pre-training artifact behind after
            # retraining — the dashboard then served epoch-1 weights while the
            # reported metrics came from the fresh run.
            if Path(model_save_path).resolve() != Path(checkpoint_path).resolve():
                shutil.copy2(checkpoint_path, model_save_path)
                logger.info("Deployment copy updated: %s", model_save_path)
        else:
            stale_epochs += 1
            logger.info("No improvement for %d/%d epochs.", stale_epochs, patience)

        # ---- early stopping --------------------------------------------------
        if stale_epochs >= patience:
            logger.warning("Early stopping after %d stale epochs.", stale_epochs)
            break

    logger.info(
        "Training complete: best_val_f1=%.4f at epoch %d (%d epochs run).",
        best_val_f1, best_epoch, len(history.train_losses),
    )
    return history


# ==============================================================================
# DEPLOYMENT MANIFEST
# ==============================================================================
def write_deployment_manifest(
    path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    pca_path: Optional[Union[str, Path]],
    model: nn.Module,
    history_result: Dict[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
    num_epochs_requested: int,
    seed: int,
    gamma: float,
    batch_size: int,
    pca_components: Optional[int],
) -> Path:
    """Describe the artifacts this run produced, next to the checkpoint.

    The manifest is what a deployment (or an examiner) reads to know which
    architecture and which metrics the shipped weights correspond to, so every
    field is taken from the live model and this run's results rather than
    hardcoded.
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    checkpoint_path = Path(checkpoint_path)
    path = Path(path)

    def _sha256(p: Path) -> Optional[str]:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    manifest = {
        "artifact_version": 4,
        "generated_by": "train.py",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "architecture": {
            "name": "FreeBuffV4",
            "deep_branch": "ResNet50 + CBAM (channel + spatial attention)",
            "handcrafted_branch": "HOG -> PCA -> MLP",
            "fusion_head": "768 -> 512 -> 256 -> 4",
            "total_params": int(n_params),
            "trainable_params": int(n_trainable),
        },
        "input": {
            "img_size": IMG_SIZE,
            "input_size": list(INPUT_SIZE),
            "normalize_mean": list(NORMALIZE_MEAN),
            "normalize_std": list(NORMALIZE_STD),
        },
        "class_names": list(CLASSES),
        "hog_pca_components": pca_components,
        "training": {
            "epochs_requested": int(num_epochs_requested),
            "epochs_completed": int(history_result.get("num_epochs_completed", 0)),
            "best_epoch": int(history_result.get("best_epoch", 0)),
            "batch_size": int(batch_size),
            "focal_gamma": float(gamma),
            "seed": int(seed),
            "device": str(history_result.get("device", "")),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_internal_test": int(n_test),
        },
        "metrics": {
            "best_val_acc": history_result.get("best_val_acc"),
            "internal_test_acc": history_result.get("test_acc"),
            "internal_test_f1": history_result.get("test_f1"),
            "note": (
                "These are the internal splits of the training folder. The "
                "held-out data/Testing figures are produced by evaluate.py."
            ),
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "pca": Path(pca_path).name if pca_path else None,
            "pca_sha256": _sha256(Path(pca_path)) if pca_path else None,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Deployment manifest written to %s", path)
    return path


# ==============================================================================
# END-TO-END RUNNER
# ==============================================================================
def run(
    data_dir: Union[str, Path],
    checkpoint_dir: Optional[Union[str, Path]] = None,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    backbone_lr: float = BACKBONE_LR,
    gamma: float = FOCAL_GAMMA,
    weight_decay: float = WEIGHT_DECAY,
    use_amp: bool = True,
    adam_eps: float = ADAMW_EPS,
    patience: int = EARLY_STOPPING_PATIENCE,
    scheduler_mode: str = "cosine",
    seed: int = 42,
    freeze_backbone: bool = False,
    freeze_bn: bool = False,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    use_pca_hog: bool = True,
    pca_components: int = 288,
    use_separate_head_lr: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """End-to-end: dataset -> PCA fit -> model -> optimizer -> train -> test."""
    set_all_seeds(seed)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset_from_folder(Path(data_dir))
    train_ds, val_ds, test_ds = split_dataset_into_train_val_test(
        dataset, val_frac=val_frac, test_frac=test_frac, seed=seed, stratify=True,
    )
    print_dataset_statistics({"train": train_ds, "val": val_ds, "test": test_ds})

    # ---- HOG PCA fitting (train split only — no leakage) --------------------
    hog_extractor: Optional[HOGExtractor] = None
    pca_model = None
    pca_out = None
    if use_pca_hog:
        hog_extractor = HOGExtractor()
        logger.info("Fitting HOG-PCA (%d components) on the training split…",
                    pca_components)
        pca_model = fit_pca_on_training_hog(
            train_ds, pca_components=pca_components, hog_extractor=hog_extractor,
        )
        import joblib
        from config import DEFAULT_PCA_PATH

        # The PCA must stay paired with the checkpoint that was trained on it:
        # a mismatched pair silently degrades the HOG branch. So when the run
        # is directed at its own checkpoint_dir, the PCA goes there too,
        # instead of overwriting the deployed saved_models artifact.
        pca_out = (Path(checkpoint_dir) / DEFAULT_PCA_PATH.name
                   if checkpoint_dir else DEFAULT_PCA_PATH)
        pca_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca_model, pca_out)
        logger.info("PCA transformer saved to %s", pca_out)

    # ---- rebuild datasets with the fitted extractor / PCA -------------------
    def _materialize(subset: torch.utils.data.Subset, augment: bool) -> BrainTumorDataset:
        samples = [subset.dataset.samples_list[i] for i in subset.indices]
        return BrainTumorDataset(
            samples=samples,
            hog_extractor=hog_extractor,
            pca=pca_model,
            use_hog=use_pca_hog,
            augment=augment,
            augment_fn=augment_mri_image if augment else None,
        )

    train_dataset = _materialize(train_ds, augment=True)
    val_dataset = _materialize(val_ds, augment=False)
    test_dataset = _materialize(test_ds, augment=False)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size, num_workers=NUM_WORKERS,
    )

    # ---- model ---------------------------------------------------------------
    model = build_freebuff_v4(
        num_classes=NUM_CLASSES, device=device, pretrained=True,
        freeze_backbone=freeze_backbone, freeze_bn=freeze_bn,
    )

    # ---- Focal Loss with inverse-frequency alpha ------------------------------
    train_labels = np.asarray(
        [s["label"] for s in train_dataset.samples_list], dtype=np.int64
    )
    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    inv_freq = 1.0 / np.maximum(counts, 1.0)
    class_weights = (inv_freq / inv_freq.mean()).astype(np.float32)
    logger.info("Class counts: %s", {CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)})
    logger.info("Focal alpha:   %s", {CLASSES[i]: round(float(class_weights[i]), 3)
                                      for i in range(NUM_CLASSES)})
    criterion = FocalLoss(
        gamma=gamma,
        alpha=torch.tensor(class_weights, device=device),
        reduction="mean",
    )

    # ---- optimizer (decoupled LRs) --------------------------------------------
    if use_separate_head_lr:
        param_groups = build_parameter_groups(
            model, backbone_lr=backbone_lr, head_lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer = AdamW(param_groups, eps=adam_eps, betas=(ADAMW_BETA1, ADAMW_BETA2))
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=weight_decay, eps=adam_eps,
                          betas=(ADAMW_BETA1, ADAMW_BETA2))

    scheduler = build_scheduler(optimizer, mode=scheduler_mode, t_max=num_epochs,
                                eta_min=COSINE_ETA_MIN)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else checkpoint_dir
    checkpoint_path = (ckpt_dir / "freebuff_v4.pt") if ckpt_dir else DEFAULT_CHECKPOINT_PATH
    model_save_path = (ckpt_dir / "freebuff_v4.pt") if ckpt_dir else DEFAULT_MODEL_SAVE_PATH

    history = run_training(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, num_epochs=num_epochs, patience=patience,
        checkpoint_path=checkpoint_path, use_amp=use_amp,
        model_save_path=model_save_path,
    )

    history.save_csv(EVALUATION_DIR / "training_history.csv")

    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    result: Dict[str, Any] = {
        "best_val_acc": history.best_val_acc(),
        "best_epoch": history.best_val_idx() + 1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "num_epochs_completed": len(history.train_losses),
        "device": str(device),
    }
    logger.info("Training run summary: %s", result)

    # ---- deployment manifest ------------------------------------------------
    # Written next to the checkpoint it describes, from the values this run
    # actually produced. Without this the manifest goes stale and ends up
    # describing a different architecture than the one that ships.
    try:
        write_deployment_manifest(
            path=model_save_path.parent / "manifest.json",
            checkpoint_path=model_save_path,
            pca_path=pca_out if use_pca_hog else None,
            model=model,
            history_result=result,
            n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
            num_epochs_requested=num_epochs,
            seed=seed, gamma=gamma, batch_size=batch_size,
            pca_components=pca_components if use_pca_hog else None,
        )
    except Exception as exc:  # never fail a finished run over metadata
        logger.warning("Could not write deployment manifest: %s", exc)

    return result


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="FreeBuff v4 Training")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Dataset root containing one subfolder per class.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone_lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--no_hog", action="store_true",
                        help="Disable the HOG-PCA branch.")
    parser.add_argument("--pca_components", type=int, default=288)

    args = parser.parse_args()

    summary = run(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        patience=args.patience,
        scheduler_mode=args.scheduler,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
        freeze_bn=args.freeze_bn,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        use_pca_hog=not args.no_hog,
        pca_components=args.pca_components,
    )

    print("\n===== TRAINING SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("============================\n")
