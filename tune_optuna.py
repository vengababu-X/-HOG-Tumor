#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Optuna Hyperparameter Tuning CLI
# ==============================================================================
# Standalone entrypoint for Bayesian hyperparameter optimization of the
# FreeBuff v4 training pipeline using Optuna. It runs short training trials
# (or full training, depending on flags), records validation weighted F1 as
# the objective, and persists the best configuration to disk.
#
# Usage:
#   python tune_optuna.py --data_dir data/ --n_trials 20
#   python tune_optuna.py --data_dir data/ --n_trials 50 --study_dir studies
# ==============================================================================
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("freebuff_v4.tune_optuna")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_trial_param_space(trial: Any) -> Dict[str, Any]:
    """Suggest hyperparameters for one Optuna trial."""
    backbone_lr = trial.suggest_float("backbone_lr", 1e-5, 1e-3, log=True)
    head_lr = trial.suggest_float("head_lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 3.0, step=0.25)
    dropout_fusion = trial.suggest_float("dropout_fusion", 0.1, 0.5, step=0.05)
    scheduler_mode = trial.suggest_categorical(
        "scheduler_mode",
        ["cosine", "plateau"],
    )
    fusion_hidden_dims_key = trial.suggest_categorical(
        "fusion_hidden_dims",
        ["d1", "d2", "d2_d1", "d2_d1_d0"],
    )
    fusion_hidden_dims_map = {
        "d1": [256],
        "d2": [512],
        "d2_d1": [512, 256],
        "d2_d1_d0": [512, 256, 128],
    }
    fusion_hidden_dims = fusion_hidden_dims_map[fusion_hidden_dims_key]
    return {
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "dropout_fusion": dropout_fusion,
        "scheduler_mode": scheduler_mode,
        "fusion_hidden_dims": fusion_hidden_dims,
    }


def run_trial_training(
    data_dir: Path,
    params: Dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    pca_components: int,
    seed: int,
    device: Any,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a single training trial with the given hyperparameters.

    For speed, this wraps the existing training entrypoint in train.py when
    available; otherwise it performs a minimal self-contained training flow.
    """
    from config import (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        NUM_EPOCHS,
        NUM_WORKERS,
        DEFAULT_LR,
        COSINE_T_MAX,
        COSINE_ETA_MIN,
        CLASSES,
        NUM_CLASSES,
        initialize_directories,
        set_all_seeds,
    )
    from dataset import (
        BrainTumorDataset,
        HOGExtractor,
        make_dataloaders,
        split_dataset_into_train_val_test,
        fit_pca_on_training_hog,
        build_dataset_from_folder,
    )
    from model import (
        build_freebuff_v4,
        build_parameter_groups,
        FocalLoss,
    )
    from preprocessing import augment_mri_image

    set_all_seeds(seed)
    initialize_directories()

    full_dataset = build_dataset_from_folder(data_dir)

    train_ds, val_ds, _ = split_dataset_into_train_val_test(
        full_dataset,
        val_frac=0.2,
        seed=seed,
        stratify=True,
    )

    import torch

    train_samples = [train_ds.dataset.samples_list[index] for index in train_ds.indices]
    val_samples = [val_ds.dataset.samples_list[index] for index in val_ds.indices]

    hog_extractor = HOGExtractor()
    pca_model = fit_pca_on_training_hog(
        train_ds,
        pca_components=pca_components,
        hog_extractor=hog_extractor,
    )

    train_dataset = BrainTumorDataset(
        samples=train_samples,
        hog_extractor=hog_extractor,
        pca=pca_model,
        use_hog=True,
        augment=True,
        augment_fn=augment_mri_image,
    )
    val_dataset = BrainTumorDataset(
        samples=val_samples,
        hog_extractor=hog_extractor,
        pca=pca_model,
        use_hog=True,
    )

    train_subset = torch.utils.data.Subset(train_dataset, list(range(len(train_dataset))))
    val_subset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset))))

    train_loader, val_loader, _test_loader = make_dataloaders(
        train_subset,
        val_subset,
        val_subset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    model = build_freebuff_v4(
        num_classes=NUM_CLASSES,
        device=device,
        pretrained=True,
        freeze_backbone=False,
        freeze_bn=False,
        fusion_hidden_dims=params["fusion_hidden_dims"],
    )

    train_labels = np.asarray(
        [sample["label"] for sample in train_samples], dtype=np.int64
    )
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    inverse_frequency = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = inverse_frequency / inverse_frequency.mean()
    criterion = FocalLoss(
        gamma=params["gamma"],
        alpha=torch.tensor(class_weights, dtype=torch.float32, device=device),
    )

    param_groups = build_parameter_groups(
        model,
        backbone_lr=params["backbone_lr"],
        head_lr=params["head_lr"],
        weight_decay=params["weight_decay"],
    )
    optimizer = _build_optimizer(param_groups, params)

    scheduler = _build_scheduler(optimizer, params["scheduler_mode"], epochs)

    # Minimal training loop with early stopping tied to trial patience
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: Dict[str, list] = {"val_f1": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=True,
            dropout_fusion=params["dropout_fusion"],
        )
        val_loss, val_acc, val_f1 = _validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        history["val_f1"].append(val_f1)

        logger.info(
            "Trial epoch %d/%d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, "
            "val_acc=%.4f, val_f1=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )

        if hasattr(scheduler, "step"):
            scheduler_name = getattr(scheduler, "__class__", type(scheduler)).__name__
            if scheduler_name == "CosineAnnealingLR":
                scheduler.step()
            elif scheduler_name == "ReduceLROnPlateau":
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping trial at epoch %d (best val_f1=%.4f)",
                epoch,
                best_val_f1,
            )
            break

    return {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "train_loss": float(history["val_f1"][-1]) if history["val_f1"] else 0.0,
    }


def _build_optimizer(param_groups, params: Dict[str, Any]) -> Any:
    from torch.optim import AdamW
    from config import ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPS

    optimizer = AdamW(
        param_groups,
        lr=params["head_lr"],
        weight_decay=params["weight_decay"],
        eps=ADAMW_EPS,
        betas=(ADAMW_BETA1, ADAMW_BETA2),
    )
    return optimizer


def _build_scheduler(optimizer: Any, mode: str, t_max: int) -> Any:
    if mode == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from config import COSINE_ETA_MIN, COSINE_T_MAX
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=COSINE_ETA_MIN)
    if mode == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=6,
            cooldown=2,
            verbose=False,
        )
    raise ValueError(f"Unsupported scheduler mode: {mode}")


def _train_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    epoch: int,
    *,
    use_amp: bool,
    dropout_fusion: float,
) -> Tuple[float, float]:
    import torch
    from torch.amp import GradScaler, autocast

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=(use_amp and device.type == "cuda"))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        hog_feats = batch.get("hog_pca")
        hog_feats = hog_feats.to(device, non_blocking=True) if hog_feats is not None else None

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with autocast(device_type="cuda", dtype=torch.float16):
                out = model(images, hog_pca=hog_feats)
                loss = criterion(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images, hog_pca=hog_feats)
            loss = criterion(out["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = out["logits"].argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def _validate(model: Any, loader: Any, criterion: Any, device: Any) -> Tuple[float, float, float]:
    import torch
    import numpy as np
    from sklearn.metrics import f1_score

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: list = []
    all_predictions: list = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            hog_feats = batch.get("hog_pca")
            hog_feats = hog_feats.to(device, non_blocking=True) if hog_feats is not None else None

            out = model(images, hog_pca=hog_feats)
            logits = out["logits"]
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

            all_targets.append(labels.detach().cpu().numpy())
            all_predictions.append(preds.detach().cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    if all_targets:
        val_f1 = float(
            f1_score(
                np.concatenate(all_targets),
                np.concatenate(all_predictions),
                average="weighted",
                zero_division=0,
            )
        )
    else:
        val_f1 = 0.0
    return avg_loss, acc, val_f1


def create_or_load_study(
    study_dir: Path,
    study_name: str,
    storage_backend: str,
    direction: str,
    sampler_name: str,
    pruner_name: str,
    sampler_kwargs: Optional[Dict[str, Any]],
    pruner_kwargs: Optional[Dict[str, Any]],
    load_if_exists: bool,
) -> Any:
    import optuna

    if sampler_kwargs is None:
        sampler_kwargs = {}
    if pruner_kwargs is None:
        pruner_kwargs = {}

    if storage_backend.lower() in {"sqlite", "sqlite3"} or str(study_dir).endswith(".db"):
        db_path = study_dir.resolve() / f"{study_name}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{db_path}"
        storage = storage_url
    else:
        storage = None

    if sampler_name == "TPESampler":
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    elif sampler_name == "CmaEsSampler":
        sampler = optuna.integration.optuna_integration.CmaEsSampler(
            **sampler_kwargs
        )
    elif sampler_name == "NSGAIISampler":
        sampler = optuna.samplers.NSGAIISampler(**sampler_kwargs)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    if pruner_name == "MedianPruner":
        pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
    elif pruner_name == "HyperbandPruner":
        pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
    elif pruner_name == "HalvingPruner":
        pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
    else:
        raise ValueError(f"Unsupported pruner: {pruner_name}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )
    return study


def run_study(
    data_dir: Path,
    *,
    n_trials: int,
    study_name: str,
    study_dir: Path,
    sampler_name: str,
    pruner_name: str,
    sampler_kwargs: Optional[Dict[str, Any]],
    pruner_kwargs: Optional[Dict[str, Any]],
    direction: str,
    load_if_exists: bool,
    epochs: int,
    batch_size: int,
    patience: int,
    pca_components: int,
    seed: int,
    device: Any,
    checkpoint_dir: Optional[Path],
    timeout: Optional[int],
    resume_best_trial: bool,
) -> Dict[str, Any]:
    import optuna

    logging.getLogger("optuna").setLevel(logging.WARNING)

    study = create_or_load_study(
        study_dir=study_dir,
        study_name=study_name,
        storage_backend="sqlite",
        direction=direction,
        sampler_name=sampler_name,
        pruner_name=pruner_name,
        sampler_kwargs=sampler_kwargs,
        pruner_kwargs=pruner_kwargs,
        load_if_exists=load_if_exists,
    )

    def objective(trial: Any) -> float:
        params = build_trial_param_space(trial)
        logger.info("Trial %s params: %s", trial.number, params)

        run_kwargs = dict(
            data_dir=data_dir,
            params=params,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            pca_components=pca_components,
            seed=seed + trial.number,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        result = run_trial_training(**run_kwargs)
        logger.info(
            "Trial %s finished: best_val_f1=%.4f",
            trial.number,
            result["best_val_f1"],
        )
        return result["best_val_f1"]

    start = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        catch=(ValueError, RuntimeError, Exception),
        show_progress_bar=False,
    )
    elapsed = time.time() - start

    completed_trials = [t for t in study.trials if t.state.name == "COMPLETED"]
    if completed_trials:
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_value = float(best_trial.value)
    else:
        best_params = {}
        best_value = 0.0

    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_value": best_value,
        "best_params": best_params,
        "elapsed_seconds": float(elapsed),
        "completed_trials": [t.number for t in study.trials if t.state.name == "COMPLETED"],
        "failed_trials": [t.number for t in study.trials if t.state.name != "COMPLETED"],
        "has_best_trial": bool(completed_trials),
    }
    logger.info(
        "Optuna study complete: best_val_f1=%.4f | best params: %s",
        best_value,
        best_params,
    )
    return summary


def _pick_device(device_str: Optional[str]) -> Any:
    import torch
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="tune_optuna.py",
        description="FreeBuff v4 Optuna hyperparameter tuning",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory containing class subfolders",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per trial",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per trial",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience per trial",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=288,
        help="PCA components for HOG branch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="freebuff_v4_tuning",
        help="Optuna study name",
    )
    parser.add_argument(
        "--study_dir",
        type=str,
        default="studies",
        help="Directory for SQLite study storage",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="TPESampler",
        choices=["TPESampler", "CmaEsSampler", "NSGAIISampler"],
        help="Optuna sampler",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="MedianPruner",
        choices=["MedianPruner", "HyperbandPruner", "HalvingPruner"],
        help="Optuna pruner",
    )
    parser.add_argument(
        "--sampler_kwargs",
        type=str,
        default=None,
        help="JSON encoded sampler kwargs",
    )
    parser.add_argument(
        "--pruner_kwargs",
        type=str,
        default=None,
        help="JSON encoded pruner kwargs",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume an existing study",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for the study",
    )
    parser.add_argument(
        "--resume_best_trial",
        action="store_true",
        help="After study, run a final full training trial with best params",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose)

    sampler_kwargs = None
    if args.sampler_kwargs:
        sampler_kwargs = json.loads(args.sampler_kwargs)

    pruner_kwargs = None
    if args.pruner_kwargs:
        pruner_kwargs = json.loads(args.pruner_kwargs)

    data_dir = Path(args.data_dir).expanduser().resolve()
    study_dir = Path(args.study_dir).expanduser().resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = study_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)

    logger.info(
        "Starting Optuna tuning: data=%s, n_trials=%d, epochs=%d, device=%s",
        data_dir,
        args.n_trials,
        args.epochs,
        device,
    )

    summary = run_study(
        data_dir=data_dir,
        n_trials=args.n_trials,
        study_name=args.study_name,
        study_dir=study_dir,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
        sampler_kwargs=sampler_kwargs,
        pruner_kwargs=pruner_kwargs,
        direction=args.direction,
        load_if_exists=not args.no_resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        pca_components=args.pca_components,
        seed=args.seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        timeout=args.timeout,
        resume_best_trial=args.resume_best_trial,
    )

    report_path = study_dir / f"{args.study_name}_best_params.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote best-params report to %s", report_path)

    return summary


if __name__ == "__main__":
    main()
