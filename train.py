"""Training utilities for the DAX momentum model."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.optim import Adam

from datasets import build_dataloader
from models import build_model
from metrics import r2_score_np


@dataclass
class TrainArtifacts:
    best_state_dict: Dict[str, torch.Tensor]
    history: List[Dict[str, float]]
    scaler_path: Path
    model_path: Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    patience: int,
    lr: float,
    *,
    weight_decay: float = 0.0,
    loss: str = "mse",
    clip_norm: float = 0.0,
    scheduler: str | None = None,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    min_lr: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    if loss.lower() in {"huber", "smoothl1"}:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_sched = None
    if scheduler and scheduler.lower() in {"plateau", "reduceonplateau", "reduce_lr_on_plateau"}:
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )
    best_r2 = -math.inf
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []
    wait = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_preds: List[np.ndarray] = []
        val_targets: List[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_batch.numpy())

        y_true = np.concatenate(val_targets)[:, 0]
        y_pred = np.concatenate(val_preds)[:, 0]
        # Guard against NaNs/Infs in predictions or targets
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not mask.all():
            logger.warning("Dropping %d invalid validation points for metric computation", int((~mask).sum()))
        safe_true = y_true[mask]
        safe_pred = y_pred[mask]
        val_r2 = r2_score_np(safe_true, safe_pred) if len(safe_true) else float("nan")
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_r2": val_r2})
        logger.info("Epoch %d: train_loss=%.6f val_r2=%.4f", epoch, epoch_loss, val_r2)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break
        if lr_sched is not None:
            lr_sched.step(val_r2)

    if not best_state:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, history


def time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, int],
    model_type: str,
    model_kwargs: Dict[str, int],
    train_kwargs: Dict[str, float],
    device: torch.device,
) -> List[Dict[str, float]]:
    splitter = TimeSeriesSplit(n_splits=config.get("n_splits", 3))
    results: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_loader = build_dataloader(X_train, y_train, batch_size=train_kwargs["batch_size"])
        val_loader = build_dataloader(X_val, y_val, batch_size=train_kwargs["batch_size"])

        model = build_model(model_type, **model_kwargs)
        best_state, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_kwargs["epochs"],
            device=device,
            patience=train_kwargs["patience"],
            lr=train_kwargs["lr"],
            weight_decay=float(train_kwargs.get("weight_decay", 0.0)),
            loss=str(train_kwargs.get("loss", "mse")),
            clip_norm=float(train_kwargs.get("clip_norm", 0.0)),
            scheduler=train_kwargs.get("scheduler"),
            lr_factor=float(train_kwargs.get("lr_factor", 0.5)),
            lr_patience=int(train_kwargs.get("lr_patience", 5)),
            min_lr=float(train_kwargs.get("min_lr", 1e-6)),
        )

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            # Train predictions for CV train R²
            preds_train = []
            for X_batch, _ in train_loader:
                preds_train.append(model(X_batch.to(device)).cpu().numpy())
            y_train_pred = np.concatenate(preds_train)[:, 0]

            # Validation predictions for CV val R²
            preds_val = []
            for X_batch, _ in val_loader:
                preds_val.append(model(X_batch.to(device)).cpu().numpy())
            y_val_pred = np.concatenate(preds_val)[:, 0]

        # Train R² with finite mask
        train_mask = np.isfinite(y_train) & np.isfinite(y_train_pred)
        if not train_mask.all():
            logger.warning(
                "Dropping %d invalid CV train points for metric computation",
                int((~train_mask).sum()),
            )
        safe_y_train = y_train[train_mask]
        safe_y_train_pred = y_train_pred[train_mask]
        train_r2 = r2_score_np(safe_y_train, safe_y_train_pred) if len(safe_y_train) else float("nan")

        # Val R² with finite mask
        val_mask = np.isfinite(y_val) & np.isfinite(y_val_pred)
        if not val_mask.all():
            logger.warning(
                "Dropping %d invalid CV val points for metric computation",
                int((~val_mask).sum()),
            )
        safe_y_val = y_val[val_mask]
        safe_y_val_pred = y_val_pred[val_mask]
        fold_r2 = r2_score_np(safe_y_val, safe_y_val_pred) if len(safe_y_val) else float("nan")
        logger.info("Fold %d R²: train=%.4f val=%.4f", fold, train_r2, fold_r2)
        results.append({"fold": fold, "r2": fold_r2, "train_r2": train_r2, "history": history})

    return results


def save_cv_report(results: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        item = {
            "fold": r.get("fold"),
            "r2": r.get("r2"),
            "history": r.get("history"),
        }
        if "train_r2" in r:
            item["train_r2"] = r["train_r2"]
        serializable.append(item)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    logger.info("Saved CV report to %s", path)
