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

from modeling.datasets import build_dataloader
from modeling.lstm import LSTMRegressor
from modeling.metrics import r2_score_np


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
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
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
        val_r2 = r2_score_np(y_true, y_pred)
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

    if not best_state:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, history


def time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, int],
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

        model = LSTMRegressor(**model_kwargs)
        best_state, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_kwargs["epochs"],
            device=device,
            patience=train_kwargs["patience"],
            lr=train_kwargs["lr"],
        )

        y_true = y_val
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = []
            for X_batch, _ in val_loader:
                preds.append(model(X_batch.to(device)).cpu().numpy())
        y_pred = np.concatenate(preds)[:, 0]
        fold_r2 = r2_score_np(y_true, y_pred)
        logger.info("Fold %d R²: %.4f", fold, fold_r2)
        results.append({"fold": fold, "r2": fold_r2, "history": history})

    return results


def save_cv_report(results: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {"fold": r["fold"], "r2": r["r2"], "history": r["history"]}
        for r in results
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    logger.info("Saved CV report to %s", path)
