"""Model evaluation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from loguru import logger

from modeling.lstm import LSTMRegressor
from modeling.metrics import mae_np, mse_np, r2_score_np


def evaluate_model(
    model_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    model_kwargs: Dict[str, int],
    device: torch.device,
) -> Dict[str, float]:
    model = LSTMRegressor(**model_kwargs)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.from_numpy(X[i : i + 512]).float().to(device)
            preds.append(model(batch).cpu().numpy())

    y_pred = np.concatenate(preds)[:, 0]
    metrics = {
        "r2": r2_score_np(y, y_pred),
        "mse": mse_np(y, y_pred),
        "mae": mae_np(y, y_pred),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def save_predictions(
    path: Path,
    timestamps: Iterable[pd.Timestamp],
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "ts": list(timestamps),
        "y_true": list(y_true),
        "y_pred": list(y_pred),
    })
    frame.to_csv(path, index=False)
    logger.info("Saved predictions to %s", path)
