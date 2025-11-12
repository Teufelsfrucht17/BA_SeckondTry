"""Model evaluation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from loguru import logger

from models import build_model
from metrics import mae_np, mse_np, r2_score_np


def evaluate_model(
    model_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    model_kwargs: Dict[str, int],
    device: torch.device,
    *,
    model_type: str = "lstm",
) -> Dict[str, float]:
    """Evaluate a saved PyTorch model on given data.

    Parameters
    - model_path: path to a saved state_dict (torch.save of model.state_dict()).
    - X, y: feature sequences and targets.
    - model_kwargs: keyword args required to build the model (e.g., in_features,...).
    - device: torch.device for inference.
    - model_type: one of {"lstm", "mlp", "linear"}; defaults to "lstm".
    """

    model = build_model(model_type, **model_kwargs)
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
    mask = np.isfinite(y) & np.isfinite(y_pred)
    if not mask.all():
        logger.warning("Dropping %d invalid points for evaluation metrics", int((~mask).sum()))
    y_true = y[mask]
    y_pred = y_pred[mask]
    metrics = {
        "r2": r2_score_np(y_true, y_pred) if len(y_true) else float("nan"),
        "mse": mse_np(y_true, y_pred) if len(y_true) else float("nan"),
        "mae": mae_np(y_true, y_pred) if len(y_true) else float("nan"),
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
