"""Evaluation metrics used throughout the project."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))
