"""Scaler utilities for consistent feature normalization."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler


ScalerType = StandardScaler


def fit_scaler(X: np.ndarray) -> ScalerType:
    """Fit a :class:`StandardScaler` on training features."""

    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))
    logger.info("Fitted scaler on shape %s", X.shape)
    return scaler


def transform(X: np.ndarray, scaler: ScalerType) -> np.ndarray:
    """Transform data using a fitted scaler."""

    original_shape = X.shape
    flattened = X.reshape(-1, X.shape[-1])
    transformed = scaler.transform(flattened)
    return transformed.reshape(original_shape)


def save_scaler(scaler: ScalerType, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(scaler, handle)
    logger.info("Saved scaler to %s", path)


def load_scaler(path: Path) -> ScalerType:
    with path.open("rb") as handle:
        scaler = pickle.load(handle)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Loaded object is not a StandardScaler")
    logger.info("Loaded scaler from %s", path)
    return scaler
