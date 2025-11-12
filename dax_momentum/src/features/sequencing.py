"""Sequence generation utilities for LSTM models."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def make_sequences(X: np.ndarray, y: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert feature matrix into sliding windows."""

    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    num_samples = X.shape[0] - time_steps + 1
    if num_samples <= 0:
        raise ValueError("Not enough samples to create sequences")

    sequences = np.stack([
        X[i : i + time_steps] for i in range(num_samples)
    ])
    targets = y[time_steps - 1 : time_steps - 1 + num_samples]
    return sequences, targets


def grouped_sequences(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    time_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sequences per RIC and concatenate them."""

    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for ric, group in df.groupby("ric"):
        group = group.sort_values("ts")
        X = group[list(feature_columns)].to_numpy(dtype=float)
        y = group[target_column].to_numpy(dtype=float)
        try:
            seq_X, seq_y = make_sequences(X, y, time_steps)
        except ValueError as exc:
            logger.warning("Skipping RIC %s: %s", ric, exc)
            continue
        features.append(seq_X)
        targets.append(seq_y)

    if not features:
        raise ValueError("No sequences created; check data availability")

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)
