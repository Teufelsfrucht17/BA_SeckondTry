"""Utility helpers for signal creation and evaluation."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


SIGNAL_MAP = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def to_signal(prediction: float, threshold: float = 0.0) -> str:
    if prediction > threshold:
        return "LONG"
    if prediction < -threshold:
        return "SHORT"
    return "FLAT"


def batch_signals(predictions: Iterable[float], threshold: float = 0.0) -> pd.Series:
    return pd.Series([to_signal(float(p), threshold) for p in predictions], name="signal")


def signal_summary(signals: pd.Series, realized_returns: pd.Series) -> pd.DataFrame:
    if len(signals) != len(realized_returns):
        raise ValueError("Signals and returns must be the same length")

    hits = ((signals == "LONG") & (realized_returns > 0)) | ((signals == "SHORT") & (realized_returns < 0))
    hit_rate = float(hits.sum()) / max(len(signals), 1)
    distribution = signals.value_counts(normalize=True).rename("share")
    summary = pd.DataFrame({"hit_rate": [hit_rate]})
    distribution_df = distribution.reset_index().rename(columns={"index": "signal"})
    return summary, distribution_df
