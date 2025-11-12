"""Feature engineering utilities for DAX momentum modeling."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from loguru import logger


FEATURE_FUNCTIONS: Dict[str, str] = {
    "ret_1": "add_returns",
    "momentum_5": "add_momentum",
}


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and logarithmic returns to the dataframe."""

    df = df.copy()
    group = df.groupby("ric")["close"]
    df["ret_1"] = group.pct_change(periods=1)
    ratio = group.transform(lambda s: s / s.shift(1))
    ratio = ratio.replace({0: np.nan})
    df["log_ret_1"] = np.log(ratio)
    return df


def add_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add momentum features defined as price difference over ``window`` steps."""

    df = df.copy()
    df[f"momentum_{window}"] = df.groupby("ric")["close"].diff(periods=window)
    return df


def make_target(df: pd.DataFrame, target_col: str = "ret_1") -> pd.DataFrame:
    """Create the next-step target column from the given base feature."""

    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"Target source column '{target_col}' missing")
    df["y_next"] = df.groupby("ric")[target_col].shift(-1)
    return df


def build_features(df: pd.DataFrame, config: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """Apply feature pipeline defined in the configuration dictionary."""

    features: List[str] = list(config.get("features", []))
    target_name: str = config.get("target", "y_next")

    logger.info("Running feature pipeline with features=%s target=%s", features, target_name)

    df = df.sort_values(["ric", "ts"]).reset_index(drop=True)
    df = add_returns(df)

    for feature in features:
        if feature.startswith("momentum_"):
            window = int(feature.split("_")[1])
            df = add_momentum(df, window=window)
        elif feature == "ret_1":
            continue  # already computed
        else:
            logger.warning("Unknown feature '%s' - ignoring", feature)

    df = make_target(df, target_col="ret_1")
    df = df.dropna().reset_index(drop=True)
    return df
