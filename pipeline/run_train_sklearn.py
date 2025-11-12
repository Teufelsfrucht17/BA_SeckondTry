"""Sklearn baseline training pipeline (no PyTorch required)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from features.engineering import build_features
from features.scaler import fit_scaler, transform
from features.sequencing import grouped_sequences
from modeling.sklearn_backend import train_sklearn_model
from pipeline.run_train import load_config, split_train_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn baseline (no torch)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    history_path = Path(config["paths"]["history"])
    if not history_path.exists() and not history_path.with_suffix(".csv").exists():
        raise FileNotFoundError(
            f"History file {history_path} (or CSV fallback) missing. Run data.fetch_history first."
        )

    logger.info("Loading history from %s", history_path)
    try:
        history = pd.read_parquet(history_path)
    except Exception:
        csv_path = history_path.with_suffix(".csv")
        history = pd.read_csv(csv_path, parse_dates=["ts"]) if csv_path.exists() else None
        if history is None:
            raise

    feature_df = build_features(history, config)
    train_df, test_df = split_train_test(feature_df, 0.1)

    feature_columns = sorted(
        set(config.get("features", [])) | {"log_ret_1", "ret_1"} | {
            col for col in feature_df.columns if col not in {"ric", "ts", "y_next"}
        }
    )
    time_steps = int(config.get("time_steps", 10))

    X_train, y_train = grouped_sequences(train_df, feature_columns, "y_next", time_steps)
    X_test, y_test = grouped_sequences(test_df, feature_columns, "y_next", time_steps)

    scaler = fit_scaler(X_train)
    X_train = transform(X_train, scaler)
    X_test = transform(X_test, scaler)

    artifacts = train_sklearn_model(
        X_train,
        y_train,
        X_test,
        y_test,
        config=config,
        train_df=train_df,
        test_df=test_df,
        time_steps=time_steps,
        scaler=scaler,
    )
    logger.info("Sklearn baseline complete with metrics: %s", artifacts.metrics)


if __name__ == "__main__":
    main()

