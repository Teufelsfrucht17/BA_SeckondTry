"""Sklearn baseline training pipeline (no PyTorch required)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.engineering import build_features
from features.scaler import fit_scaler, save_scaler, transform
from features.sequencing import grouped_sequences
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

    # Flatten sequences for sklearn (N, T, F) -> (N, T*F)
    def flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    X_train_f = flatten(X_train)
    X_test_f = flatten(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_f, y_train)
    y_pred = model.predict(X_test_f)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    logger.info("Sklearn baseline metrics: %s", metrics)

    # Save scaler to keep shape consistent with rest of pipeline
    scaler_path = Path(config["paths"]["scaler"])
    save_scaler(scaler, scaler_path)

    # Save predictions similar to torch pipeline
    timestamps = test_df.sort_values(["ric", "ts"])["ts"].iloc[time_steps - 1 : time_steps - 1 + len(y_test)]
    pred_path = Path(config["paths"]["predictions"]).with_name("predictions_sklearn.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ts": list(timestamps),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }).to_csv(pred_path, index=False)
    logger.info("Saved sklearn predictions to %s", pred_path)


if __name__ == "__main__":
    main()

