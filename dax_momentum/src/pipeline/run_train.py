"""Entry point for training the DAX momentum model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from features import build_features
from features.scaler import fit_scaler, save_scaler, transform
from features.sequencing import grouped_sequences

TEST_SHARE = 0.1

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_train_test(df: pd.DataFrame, test_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = df["ts"].sort_values().unique()
    cutoff_index = int(len(timestamps) * (1 - test_share))
    cutoff_index = max(1, min(cutoff_index, len(timestamps) - 1))
    cutoff_ts = timestamps[cutoff_index]
    train_df = df[df["ts"] <= cutoff_ts].copy()
    test_df = df[df["ts"] > cutoff_ts].copy()
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DAX momentum model")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    history_path = Path(config["paths"]["history"])
    if not history_path.exists():
        raise FileNotFoundError(
            f"History file {history_path} missing. Run python -m src.data.fetch_history first."
        )

    logger.info("Loading history from %s", history_path)
    try:
        history = pd.read_parquet(history_path)
    except (ImportError, ModuleNotFoundError):
        # Parquet engine missing; try CSV fallback with same basename
        csv_path = history_path.with_suffix('.csv')
        if csv_path.exists():
            logger.warning("Parquet engine missing; loading CSV fallback at %s", csv_path)
            history = pd.read_csv(csv_path, parse_dates=["ts"]) if csv_path.exists() else None
        else:
            raise

    feature_df = build_features(history, config)
    train_df, test_df = split_train_test(feature_df, TEST_SHARE)

    feature_columns = sorted(
        set(config.get("features", [])) | {"log_ret_1", "ret_1"} | {
            col
            for col in feature_df.columns
            if col not in {"ric", "ts", "y_next"}
        }
    )
    time_steps = config["time_steps"]

    X_train, y_train = grouped_sequences(train_df, feature_columns, "y_next", time_steps)
    X_test, y_test = grouped_sequences(test_df, feature_columns, "y_next", time_steps)

    scaler = fit_scaler(X_train)
    X_train = transform(X_train, scaler)
    X_test = transform(X_test, scaler)

    scaler_path = Path(config["paths"]["scaler"])
    save_scaler(scaler, scaler_path)

    # Import torch and modeling modules lazily to allow environments without torch
    try:
        import torch  # type: ignore
        from modeling.datasets import build_dataloader
        from modeling.evaluate import evaluate_model, save_predictions
        from modeling.lstm import LSTMRegressor
        from modeling.train import set_seed, time_series_cv, train_model, save_cv_report
    except Exception as exc:
        logger.error(
            "PyTorch or modeling modules are unavailable (%s). "
            "If installing torch is not possible, run the sklearn baseline: "
            "python -m pipeline.run_train_sklearn --config %s",
            exc,
            args.config,
        )
        raise

    device = torch.device(config["train"].get("device", "cpu"))
    set_seed(42)

    model_kwargs = {
        "in_features": X_train.shape[-1],
        "hidden_size": config["train"]["hidden_size"],
    }

    train_kwargs = {
        "batch_size": config["train"]["batch_size"],
        "epochs": config["train"]["epochs"],
        "lr": config["train"]["lr"],
        "patience": config["train"]["patience"],
    }

    if config.get("cv", {}).get("n_splits", 0) > 1:
        cv_results = time_series_cv(
            X_train,
            y_train,
            config["cv"],
            model_kwargs,
            train_kwargs,
            device,
        )
        save_cv_report(cv_results, Path(config["paths"]["cv_report"]))

    val_size = max(1, int(0.1 * len(X_train)))
    if val_size >= len(X_train):
        val_size = max(1, len(X_train) // 5)
    split_idx = len(X_train) - val_size
    train_loader = build_dataloader(
        X_train[:split_idx],
        y_train[:split_idx],
        batch_size=train_kwargs["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        X_train[split_idx:],
        y_train[split_idx:],
        batch_size=train_kwargs["batch_size"],
        shuffle=False,
    )

    model = LSTMRegressor(**model_kwargs)
    best_state, history_records = train_model(
        model,
        train_loader,
        val_loader,
        epochs=train_kwargs["epochs"],
        device=device,
        patience=train_kwargs["patience"],
        lr=train_kwargs["lr"],
    )

    model.load_state_dict(best_state)
    model_path = Path(config["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)
    logger.info("Saved best model to %s", model_path)

    metrics = evaluate_model(
        model_path,
        X_test,
        y_test,
        {**model_kwargs},
        device,
    )

    timestamps = test_df.sort_values(["ric", "ts"])["ts"].iloc[time_steps - 1 : time_steps - 1 + len(y_test)]
    model = LSTMRegressor(**model_kwargs)
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()[:, 0]

    save_predictions(
        Path(config["paths"]["predictions"]),
        timestamps,
        y_test,
        predictions,
    )

    logger.info("Training complete with metrics: %s", metrics)


if __name__ == "__main__":
    main()
