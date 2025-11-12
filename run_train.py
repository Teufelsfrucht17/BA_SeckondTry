"""Entry point for training the DAX momentum model using PyTorch."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas import DatetimeTZDtype
import yaml
from loguru import logger

from engineering import build_features
from scaler import fit_scaler, save_scaler, transform
from sequencing import grouped_sequences


try:
    import torch  # type: ignore
    from datasets import build_dataloader
    from evaluate import evaluate_model, save_predictions
    from models import build_model
    from train import (
        save_cv_report,
        set_seed,
        time_series_cv,
        train_model,
    )
except Exception as exc:  # pragma: no cover - explicit guidance for missing deps
    raise ImportError(
        "PyTorch training dependencies are unavailable. "
        "Install the PyTorch stack or run `python -m pipeline.run_train_sklearn` "
        "for the sklearn baseline."
    ) from exc

TEST_SHARE = 0.1

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_train_test(df: pd.DataFrame, test_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust time-based split that tolerates missing/invalid timestamps.

    - Ensures 'ts' is datetime (UTC) and drops invalid rows.
    - Handles edge cases with 0 or 1 unique timestamps.
    - Guarantees a non-empty test set when possible.
    """

    # Ensure ts is datetime (supports tz-aware types)
    if not is_datetime64_any_dtype(df["ts"]):
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        # Normalize to UTC consistently
        df = df.copy()
        if isinstance(df["ts"].dtype, DatetimeTZDtype):
            df["ts"] = df["ts"].dt.tz_convert("UTC")
        else:
            df["ts"] = df["ts"].dt.tz_localize("UTC")

    # Drop rows with invalid timestamps
    df = df.dropna(subset=["ts"])  # type: ignore[arg-type]

    timestamps = pd.Series(df["ts"]).sort_values().unique()
    n_ts = len(timestamps)
    if n_ts == 0:
        # Fallback: index-based split when timestamps are unavailable
        logger.warning(
            "No valid timestamps found; falling back to index-based split."
        )
        n = len(df)
        if n <= 1:
            return df.copy(), df.iloc[0:0].copy()
        split_idx = int(n * (1 - test_share))
        split_idx = max(1, min(split_idx, n - 1))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    if n_ts == 1:
        # All data at one timestamp: keep all for train, empty test
        return df.copy(), df.iloc[0:0].copy()

    cutoff_index = int(n_ts * (1 - test_share))
    cutoff_index = max(1, min(cutoff_index, n_ts - 1))
    cutoff_ts = timestamps[cutoff_index]

    train_df = df[df["ts"] <= cutoff_ts].copy()
    test_df = df[df["ts"] > cutoff_ts].copy()

    # Guarantee non-empty test if possible
    if test_df.empty and n_ts >= 2:
        cutoff_ts = timestamps[-2]
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

    # Select features: required base features plus only extra columns that are fully numeric and non-NaN
    base_feats = set(config.get("features", [])) | {"log_ret_1", "ret_1"}
    extra: set[str] = set()
    for col in feature_df.columns:
        if col in {"ric", "ts", "y_next"} or col in base_feats:
            continue
        series = pd.to_numeric(feature_df[col], errors="coerce")
        if series.notna().all() and np.isfinite(series.to_numpy()).all():
            extra.add(col)

    feature_columns = sorted(base_feats | extra)
    time_steps = config["time_steps"]

    X_train, y_train = grouped_sequences(train_df, feature_columns, "y_next", time_steps)
    X_test, y_test = grouped_sequences(test_df, feature_columns, "y_next", time_steps)

    scaler = fit_scaler(X_train)
    X_train = transform(X_train, scaler)
    X_test = transform(X_test, scaler)

    scaler_path = Path(config["paths"]["scaler"])

    # Import torch and modeling modules lazily to allow environments without torch
    try:
        import torch  # type: ignore
        from datasets import build_dataloader
        from evaluate import evaluate_model, save_predictions
        from lstm import LSTMRegressor
        from train import set_seed, time_series_cv, train_model, save_cv_report
        from metrics import r2_score_np, mse_np, mae_np
    except Exception as exc:
        logger.warning(
            "PyTorch stack unavailable (%s); falling back to sklearn baseline.",
            exc,
        )
        from sklearn_backend import train_sklearn_model

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
        logger.info("Sklearn fallback complete with metrics: %s", artifacts.metrics)
        return

    save_scaler(scaler, scaler_path)

    device = torch.device(config["train"].get("device", "cpu"))
    set_seed(42)

    model_type = str(config.get("model", "lstm")).lower()
    model_kwargs = {
        "in_features": X_train.shape[-1],
        "hidden_size": config["train"].get("hidden_size", 64),
        "num_layers": config["train"].get("num_layers", 1),
        "dropout": config["train"].get("dropout", 0.0),
    }

    train_kwargs = {
        "batch_size": config["train"].get("batch_size", 128),
        "epochs": config["train"].get("epochs", 25),
        "lr": config["train"].get("lr", 1e-3),
        "patience": config["train"].get("patience", 5),
        "weight_decay": config["train"].get("weight_decay", 0.0),
        "clip_norm": config["train"].get("clip_norm", 0.0),
        "loss": config["train"].get("loss", "mse"),
        "scheduler": config["train"].get("scheduler"),
        "lr_factor": config["train"].get("lr_factor", 0.5),
        "lr_patience": config["train"].get("lr_patience", 5),
        "min_lr": config["train"].get("min_lr", 1e-6),
    }

    if config.get("cv", {}).get("n_splits", 0) > 1:
        cv_results = time_series_cv(
            X_train,
            y_train,
            config["cv"],
            model_type,
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

    model = build_model(model_type, **model_kwargs)
    best_state, history_records = train_model(
        model,
        train_loader,
        val_loader,
        epochs=train_kwargs["epochs"],
        device=device,
        patience=train_kwargs["patience"],
        lr=train_kwargs["lr"],
        weight_decay=float(train_kwargs.get("weight_decay", 0.0)),
        loss=str(train_kwargs.get("loss", "mse")),
        clip_norm=float(train_kwargs.get("clip_norm", 0.0)),
        scheduler=train_kwargs.get("scheduler"),
        lr_factor=float(train_kwargs.get("lr_factor", 0.5)),
        lr_patience=int(train_kwargs.get("lr_patience", 5)),
        min_lr=float(train_kwargs.get("min_lr", 1e-6)),
    )

    model.load_state_dict(best_state)
    model_path = Path(config["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)
    logger.info("Saved best model to %s", model_path)

    # Compute train and test metrics using the in-memory model
    model.eval()
    with torch.no_grad():
        y_train_pred = model(torch.from_numpy(X_train).float().to(device)).cpu().numpy()[:, 0]
        y_test_pred = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()[:, 0]

    # Filter invalid values
    def _safe_metrics(y_true_np, y_pred_np):
        import numpy as _np
        mask = _np.isfinite(y_true_np) & _np.isfinite(y_pred_np)
        if not mask.all():
            logger.warning("Dropping %d invalid points for metric computation", int((~mask).sum()))
        y_t = y_true_np[mask]
        y_p = y_pred_np[mask]
        return {
            "r2": r2_score_np(y_t, y_p) if len(y_t) else float("nan"),
            "mse": mse_np(y_t, y_p) if len(y_t) else float("nan"),
            "mae": mae_np(y_t, y_p) if len(y_t) else float("nan"),
        }

    train_metrics = _safe_metrics(y_train, y_train_pred)
    test_metrics = _safe_metrics(y_test, y_test_pred)
    logger.info("Train metrics: %s", train_metrics)
    logger.info("Test metrics: %s", test_metrics)

    # Persist combined metrics to JSON
    metrics_combined = {
        "train_r2": train_metrics.get("r2"),
        "train_mse": train_metrics.get("mse"),
        "train_mae": train_metrics.get("mae"),
        "test_r2": test_metrics.get("r2"),
        "test_mse": test_metrics.get("mse"),
        "test_mae": test_metrics.get("mae"),
    }
    metrics_path = Path(config["paths"].get("metrics", "artifacts/metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_combined, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    timestamps = test_df.sort_values(["ric", "ts"])["ts"].iloc[time_steps - 1 : time_steps - 1 + len(y_test)]
    model = build_model(model_type, **model_kwargs)
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()[:, 0]

    save_predictions(
        Path(config["paths"]["predictions"]),
        timestamps,
        y_test,
        y_test_pred,
    )

    logger.info("Training complete. Train R2=%.4f, Test R2=%.4f", train_metrics.get("r2", float("nan")), test_metrics.get("r2", float("nan")))


if __name__ == "__main__":
    main()
