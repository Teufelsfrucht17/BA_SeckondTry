"""Generate trading signals on the test set as a backtest stub."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from artifact_paths import resolve_backend_paths
from engineering import build_features
from scaler import load_scaler, transform
from sequencing import grouped_sequences
from models import build_model
from utils import batch_signals, signal_summary
from run_train import TEST_SHARE, load_config, split_train_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest signal generation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    history_path = Path(paths_cfg.get("history", "artifacts/history.parquet"))
    if not history_path.exists():
        raise FileNotFoundError("History data not found. Run fetch_history first.")

    pytorch_paths = resolve_backend_paths(paths_cfg, "pytorch")

    history = pd.read_parquet(history_path)
    feature_df = build_features(history, config)
    _, test_df = split_train_test(feature_df, TEST_SHARE)

    feature_columns = sorted(
        set(config.get("features", [])) | {"log_ret_1", "ret_1"} | {
            col
            for col in feature_df.columns
            if col not in {"ric", "ts", "y_next"}
        }
    )

    time_steps = config["time_steps"]
    X_test, y_test = grouped_sequences(test_df, feature_columns, "y_next", time_steps)
    scaler = load_scaler(pytorch_paths.scaler)
    X_test = transform(X_test, scaler)

    device = torch.device(config["train"].get("device", "cpu"))
    model = build_model(
        config.get("model", "lstm"),
        in_features=X_test.shape[-1],
        hidden_size=config["train"].get("hidden_size", 64),
        num_layers=config["train"].get("num_layers", 1),
        dropout=config["train"].get("dropout", 0.0),
        bidirectional=bool(config["train"].get("bidirectional", False)),
        head_hidden_size=config["train"].get("head_hidden_size"),
        head_dropout=config["train"].get("head_dropout", 0.0),
        layer_norm=bool(config["train"].get("layer_norm", False)),
    )
    state_dict = torch.load(pytorch_paths.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()[:, 0]

    signals = batch_signals(preds)
    realized = pd.Series(y_test, name="y_true")
    summary, distribution = signal_summary(signals, realized)

    output_path = Path(config["paths"]["signals"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_frame = pd.DataFrame({
        "prediction": preds,
        "signal": signals.values,
        "y_true": realized,
    })
    signal_frame.to_csv(output_path, index=False)

    logger.info("Signal summary: %s", summary.to_dict("records"))
    logger.info("Distribution: %s", distribution.to_dict("records"))
    logger.info("Signals saved to %s", output_path)

    # TODO: integrate with a full PnL backtesting engine.


if __name__ == "__main__":
    main()
