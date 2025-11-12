"""Demonstration stub for live momentum predictions."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from loguru import logger

from artifact_paths import resolve_backend_paths
from predict_live import LivePredictor, fetch_latest_bar_stub
from run_train import load_config


STUB_ITERATIONS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live prediction stub")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    live_cfg = config.get("live", {})
    polling_minutes = live_cfg.get("polling_minutes", 30)
    tickers = config["tickers"]
    time_steps = config["time_steps"]
    device = torch.device(config["train"].get("device", "cpu"))

    pytorch_paths = resolve_backend_paths(paths_cfg, "pytorch")
    model_kwargs = {
        "in_features": len(set(config.get("features", [])) | {"ret_1", "log_ret_1"}),
        "hidden_size": config["train"].get("hidden_size", 64),
        "num_layers": config["train"].get("num_layers", 1),
        "dropout": config["train"].get("dropout", 0.0),
        "bidirectional": bool(config["train"].get("bidirectional", False)),
        "head_hidden_size": config["train"].get("head_hidden_size"),
        "head_dropout": config["train"].get("head_dropout", 0.0),
        "layer_norm": bool(config["train"].get("layer_norm", False)),
    }

    predictor = LivePredictor(
        model_path=pytorch_paths.model,
        scaler_path=pytorch_paths.scaler,
        model_kwargs=model_kwargs,
        model_type=config.get("model", "lstm"),
        time_steps=time_steps,
        tickers=tickers,
        max_history_bars=live_cfg.get("max_history_bars", 200),
        device=device,
    )

    logger.info("Starting live prediction stub for %d iterations", STUB_ITERATIONS)
    for iteration in range(1, STUB_ITERATIONS + 1):
        if live_cfg.get("offline_mode", False):
            bars = fetch_latest_bar_stub(tickers, polling_minutes)
        else:
            logger.warning("Real-time Refinitiv integration not implemented. Using stub data.")
            bars = fetch_latest_bar_stub(tickers, polling_minutes)

        logger.info("Iteration %d: fetched %d bars", iteration, len(bars))
        result = predictor.update(bars, config)
        if result.empty:
            logger.info("Not enough data accumulated yet")
        else:
            logger.info("Live signals:\n%s", result.tail())

        time.sleep(0.1)  # placeholder for real scheduling logic

    logger.info("Live stub finished. TODO: integrate scheduler & real data feed.")


if __name__ == "__main__":
    main()
