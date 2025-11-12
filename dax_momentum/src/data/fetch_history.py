"""CLI utility to fetch historical OHLCV data and persist it as parquet."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from loguru import logger

from .refinitiv_client import fetch_history


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical data from Refinitiv")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    tickers = config["tickers"]
    history_cfg = config["history"]
    interval = config["interval"]
    paths_cfg = config.get("paths", {})
    offline_mode = config.get("live", {}).get("offline_mode", False)

    output_path = Path(paths_cfg.get("history", "artifacts/history.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching history for %s", tickers)
    df = fetch_history(
        tickers=tickers,
        start=history_cfg["start"],
        end=history_cfg["end"],
        interval=interval,
        offline_mode=offline_mode,
    )

    logger.info("Writing history to %s", output_path)
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
