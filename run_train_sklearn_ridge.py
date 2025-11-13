"""Train the configured Ridge Regression sklearn model."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from run_train import load_config
from run_train_all_models import get_model_spec, train_named_sklearn_model
from run_train_sklearn import prepare_sklearn_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Ridge sklearn model")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = prepare_sklearn_dataset(config)
    spec = get_model_spec(config, "ridge", default_model_type="ridge")
    result = train_named_sklearn_model(
        dataset,
        config,
        spec,
        name="ridge",
    )
    logger.info("Completed Ridge training with metrics: %s", result.get("metrics"))


if __name__ == "__main__":
    main()
