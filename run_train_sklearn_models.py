"""Train multiple sklearn models and aggregate their evaluation metrics."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from loguru import logger

from run_train import load_config
from run_train_sklearn import SklearnDataset, prepare_sklearn_dataset
from sklearn_backend import train_sklearn_model


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_model_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract model configurations from the ``sklearn`` config block."""

    sklearn_cfg = _as_mapping(config.get("sklearn"))
    models = sklearn_cfg.get("models")
    if isinstance(models, Iterable) and not isinstance(models, (str, bytes)):
        specs: List[Dict[str, Any]] = []
        for entry in models:
            if isinstance(entry, Mapping):
                specs.append(dict(entry))
        if specs:
            return specs

    fallback = dict(sklearn_cfg)
    fallback.pop("models", None)
    return [fallback]


def _model_name(spec: Mapping[str, Any], index: int) -> str:
    candidate = spec.get("name") or spec.get("model_type") or spec.get("model")
    if candidate:
        return str(candidate).strip().lower()
    return f"model_{index}"


def _build_backend_paths(
    paths_cfg: Mapping[str, Any],
    name: str,
) -> Dict[str, str]:
    """Construct backend-specific artifact paths for ``name``."""

    base_backend = _as_mapping(paths_cfg.get("sklearn"))
    model_path = Path(base_backend.get("model", "artifacts/sklearn/best_model.joblib"))
    root = model_path.parent / name
    return {
        "model": str(root / "best_model.joblib"),
        "scaler": str(root / "scaler.pkl"),
        "predictions": str(root / "predictions.csv"),
        "metrics": str(root / "metrics.json"),
        "cv_report": str(root / "cv_metrics.json"),
    }


def _summary_path(paths_cfg: Mapping[str, Any]) -> Path:
    base_backend = _as_mapping(paths_cfg.get("sklearn"))
    model_path = Path(base_backend.get("model", "artifacts/sklearn/best_model.joblib"))
    default_summary = model_path.parent / "summary_metrics.json"
    override = paths_cfg.get("sklearn_summary")
    return Path(override) if override is not None else default_summary


def _run_model(
    dataset: SklearnDataset,
    base_config: Dict[str, Any],
    model_spec: Mapping[str, Any],
    backend_name: str,
    backend_paths: Mapping[str, str],
) -> Dict[str, Any]:
    """Train a single model using ``model_spec`` and return diagnostics."""

    spec_dict = dict(model_spec)
    config_copy = copy.deepcopy(base_config)
    paths_cfg = dict(_as_mapping(config_copy.get("paths")))
    paths_cfg[backend_name] = dict(backend_paths)
    config_copy["paths"] = paths_cfg

    artifacts = train_sklearn_model(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        config=config_copy,
        train_df=dataset.train_df,
        test_df=dataset.test_df,
        time_steps=dataset.time_steps,
        scaler=dataset.scaler,
        backend_name=backend_name,
        model_config=spec_dict,
    )

    return {
        "backend": backend_name,
        "model_type": artifacts.model_type,
        "metrics": artifacts.metrics,
        "config": spec_dict,
        "artifacts": {
            "model": str(artifacts.model_path),
            "predictions": str(artifacts.predictions_path),
            "scaler": str(artifacts.scaler_path),
            "cv_report": str(artifacts.cv_report_path) if artifacts.cv_report_path else None,
            "metrics_file": str(paths_cfg[backend_name]["metrics"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple sklearn models")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = prepare_sklearn_dataset(config)

    paths_cfg = _as_mapping(config.get("paths"))
    summary_path = _summary_path(paths_cfg)

    specs = _resolve_model_specs(config)
    results: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            continue
        name = _model_name(spec, idx)
        backend_name = f"sklearn_{name}"
        backend_paths = _build_backend_paths(paths_cfg, name)
        logger.info("Training sklearn model '%s' with spec %s", name, spec)
        result = _run_model(dataset, config, spec, backend_name, backend_paths)
        result["name"] = name
        results.append(result)

    summary_payload = {"models": results}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    logger.info("Wrote sklearn summary metrics to %s", summary_path)


if __name__ == "__main__":
    main()
