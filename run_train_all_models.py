"""Run every configured model backend (sklearn variants and PyTorch NN)."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from loguru import logger

from artifact_paths import resolve_backend_paths
from run_train import load_config
from run_train_sklearn import SklearnDataset, prepare_sklearn_dataset
from sklearn_backend import train_sklearn_model


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` if mapping-like, otherwise an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _resolve_model_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract model configurations from the ``sklearn`` config block."""

    sklearn_cfg = as_mapping(config.get("sklearn"))
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


def get_model_spec(
    config: Mapping[str, Any],
    target_name: str,
    *,
    default_model_type: str,
) -> Dict[str, Any]:
    """Return the sklearn spec matching ``target_name`` or a default one."""

    target = target_name.strip().lower()
    specs = _resolve_model_specs(config)
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            continue
        if model_name(spec, idx) == target:
            return dict(spec)

    return {"model_type": default_model_type, "name": target}


def model_name(spec: Mapping[str, Any], index: int) -> str:
    candidate = spec.get("name") or spec.get("model_type") or spec.get("model")
    if candidate:
        return str(candidate).strip().lower()
    return f"model_{index}"


def build_backend_paths(paths_cfg: Mapping[str, Any], name: str) -> Dict[str, str]:
    """Construct backend-specific artifact paths for ``name``."""

    sklearn_cfg = as_mapping(paths_cfg.get("sklearn"))
    model_path = Path(sklearn_cfg.get("model", "artifacts/sklearn/best_model.joblib"))
    root = model_path.parent / name
    return {
        "model": str(root / "best_model.joblib"),
        "scaler": str(root / "scaler.pkl"),
        "predictions": str(root / "predictions.csv"),
        "metrics": str(root / "metrics.json"),
        "cv_report": str(root / "cv_metrics.json"),
    }


def summary_path(paths_cfg: Mapping[str, Any]) -> Path:
    """Determine where to write the consolidated metrics summary."""

    paths_cfg = dict(paths_cfg) if isinstance(paths_cfg, MutableMapping) else dict(as_mapping(paths_cfg))

    for key in ("model_summary", "summary", "summary_metrics"):
        candidate = paths_cfg.get(key)
        if candidate:
            return Path(candidate)

    sklearn_cfg = as_mapping(paths_cfg.get("sklearn"))
    for key in ("summary", "summary_metrics"):
        candidate = sklearn_cfg.get(key)
        if candidate:
            return Path(candidate)

    model_candidate = sklearn_cfg.get("model")
    if model_candidate:
        return Path(model_candidate).parent / "summary_metrics.json"

    return Path("artifacts/summary_metrics.json")


def train_named_sklearn_model(
    dataset: SklearnDataset,
    base_config: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    *,
    name: str,
    backend_name: str | None = None,
) -> Dict[str, Any]:
    """Train a single sklearn model using ``model_spec`` and return diagnostics."""

    config_copy: Dict[str, Any] = copy.deepcopy(dict(base_config))
    spec_dict = dict(model_spec)

    backend_name = backend_name or f"sklearn_{name}"
    paths_cfg = dict(as_mapping(config_copy.get("paths")))
    backend_paths = build_backend_paths(paths_cfg, name)
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

    result = {
        "name": name,
        "backend": backend_name,
        "model_type": artifacts.model_type,
        "metrics": artifacts.metrics,
        "config": spec_dict,
        "artifacts": {
            "model": str(artifacts.model_path),
            "predictions": str(artifacts.predictions_path),
            "scaler": str(artifacts.scaler_path),
            "cv_report": str(artifacts.cv_report_path) if artifacts.cv_report_path else None,
            "metrics_file": backend_paths["metrics"],
        },
        "status": "success",
    }

    return result


def run_pytorch_model(config_path: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute the PyTorch training pipeline and collect metrics."""

    model_label = str(config.get("model", "lstm"))

    try:  # Ensure PyTorch is available before spawning the subprocess
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.warning("Skipping PyTorch training due to missing dependency: %s", exc)
        return {
            "name": "pytorch",
            "backend": "pytorch",
            "model_type": model_label,
            "status": "skipped",
            "reason": f"Missing PyTorch dependency: {exc}",
        }

    script_path = Path(__file__).with_name("run_train.py")
    command = [sys.executable, str(script_path), "--config", str(config_path)]

    try:
        subprocess.run(command, check=True, capture_output=False)
    except FileNotFoundError as exc:
        logger.error("PyTorch training script missing: %s", exc)
        return {
            "name": "pytorch",
            "backend": "pytorch",
            "model_type": model_label,
            "status": "error",
            "error": f"Script not found: {exc}",
        }
    except subprocess.CalledProcessError as exc:
        logger.error("PyTorch training failed with return code %d", exc.returncode)
        return {
            "name": "pytorch",
            "backend": "pytorch",
            "model_type": model_label,
            "status": "error",
            "error": f"Training failed with return code {exc.returncode}",
        }

    paths_cfg = as_mapping(config.get("paths"))
    pytorch_paths = resolve_backend_paths(paths_cfg, "pytorch")
    metrics_path = pytorch_paths.metrics
    metrics: Dict[str, Any] = {}

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        logger.warning("Expected PyTorch metrics at %s but file is missing", metrics_path)

    return {
        "name": "pytorch",
        "backend": "pytorch",
        "model_type": model_label,
        "metrics": metrics,
        "status": "success" if metrics else "unknown",
        "artifacts": {
            "model": str(pytorch_paths.model),
            "predictions": str(pytorch_paths.predictions),
            "scaler": str(pytorch_paths.scaler),
            "cv_report": str(pytorch_paths.cv_report),
            "metrics_file": str(metrics_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all configured models")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    dataset = prepare_sklearn_dataset(config)

    paths_cfg = as_mapping(config.get("paths"))
    summary = summary_path(paths_cfg)

    results: List[Dict[str, Any]] = []

    specs = _resolve_model_specs(config)
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            continue
        name = model_name(spec, idx)
        logger.info("Training sklearn model '%s' with spec %s", name, spec)
        result = train_named_sklearn_model(dataset, config, spec, name=name)
        results.append(result)

    pytorch_result = run_pytorch_model(config_path, config)
    results.append(pytorch_result)

    summary_payload = {"models": results}
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    logger.info("Wrote aggregated model summary to %s", summary)


if __name__ == "__main__":
    main()
