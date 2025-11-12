"""Helpers for resolving backend-specific artifact paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class BackendArtifactPaths:
    """Container describing all filesystem artifacts for a backend."""

    model: Path
    scaler: Path
    predictions: Path
    metrics: Path
    cv_report: Path


_DEFAULTS = {
    "model": "artifacts/model.pt",
    "scaler": "artifacts/scaler.pkl",
    "predictions": "artifacts/predictions.csv",
    "metrics": "artifacts/metrics.json",
    "cv_report": "artifacts/cv_metrics.json",
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def resolve_backend_paths(paths_cfg: Mapping[str, Any] | None, backend: str) -> BackendArtifactPaths:
    """Resolve artifact paths for ``backend`` with graceful fallbacks."""

    paths_cfg = _as_mapping(paths_cfg)
    backend_cfg = _as_mapping(paths_cfg.get(backend))

    def _resolve(key: str) -> Path:
        candidate = backend_cfg.get(key)
        if candidate is None:
            candidate = paths_cfg.get(key)
        if candidate is None:
            candidate = _DEFAULTS[key]
        return Path(candidate)

    return BackendArtifactPaths(
        model=_resolve("model"),
        scaler=_resolve("scaler"),
        predictions=_resolve("predictions"),
        metrics=_resolve("metrics"),
        cv_report=_resolve("cv_report"),
    )
