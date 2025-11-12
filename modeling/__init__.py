"""Lazy loading facade for modeling utilities.

This module exposes the most frequently used helpers via attribute access
without importing heavy submodules (and their optional dependencies) eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

_ATTR_MAP: Dict[str, Tuple[str, str]] = {
    "LivePredictor": ("predict_live", "LivePredictor"),
    "LSTMRegressor": ("lstm", "LSTMRegressor"),
    "SklearnArtifacts": ("sklearn_backend", "SklearnArtifacts"),
    "TrainArtifacts": ("train", "TrainArtifacts"),
    "batch_signals": ("utils", "batch_signals"),
    "build_dataloader": ("datasets", "build_dataloader"),
    "evaluate_model": ("evaluate", "evaluate_model"),
    "fetch_latest_bar_stub": ("predict_live", "fetch_latest_bar_stub"),
    "mae_np": ("metrics", "mae_np"),
    "mse_np": ("metrics", "mse_np"),
    "r2_score_np": ("metrics", "r2_score_np"),
    "save_cv_report": ("train", "save_cv_report"),
    "save_predictions": ("evaluate", "save_predictions"),
    "set_seed": ("train", "set_seed"),
    "signal_summary": ("utils", "signal_summary"),
    "time_series_cv": ("train", "time_series_cv"),
    "to_signal": ("utils", "to_signal"),
    "train_model": ("train", "train_model"),
    "train_sklearn_model": ("sklearn_backend", "train_sklearn_model"),
}

__all__ = sorted(_ATTR_MAP)


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    try:
        module_name, attr_name = _ATTR_MAP[name]
    except KeyError as exc:  # pragma: no cover - mirrors default behaviour
        raise AttributeError(f"module 'modeling' has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - mirrors default behaviour
    return sorted(list(globals().keys()) + __all__)
