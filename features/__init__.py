"""Convenience re-exports for the :mod:`features` package.

The training and prediction pipelines rely on ``from features import ...`` style
imports.  When ``__init__`` was empty these imports failed with
``ImportError``.  To keep the public API stable we expose the commonly used
helpers from here.
"""

from .engineering import (  # noqa: F401
    build_features,
    make_target,
    add_momentum,
    add_returns,
)
from .scaler import fit_scaler, load_scaler, save_scaler, transform  # noqa: F401
from .sequencing import grouped_sequences, make_sequences  # noqa: F401

__all__ = [
    "add_momentum",
    "add_returns",
    "build_features",
    "fit_scaler",
    "grouped_sequences",
    "load_scaler",
    "make_sequences",
    "make_target",
    "save_scaler",
    "transform",
]
