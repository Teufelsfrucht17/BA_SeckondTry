"""Compatibility wrapper exposing ``dax_momentum.pipeline`` as top-level ``pipeline``."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_package_root = Path(__file__).resolve().parent.parent / "dax_momentum"
if _package_root.exists() and str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

_module = importlib.import_module("dax_momentum.pipeline")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
