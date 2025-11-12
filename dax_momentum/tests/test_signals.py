from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.utils import batch_signals, signal_summary, to_signal


def test_to_signal():
    assert to_signal(0.5) == "LONG"
    assert to_signal(-0.2) == "SHORT"
    assert to_signal(0.0) == "FLAT"
    assert to_signal(0.01, threshold=0.02) == "FLAT"


def test_batch_signals():
    signals = batch_signals([0.1, -0.3, 0.0])
    assert list(signals) == ["LONG", "SHORT", "FLAT"]


def test_signal_summary():
    signals = pd.Series(["LONG", "SHORT", "FLAT", "LONG"])
    realized = pd.Series([0.1, -0.2, 0.0, 0.3])
    summary, distribution = signal_summary(signals, realized)
    assert 0 <= summary["hit_rate"][0] <= 1
    assert distribution["share"].sum() == pytest.approx(1.0)
