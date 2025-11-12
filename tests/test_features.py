from __future__ import annotations

import pandas as pd
import pytest

from engineering import add_momentum, add_returns, build_features, make_target


def sample_df() -> pd.DataFrame:
    data = {
        "ric": ["AAA"] * 6,
        "ts": pd.date_range("2024-01-01", periods=6, freq="30T", tz="UTC"),
        "close": [100, 101, 102, 103, 104, 105],
        "open": [99, 100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105, 106],
        "low": [98, 99, 100, 101, 102, 103],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500],
    }
    return pd.DataFrame(data)


def test_add_returns():
    df = add_returns(sample_df())
    assert df.loc[1, "ret_1"] == pytest.approx(0.01, rel=1e-4)
    assert pd.notna(df.loc[2, "log_ret_1"])


def test_add_momentum():
    df = add_momentum(sample_df(), window=5)
    assert df.loc[5, "momentum_5"] == pytest.approx(5.0)


def test_make_target():
    df = add_returns(sample_df())
    df = make_target(df, target_col="ret_1")
    assert df.loc[0, "y_next"] == pytest.approx(df.loc[1, "ret_1"])


def test_build_features():
    df = build_features(sample_df(), {"features": ["ret_1", "momentum_5"], "target": "y_next"})
    assert {"ret_1", "momentum_5", "y_next"}.issubset(df.columns)
    assert df.isna().sum().sum() == 0
