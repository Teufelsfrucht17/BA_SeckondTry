from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.sequencing import grouped_sequences, make_sequences


def test_make_sequences_shapes():
    X = np.arange(30).reshape(10, 3)
    y = np.arange(10)
    seq_X, seq_y = make_sequences(X, y, time_steps=4)
    assert seq_X.shape == (7, 4, 3)
    assert seq_y.shape == (7,)


def test_grouped_sequences():
    df = pd.DataFrame(
        {
            "ric": ["A"] * 5 + ["B"] * 5,
            "ts": pd.date_range("2024-01-01", periods=10, freq="30T", tz="UTC"),
            "feat1": np.arange(10),
            "feat2": np.arange(10) * 2,
            "y_next": np.arange(10) * 0.1,
        }
    )
    X, y = grouped_sequences(df, ["feat1", "feat2"], "y_next", time_steps=3)
    assert X.shape[1:] == (3, 2)
    assert y.ndim == 1
