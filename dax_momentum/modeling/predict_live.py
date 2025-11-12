"""Live prediction stub that mimics rolling inference every 30 minutes."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from loguru import logger

from features.engineering import build_features
from features.scaler import load_scaler, transform
from features.sequencing import grouped_sequences
from modeling.lstm import LSTMRegressor
from modeling.utils import batch_signals


@dataclass
class RollingWindow:
    maxlen: int
    buffer: Deque[pd.Series] = field(default_factory=deque)

    def append(self, bar: pd.Series) -> None:
        self.buffer.append(bar)
        if len(self.buffer) > self.maxlen:
            self.buffer.popleft()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.buffer))


class LivePredictor:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        model_kwargs: Dict[str, int],
        time_steps: int,
        tickers: Iterable[str],
        max_history_bars: int,
        device: torch.device,
    ) -> None:
        self.scaler = load_scaler(scaler_path)
        inferred_features = model_kwargs.get("in_features") or len(self.scaler.mean_)
        model_kwargs = {**model_kwargs, "in_features": inferred_features}
        self.model = LSTMRegressor(**model_kwargs)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.time_steps = time_steps
        self.device = device
        self.windows: Dict[str, RollingWindow] = {
            ric: RollingWindow(maxlen=max_history_bars) for ric in tickers
        }

    def update(self, bars: pd.DataFrame, feature_config: Dict[str, Iterable[str]]) -> pd.DataFrame:
        for _, row in bars.iterrows():
            self.windows[row["ric"]].append(row)

        frames = []
        for ric, window in self.windows.items():
            frame = window.to_frame()
            if frame.empty or len(frame) < self.time_steps:
                continue
            frame = build_features(frame, feature_config)
            frame["ric"] = ric
            frames.append(frame)

        if not frames:
            logger.warning("Insufficient data for live prediction")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["ric", "ts"])
        feature_cols = [col for col in combined.columns if col not in {"ric", "ts", "y_next"}]
        X, _ = grouped_sequences(combined, feature_cols, "y_next", self.time_steps)
        X_scaled = transform(X, self.scaler)
        tensor = torch.from_numpy(X_scaled).float().to(self.device)
        with torch.no_grad():
            preds = self.model(tensor).cpu().numpy()[:, 0]
        signals = batch_signals(preds)
        result = combined.iloc[self.time_steps - 1 : self.time_steps - 1 + len(preds)][
            ["ric", "ts", "y_next"]
        ].copy()
        result["prediction"] = preds
        result["signal"] = signals.values
        return result


def fetch_latest_bar_stub(tickers: Iterable[str], interval_minutes: int) -> pd.DataFrame:
    now = pd.Timestamp.utcnow().floor(f"{interval_minutes}T")
    data = []
    rng = np.random.default_rng(seed=int(now.timestamp()) % 10_000)
    for ric in tickers:
        price = 100 + rng.normal(0, 1)
        data.append(
            {
                "ric": ric,
                "ts": now,
                "open": price,
                "high": price + rng.uniform(0, 1),
                "low": price - rng.uniform(0, 1),
                "close": price + rng.normal(0, 0.5),
                "volume": rng.integers(1000, 5000),
            }
        )
    return pd.DataFrame(data)
