"""Refinitiv Workspace client wrapper.

Provides session management and historical price retrieval with optional
mocked data generation for offline development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:  # pragma: no cover - optional dependency
    import refinitiv.data as rd
except ImportError:  # pragma: no cover - executed in environments without access
    rd = None  # type: ignore


@dataclass
class RefinitivCredentials:
    """Container for Refinitiv Workspace credentials."""

    app_key: Optional[str] = None
    desktop_app_id: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RefinitivCredentials":
        """Load credentials from environment variables."""

        return cls(
            app_key=os.getenv("APP_KEY"),
            desktop_app_id=os.getenv("DESKTOP_APP_ID"),
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
        )


class RefinitivClient:
    """Simple session manager for the Refinitiv Data Library."""

    def __init__(self, credentials: Optional[RefinitivCredentials] = None) -> None:
        self.credentials = credentials or RefinitivCredentials.from_env()
        self._session_open = False

    def open_session(self) -> None:
        """Open a session if the library is available."""

        if rd is None:
            logger.warning(
                "refinitiv.data library not available; falling back to offline mode"
            )
            return

        if self.credentials.app_key is None:
            raise RuntimeError("APP_KEY environment variable not set for Refinitiv")

        if self._session_open:
            return

        logger.info("Opening Refinitiv session")
        rd.open_session(app_key=self.credentials.app_key)  # type: ignore[arg-type]
        self._session_open = True

    def close_session(self) -> None:
        """Close the Refinitiv session."""

        if rd is None or not self._session_open:
            return
        logger.info("Closing Refinitiv session")
        rd.close_session()
        self._session_open = False

    def __enter__(self) -> "RefinitivClient":  # pragma: no cover - thin wrapper
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        self.close_session()

    def fetch_history(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str,
        fields: Optional[Iterable[str]] = None,
        offline_mode: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Parameters
        ----------
        tickers:
            RIC codes to request.
        start, end:
            ISO strings describing the history window.
        interval:
            ISO 8601 interval string, e.g. ``PT30M``.
        fields:
            Additional fields to request. Defaults map to OHLCV.
        offline_mode:
            If ``True`` or the library is not available, a mock dataset is
            generated to keep the pipeline functional.
        """

        resolved_fields = list(fields or [
            "OPEN_PRC",
            "HIGH_1",
            "LOW_1",
            "TRDPRC_1",
            "ACVOL_1",
        ])

        if offline_mode or rd is None:
            logger.info("Generating synthetic history for offline mode")
            return self._generate_mock_history(tickers, start, end, interval)

        if not self._session_open:
            self.open_session()

        try:
            logger.info(
                "Requesting history for %d tickers between %s and %s",
                len(list(tickers)),
                start,
                end,
            )
            data = rd.content.historical_pricing.Definition(  # type: ignore[attr-defined]
                universe=list(tickers),
                fields=resolved_fields,
                interval=interval,
                start=start,
                end=end,
            ).get_data()
        except Exception as exc:  # pragma: no cover - requires live service
            logger.exception("Refinitiv request failed, falling back to offline mode")
            return self._generate_mock_history(tickers, start, end, interval)

        frame = data.df.copy()
        frame = frame.rename(
            columns={
                "Instrument": "ric",
                "Date": "ts",
                "OPEN_PRC": "open",
                "HIGH_1": "high",
                "LOW_1": "low",
                "TRDPRC_1": "close",
                "ACVOL_1": "volume",
            }
        )
        frame = frame[["ric", "ts", "open", "high", "low", "close", "volume"]]
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame.sort_values(["ric", "ts"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    @staticmethod
    def _generate_mock_history(
        tickers: Iterable[str], start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for offline development."""

        tickers = list(tickers)
        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)
        interval_minutes = RefinitivClient._parse_interval_minutes(interval)
        index = pd.date_range(start_dt, end_dt, freq=f"{interval_minutes}T", inclusive="left")

        records: List[pd.DataFrame] = []
        rng = np.random.default_rng(seed=42)
        for ric in tickers:
            prices = np.cumsum(rng.normal(loc=0.02, scale=0.5, size=len(index))) + 100
            highs = prices + rng.uniform(0.0, 1.0, size=len(index))
            lows = prices - rng.uniform(0.0, 1.0, size=len(index))
            opens = prices + rng.normal(0, 0.2, size=len(index))
            volumes = rng.integers(low=1000, high=5000, size=len(index))
            df = pd.DataFrame(
                {
                    "ric": ric,
                    "ts": index,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": prices,
                    "volume": volumes,
                }
            )
            records.append(df)

        result = pd.concat(records, ignore_index=True)
        return result

    @staticmethod
    def _parse_interval_minutes(interval: str) -> int:
        """Parse ISO-8601 duration string (PTxxM) into minutes."""

        if not interval.startswith("PT") or not interval.endswith("M"):
            raise ValueError(f"Unsupported interval format: {interval}")
        minutes = int(interval[2:-1])
        if minutes <= 0:
            raise ValueError("Interval must be positive")
        return minutes


def fetch_history(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str,
    offline_mode: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper for the :class:`RefinitivClient`."""

    client = RefinitivClient()
    with client:
        return client.fetch_history(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            offline_mode=offline_mode,
        )
