"""Refinitiv Workspace client wrapper.

Provides session management and historical price retrieval with optional
mocked data generation for offline development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, List, Optional
import re

import numpy as np
import pandas as pd
from loguru import logger

try:  # pragma: no cover - optional dependency
    import refinitiv.data as rd
    import lseg.data as ld
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

        if ld is None:
            logger.warning(
                "refinitiv.data library not available; falling back to offline mode"
            )
            return


        if self._session_open:
            return

        logger.info("Opening Refinitiv session")
        ld.open_session()  # type: ignore[arg-type]
        self._session_open = True

    def close_session(self) -> None:
        """Close the Refinitiv session."""

        if ld is None or not self._session_open:
            return
        logger.info("Closing Refinitiv session")
        ld.close_session()
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

        if offline_mode or ld is None:
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
            """ 
            data = ld.content.historical_pricing.events.Definition(  # type: ignore[attr-defined]
                universe=list(tickers),
                fields=resolved_fields,
              #  interval=interval,
                start=start,
                end=end,
            ).get_data()
            """
            data = ld.get_history(
                universe=list(tickers),
                fields=resolved_fields,
                start=start,
                end=end, )
        except Exception as exc:  # pragma: no cover - requires live service
            logger.exception("Refinitiv request failed, falling back to offline mode")
            return self._generate_mock_history(tickers, start, end, interval)

        logger.debug("History response type: %s", type(data))
        frame = self._to_dataframe(data)
        try:
            cols_list = list(frame.columns)
        except Exception:
            cols_list = ["<unavailable>"]
        logger.debug("History DataFrame columns: %s", cols_list)
        frame = self._standardize_history_columns(frame)
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
    def _to_dataframe(data: Any) -> pd.DataFrame:
        """Best-effort conversion of Refinitiv SDK responses into DataFrames."""

        # Common response shapes observed in the SDK
        candidates = [
            data,
            getattr(data, "df", None),
            getattr(data, "data", None),
            getattr(getattr(data, "data", None), "df", None),
        ]

        for candidate in candidates:
            if isinstance(candidate, pd.DataFrame):
                return candidate

        if isinstance(data, dict):
            for key in ("data", "table", "rows"):
                candidate = data.get(key)
                if isinstance(candidate, pd.DataFrame):
                    return candidate
                if isinstance(candidate, list):
                    try:
                        return pd.DataFrame(candidate)
                    except ValueError:
                        continue

        try:
            return pd.DataFrame(data)
        except ValueError as exc:  # pragma: no cover - depends on live SDK
            raise ValueError(
                "Refinitiv response cannot be coerced to DataFrame; "
                f"received type {type(data)!r}"
            ) from exc

    @staticmethod
    def _standardize_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize vendor column names to [ric, ts, open, high, low, close, volume].

        Handles both long/table formats (columns like Instrument/Date/OPEN_PRC)
        and wide MultiIndex formats (columns like (Instrument, Field)).
        Missing close/volume are backfilled conservatively if needed.
        """

        def canonical(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

        alias_map = {
            "ric": ["ric", "instrument", "ticker", "symbol", "riccode"],
            "ts": [
                "ts",
                "date",
                "datetime",
                "eventdate",
                "event_time",
                "time",
                "timestamp",
                "date_gmt",
            ],
            "open": ["open", "open_prc", "openprice", "open_bid", "open_1", "openprc"],
            "high": ["high", "high_1", "high_prc", "highprice", "highprc"],
            "low": ["low", "low_1", "low_prc", "lowprice", "lowprc"],
            "close": ["close", "trdprc_1", "last", "close_prc", "closeprice", "lastprice", "last_prc"],
            "volume": ["volume", "acvol_1", "turnover", "totalvolume", "totvolume"],
        }

        field_aliases = set(
            canonical(a)
            for k in ("open", "high", "low", "close", "volume")
            for a in alias_map[k]
        )

        df = frame.copy()

        # Case A: columns are MultiIndex with instruments x fields
        if isinstance(df.columns, pd.MultiIndex):
            mi = df.columns
            nlevels = mi.nlevels

            # Identify which level holds the field names by matching aliases
            match_counts = []
            for i in range(nlevels):
                vals = [canonical(v) for v in mi.get_level_values(i).unique().tolist()]
                match_counts.append(sum(1 for v in vals if v in field_aliases))

            field_level = None
            if max(match_counts) > 0:
                field_level = match_counts.index(max(match_counts))

            if field_level is None or nlevels < 2:
                # Fallback: flatten and continue in simple path
                df.columns = ["_".join(str(p) for p in col if p not in (None, "")) for col in mi]
            else:
                instrument_level = 0 if field_level == 1 else 1
                names = [None] * nlevels
                names[instrument_level] = "instrument"
                names[field_level] = "field"
                df.columns = df.columns.set_names(names)

                # Bring instrument into rows; fields stay as columns
                long = df.stack(level="instrument").reset_index()
                long = long.rename(columns={"instrument": "ric"})

                # Try to detect and rename a timestamp column
                ts_aliases = set(canonical(x) for x in alias_map["ts"])
                ts_col = None
                for c in long.columns:
                    if canonical(c) in ts_aliases:
                        ts_col = c
                        break
                if ts_col is None:
                    # Pandas creates default names like 'level_0' for unnamed index
                    for c in long.columns:
                        if str(c).startswith("level_"):
                            ts_col = c
                            break
                if ts_col is None and len(long.columns) >= 2:
                    # heuristic: first non-'ric' column
                    for c in long.columns:
                        if c != "ric" and c not in df.columns.names:
                            ts_col = c
                            break

                if ts_col is None:
                    raise KeyError("Could not identify timestamp column in history response after unpivoting.")

                long = long.rename(columns={ts_col: "ts"})

                # Now rename field columns to canonical names
                available = {canonical(c): c for c in long.columns}
                rename_fields: dict[str, str] = {}
                for target, aliases in alias_map.items():
                    if target in ("ric", "ts"):
                        continue
                    for alias in aliases:
                        cand = available.get(canonical(alias))
                        if cand:
                            rename_fields[cand] = target
                            break

                long = long.rename(columns=rename_fields)

                # Ensure required columns exist; if some fields missing, create them as NaN
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in long.columns:
                        long[col] = pd.NA

                ordered = ["ric", "ts", "open", "high", "low", "close", "volume"]
                return long[ordered]

        # Case B: simple wide/long table with vendor column names
        df.columns = [str(c) for c in df.columns]
        available = {canonical(c): c for c in df.columns}

        rename_map: dict[str, str] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                cand = available.get(canonical(alias))
                if cand:
                    rename_map[cand] = target
                    break

        df = df.rename(columns=rename_map)

        # If ric or ts were not present as columns, try promoting index
        if "ts" not in df.columns and df.index is not None:
            # If named index matches ts alias or is unnamed, bring it in
            name = df.index.name or "ts"
            df = df.reset_index().rename(columns={name: "ts"})
        if "ric" not in df.columns and "Instrument" in frame.columns:
            df = df.rename(columns={"Instrument": "ric"})

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = pd.NA

        missing_core = [c for c in ("ric", "ts") if c not in df.columns]
        if missing_core:
            cols_str = ", ".join(map(str, frame.columns))
            raise KeyError(
                f"Missing required columns {missing_core} in Refinitiv response. Available columns: {cols_str}"
            )

        ordered = ["ric", "ts", "open", "high", "low", "close", "volume"]
        return df[ordered]

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
