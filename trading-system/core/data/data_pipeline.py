"""DataPipeline — historical fetch, live updates, multi-timeframe alignment, and validation."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import pandas as pd

from core.utils.logger import get_logger
from core.utils.helpers import atr
from core.execution.mt5_connector import MT5Connector as MT5Client, MT5ConnectionError
from core.data.db_manager import DatabaseManager

logger = get_logger(__name__)

# Timeframe durations in minutes
_TF_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

_DEFAULT_TIMEFRAMES = ["M15", "H1", "H4"]
_PAGE_SIZE = 2000          # bars per copy_rates_range page
_POLL_INTERVAL = 15        # seconds between polling ticks in on_new_candle


class DataPipeline:
    """Orchestrates data flow between the MT5 bridge, PostgreSQL, and downstream consumers.

    Args:
        mt5_client:  Configured :class:`~core.execution.mt5_client.MT5Client` instance.
        db_manager:  Connected :class:`~core.data.db_manager.DatabaseManager` instance.
    """

    def __init__(self, mt5_client: MT5Client, db_manager: DatabaseManager) -> None:
        self._mt5 = mt5_client
        self._db = db_manager
        self._stop_polling = False
        logger.debug("DataPipeline initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        bars: int = 5000,
        store_to_db: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV from the MT5 bridge and optionally persist to PostgreSQL.

        Fetches in pages of up to 2 000 bars so that large pulls do not time out.
        Each page is validated with :meth:`validate_data` before accumulation.
        Duplicate timestamps are dropped, and the final DataFrame is sorted
        ascending by timestamp.

        Args:
            symbol:      Instrument symbol, e.g. ``"EURUSD"``.
            timeframe:   Timeframe string, e.g. ``"M15"``.
            start_date:  Inclusive start of the fetch window (UTC).
            bars:        Approximate maximum number of bars to retrieve
                         (default 5 000).  The actual number returned may be
                         slightly less due to market gaps.
            store_to_db: When ``True`` (default) validated rows are inserted
                         into PostgreSQL via
                         :meth:`~core.data.db_manager.DatabaseManager.insert_candles`.

        Returns:
            Clean, ascending-sorted DataFrame with columns:
            ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``, ``spread``.

        Raises:
            ValueError: If the fetched data is empty after all pages.
        """
        logger.info(
            "fetch_historical: %s/%s from %s, target %d bars",
            symbol,
            timeframe,
            start_date.isoformat(),
            bars,
        )

        tf_minutes = _TF_MINUTES.get(timeframe, 15)
        all_frames: list[pd.DataFrame] = []
        page_start = start_date
        bars_remaining = bars

        while bars_remaining > 0:
            page_bars = min(_PAGE_SIZE, bars_remaining)
            page_end = page_start + timedelta(minutes=tf_minutes * page_bars)

            logger.debug(
                "fetch_historical page: %s — %s to %s (%d bars)",
                symbol,
                page_start.isoformat(),
                page_end.isoformat(),
                page_bars,
            )

            rates = self._mt5.copy_rates_range(symbol, timeframe, page_start, page_end)
            if not rates:
                logger.debug(
                    "fetch_historical: no data returned for page %s — %s, stopping",
                    page_start.isoformat(),
                    page_end.isoformat(),
                )
                break

            page_df = self._convert_mt5_rates(rates)
            is_valid, issues = self.validate_data(page_df)
            if not is_valid:
                logger.warning(
                    "fetch_historical: validation failed for %s/%s page starting %s: %s",
                    symbol,
                    timeframe,
                    page_start.isoformat(),
                    issues,
                )

            all_frames.append(page_df)
            bars_remaining -= len(page_df)

            # Advance the window past the last timestamp we received
            last_ts = page_df["timestamp"].max()
            next_start = last_ts + timedelta(minutes=tf_minutes)
            if next_start <= page_start:
                # Guard against infinite loop if timestamps don't advance
                break
            page_start = next_start

            # If the page returned fewer bars than requested we are at the
            # end of available data
            if len(page_df) < page_bars:
                break

        if not all_frames:
            logger.warning(
                "fetch_historical: no data retrieved for %s/%s", symbol, timeframe
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "spread"]
            )

        combined = pd.concat(all_frames, ignore_index=True)
        combined = (
            combined
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        logger.info(
            "fetch_historical: %s/%s — %d bars assembled",
            symbol,
            timeframe,
            len(combined),
        )

        if store_to_db:
            try:
                inserted = self._db.insert_candles(symbol, timeframe, combined)
                logger.info(
                    "fetch_historical: stored %d rows to DB for %s/%s",
                    inserted,
                    symbol,
                    timeframe,
                )
            except Exception as exc:
                logger.error(
                    "fetch_historical: DB insert failed for %s/%s: %s",
                    symbol,
                    timeframe,
                    exc,
                )

        return combined

    def fetch_latest(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch the most recent *count* candles and return a clean DataFrame.

        Calls :meth:`~core.execution.mt5_client.MT5Client.copy_rates_from_pos`
        with ``start_pos=0`` (the current, most-recent bar) and forward-fills
        any small gaps of fewer than 5 consecutive missing bars.

        Args:
            symbol:    Instrument symbol, e.g. ``"EURUSD"``.
            timeframe: Timeframe string, e.g. ``"M15"``.
            count:     Number of most-recent bars to fetch (default 500).

        Returns:
            Clean, ascending-sorted DataFrame with columns:
            ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``, ``spread``.
        """
        logger.debug("fetch_latest: %s/%s count=%d", symbol, timeframe, count)

        rates = self._mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if not rates:
            logger.warning("fetch_latest: no data for %s/%s", symbol, timeframe)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "spread"]
            )

        df = self._convert_mt5_rates(rates)

        # Build a complete, evenly-spaced index and forward-fill small gaps
        tf_minutes = _TF_MINUTES.get(timeframe, 15)
        df = df.set_index("timestamp").sort_index()

        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=f"{tf_minutes}min",
            tz=df.index.tz,
        )
        df = df.reindex(full_index)

        # Forward-fill only gaps smaller than 5 consecutive bars
        # pandas ffill with limit fills at most `limit` consecutive NaNs
        df = df.ffill(limit=4)

        # Drop rows still NaN (gaps >= 5 bars) then restore timestamp column
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.reset_index().rename(columns={"index": "timestamp"})
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.debug("fetch_latest: %s/%s — %d bars returned", symbol, timeframe, len(df))
        return df

    def get_multi_timeframe(
        self, symbol: str, timeframes: Optional[list[str]] = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch the latest data for all requested timeframes.

        Args:
            symbol:     Instrument symbol, e.g. ``"EURUSD"``.
            timeframes: List of timeframe strings to fetch.
                        Defaults to ``['M15', 'H1', 'H4']``.

        Returns:
            Dict mapping each timeframe string to its DataFrame, e.g.
            ``{'M15': df_m15, 'H1': df_h1, 'H4': df_h4}``.
        """
        if timeframes is None:
            timeframes = _DEFAULT_TIMEFRAMES

        logger.debug(
            "get_multi_timeframe: %s timeframes=%s", symbol, timeframes
        )

        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                result[tf] = self.fetch_latest(symbol, tf)
            except Exception as exc:
                logger.error(
                    "get_multi_timeframe: failed to fetch %s/%s: %s", symbol, tf, exc
                )
                result[tf] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "spread"]
                )

        return result

    def align_timeframes(self, data_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Align H1 and H4 data to the M15 anchor index via forward-fill.

        The M15 DataFrame defines the master timestamp index.  Higher-timeframe
        DataFrames are reindexed to that same index with forward-fill so that
        every M15 bar carries the value of its enclosing H1 / H4 bar.

        Non-M15 column names are prefixed with the lowercase timeframe label,
        e.g. ``h1_open``, ``h1_close``, ``h4_open``, ``h4_close``.

        Args:
            data_dict: Dict returned by :meth:`get_multi_timeframe`.
                       Must contain an ``'M15'`` key.

        Returns:
            New dict where all DataFrames share the M15 timestamp index.
            The ``'M15'`` entry is unchanged; other entries have prefixed
            column names and the M15 index.

        Raises:
            KeyError: If ``'M15'`` is not present in *data_dict*.
        """
        if "M15" not in data_dict:
            raise KeyError("align_timeframes requires 'M15' data as the anchor")

        m15_df = data_dict["M15"].copy()

        # Ensure M15 uses timestamp as index
        if "timestamp" in m15_df.columns:
            m15_df = m15_df.set_index("timestamp")

        aligned: dict[str, pd.DataFrame] = {"M15": m15_df.reset_index()}

        ohlcv_cols = ["open", "high", "low", "close", "volume", "spread"]

        for tf, df in data_dict.items():
            if tf == "M15":
                continue

            prefix = tf.lower()
            tf_df = df.copy()

            if "timestamp" in tf_df.columns:
                tf_df = tf_df.set_index("timestamp")

            # Keep only the columns that exist in this DataFrame
            cols_to_use = [c for c in ohlcv_cols if c in tf_df.columns]
            tf_df = tf_df[cols_to_use]

            # Rename columns with prefix
            tf_df = tf_df.rename(columns={c: f"{prefix}_{c}" for c in cols_to_use})

            # Reindex to M15 index and forward-fill
            tf_df = tf_df.reindex(m15_df.index)
            tf_df = tf_df.ffill()

            # Attach M15 timestamp as a column for consistency
            tf_df = tf_df.reset_index().rename(columns={"index": "timestamp"})
            aligned[tf] = tf_df

        logger.debug(
            "align_timeframes: aligned %d timeframes to M15 index (%d bars)",
            len(aligned),
            len(m15_df),
        )
        return aligned

    def validate_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate an OHLCV DataFrame for common data-quality issues.

        Performs the following checks in order:

        1. **No duplicate timestamps** — critical; fails validation.
        2. **No NaN in OHLC columns** — critical; fails validation.
        3. **OHLC logic** — ``high >= max(open, close)`` and
           ``low <= min(open, close)``; critical.
        4. **Extreme outliers** — consecutive close change > 10 × ATR(14);
           critical.
        5. **Zero-volume bars** — non-critical warning only.
        6. **Timestamp gaps** — any gap > 3 × expected bar duration is flagged;
           non-critical warning.

        The expected bar duration for gap detection is inferred from the
        median inter-bar timedelta so no explicit timeframe argument is needed.

        Args:
            df: DataFrame with at minimum ``timestamp``, ``open``, ``high``,
                ``low``, ``close``, ``volume`` columns.

        Returns:
            ``(True, [])`` when all *critical* checks pass.
            ``(True, [warning, …])`` when only non-critical issues are found.
            ``(False, [error, …])`` when any critical check fails (warnings may
            also appear in the list).
        """
        issues: list[str] = []
        is_valid = True

        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        # ── 1. Duplicate timestamps ──────────────────────────────────────────
        n_dupes = df["timestamp"].duplicated().sum()
        if n_dupes > 0:
            issues.append(f"Duplicate timestamps: {n_dupes} duplicates found")
            is_valid = False

        # ── 2. NaN in OHLC columns ───────────────────────────────────────────
        ohlc_cols = ["open", "high", "low", "close"]
        for col in ohlc_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: '{col}'")
                is_valid = False
                continue
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                issues.append(f"NaN values in '{col}': {n_nan} rows")
                is_valid = False

        # Skip remaining checks if required columns are absent
        missing_critical = [c for c in ohlc_cols if c not in df.columns]
        if missing_critical:
            return False, issues

        # ── 3. OHLC logic ────────────────────────────────────────────────────
        high_violation = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        if high_violation > 0:
            issues.append(
                f"OHLC logic violation: high < max(open, close) in {high_violation} rows"
            )
            is_valid = False

        low_violation = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if low_violation > 0:
            issues.append(
                f"OHLC logic violation: low > min(open, close) in {low_violation} rows"
            )
            is_valid = False

        # ── 4. Extreme outliers via ATR(14) ──────────────────────────────────
        if len(df) >= 15:
            atr_series = atr(df["high"], df["low"], df["close"], period=14)
            close_change = df["close"].diff().abs()
            threshold = 10 * atr_series
            outlier_mask = close_change > threshold
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                issues.append(
                    f"Extreme outliers: {n_outliers} bars with consecutive close "
                    f"change > 10 * ATR(14)"
                )
                is_valid = False

        # ── 5. Zero-volume bars (non-critical) ───────────────────────────────
        if "volume" in df.columns:
            n_zero_vol = (df["volume"] == 0).sum()
            if n_zero_vol > 0:
                issues.append(
                    f"Warning: {n_zero_vol} bars with zero volume (non-critical)"
                )
                # does not flip is_valid to False

        # ── 6. Timestamp gaps (non-critical) ─────────────────────────────────
        if len(df) >= 2:
            ts_sorted = pd.to_datetime(df["timestamp"]).sort_values()
            deltas = ts_sorted.diff().dropna()
            if not deltas.empty:
                median_delta = deltas.median()
                gap_threshold = 3 * median_delta
                large_gaps = (deltas > gap_threshold).sum()
                if large_gaps > 0:
                    issues.append(
                        f"Warning: {large_gaps} timestamp gaps > 3x expected bar "
                        f"duration ({median_delta}) detected (non-critical)"
                    )
                    # does not flip is_valid to False

        return is_valid, issues

    def on_new_candle(
        self, symbol: str, timeframe: str, callback: Callable[[str, str, dict], None]
    ) -> None:
        """Start a background polling thread that fires *callback* on each new candle.

        Polls the MT5 bridge every 15 seconds.  When the most-recent bar's
        timestamp differs from the last-seen timestamp a new candle is deemed
        closed and *callback* is invoked with::

            callback(symbol, timeframe, latest_candle_dict)

        where *latest_candle_dict* is the row converted to a plain Python dict.

        The thread is daemonised so it exits automatically when the main
        process terminates.  Set :attr:`_stop_polling` to ``True`` to stop
        the loop gracefully from another thread.

        Args:
            symbol:    Instrument symbol to watch.
            timeframe: Timeframe to watch.
            callback:  Callable invoked as ``callback(symbol, timeframe, candle_dict)``
                       each time a new candle closes.
        """
        self._stop_polling = False

        def _poll() -> None:
            last_timestamp: Optional[datetime] = None
            logger.info(
                "on_new_candle: polling started for %s/%s (interval=%ds)",
                symbol,
                timeframe,
                _POLL_INTERVAL,
            )
            while not self._stop_polling:
                try:
                    df = self.fetch_latest(symbol, timeframe, count=2)
                    if df.empty:
                        time.sleep(_POLL_INTERVAL)
                        continue

                    latest_row = df.iloc[-1]
                    current_ts = latest_row["timestamp"]

                    if last_timestamp is None:
                        last_timestamp = current_ts
                        logger.debug(
                            "on_new_candle: %s/%s initial timestamp %s",
                            symbol,
                            timeframe,
                            current_ts,
                        )
                    elif current_ts != last_timestamp:
                        last_timestamp = current_ts
                        candle_dict = latest_row.to_dict()
                        logger.info(
                            "on_new_candle: new candle for %s/%s at %s",
                            symbol,
                            timeframe,
                            current_ts,
                        )
                        try:
                            callback(symbol, timeframe, candle_dict)
                        except Exception as cb_exc:
                            logger.error(
                                "on_new_candle: callback raised an exception "
                                "for %s/%s: %s",
                                symbol,
                                timeframe,
                                cb_exc,
                            )

                except Exception as exc:
                    logger.error(
                        "on_new_candle: polling error for %s/%s: %s",
                        symbol,
                        timeframe,
                        exc,
                    )

                time.sleep(_POLL_INTERVAL)

            logger.info(
                "on_new_candle: polling stopped for %s/%s", symbol, timeframe
            )

        thread = threading.Thread(
            target=_poll,
            name=f"candle-poll-{symbol}-{timeframe}",
            daemon=True,
        )
        thread.start()
        logger.debug(
            "on_new_candle: daemon thread started — %s", thread.name
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_mt5_rates(rates: list[dict]) -> pd.DataFrame:
        """Convert the raw MT5 bridge response to a clean OHLCV DataFrame.

        The bridge returns each bar as a dict whose ``time`` field is an ISO
        8601 string (e.g. ``"2024-01-15T08:15:00"``).  This method:

        * Parses the ``time`` field into a UTC-aware :class:`pandas.Timestamp`.
        * Renames ``time`` → ``timestamp`` and ``tick_volume`` → ``volume``
          (when the ``volume`` column is not already present).
        * Ensures the returned DataFrame has the canonical column set:
          ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``,
          ``spread``.

        Args:
            rates: List of bar dicts from the MT5 bridge, each containing at
                   minimum ``time``, ``open``, ``high``, ``low``, ``close``,
                   and ``tick_volume`` keys.  A ``spread`` key is optional.

        Returns:
            DataFrame with columns listed above, index reset to integers.
        """
        if not rates:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "spread"]
            )

        df = pd.DataFrame(rates)

        # Rename time -> timestamp
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})

        # Parse timestamp — ISO string to UTC-aware Timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Normalise volume column name
        if "volume" not in df.columns and "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})

        # Ensure spread column exists (may be absent from some bridge responses)
        if "spread" not in df.columns:
            df["spread"] = float("nan")

        # Select and order the canonical columns; ignore any extras
        canonical = ["timestamp", "open", "high", "low", "close", "volume", "spread"]
        available = [c for c in canonical if c in df.columns]
        df = df[available].reset_index(drop=True)

        return df
