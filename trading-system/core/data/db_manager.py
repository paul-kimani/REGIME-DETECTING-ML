"""DatabaseManager — SQLAlchemy/psycopg2 wrapper for all PostgreSQL read/write operations."""

import os
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from core.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages all PostgreSQL interactions for the trading system.

    Uses SQLAlchemy 2.x core (not ORM) with a connection pool backed by psycopg2.
    All credentials are read from environment variables at construction time.
    """

    def __init__(self) -> None:
        """Initialise the manager by reading connection parameters from the environment.

        Environment variables consumed:
            DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        """
        self._host: str = os.environ["DB_HOST"]
        self._port: str = os.environ.get("DB_PORT", "5432")
        self._name: str = os.environ["DB_NAME"]
        self._user: str = os.environ["DB_USER"]
        self._password: str = os.environ["DB_PASSWORD"]
        self._engine = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create the SQLAlchemy engine and verify the connection is reachable.

        Raises:
            SQLAlchemyError: if the database cannot be reached.
        """
        url = (
            f"postgresql+psycopg2://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._name}"
        )
        try:
            self._engine = create_engine(
                url,
                pool_size=5,
                max_overflow=10,
                future=True,
            )
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DatabaseManager: connected to %s/%s", self._host, self._name)
        except SQLAlchemyError as exc:
            logger.error("DatabaseManager.connect failed: %s", exc)
            raise

    def disconnect(self) -> None:
        """Dispose the engine and release all pooled connections.

        Safe to call even if connect() was never called.
        """
        if self._engine is not None:
            try:
                self._engine.dispose()
                logger.info("DatabaseManager: engine disposed")
            except SQLAlchemyError as exc:
                logger.error("DatabaseManager.disconnect failed: %s", exc)
                raise
            finally:
                self._engine = None

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def insert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Bulk-insert OHLCV rows from *df* into the ``candles`` table.

        Duplicate rows (same symbol + timeframe + timestamp) are silently
        skipped via ``ON CONFLICT DO NOTHING``.

        Args:
            symbol:    Instrument identifier, e.g. ``"EURUSD"``.
            timeframe: Candle interval string, e.g. ``"M15"``.
            df:        DataFrame with columns ``timestamp``, ``open``, ``high``,
                       ``low``, ``close``, ``volume``, and optionally ``spread``.
                       The ``timestamp`` column may be timezone-aware or naive;
                       it is normalised to UTC before insertion.

        Returns:
            Number of rows actually inserted (duplicates excluded).

        Raises:
            SQLAlchemyError: on any database error.
        """
        if df.empty:
            return 0

        df = df.copy()

        # --- normalise timestamp to UTC-aware datetime objects -----------
        ts = df["timestamp"]
        if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
            # Already tz-aware — convert to UTC.
            df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # Naive — assume UTC, strip any remaining tzinfo artefacts.
            df["timestamp"] = pd.to_datetime(ts, utc=False).dt.tz_localize(None)

        has_spread = "spread" in df.columns

        sql = text(
            """
            INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume, spread)
            VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :spread)
            ON CONFLICT DO NOTHING
            """
        )

        records = []
        for row in df.itertuples(index=False):
            records.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": row.timestamp,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "spread": float(row.spread) if has_spread else None,
                }
            )

        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, records)
                conn.commit()
            inserted = result.rowcount if result.rowcount >= 0 else len(records)
            logger.debug(
                "insert_candles: %s/%s — %d rows inserted", symbol, timeframe, inserted
            )
            return inserted
        except SQLAlchemyError as exc:
            logger.error("insert_candles failed (%s/%s): %s", symbol, timeframe, exc)
            raise

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Fetch candles for *symbol*/*timeframe* within [*start*, *end*].

        Args:
            symbol:    Instrument identifier.
            timeframe: Candle interval string.
            start:     Inclusive lower bound on ``timestamp``.
            end:       Inclusive upper bound on ``timestamp``.
            limit:     Maximum number of rows returned (default 10 000).

        Returns:
            DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, ordered by ``timestamp`` ascending.

        Raises:
            SQLAlchemyError: on any database error.
        """
        sql = text(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol    = :symbol
              AND timeframe = :timeframe
              AND timestamp >= :start
              AND timestamp <= :end
            ORDER BY timestamp ASC
            LIMIT :limit
            """
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    sql,
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": limit,
                    },
                )
                rows = result.fetchall()
                columns = list(result.keys())
            df = pd.DataFrame(rows, columns=columns)
            logger.debug(
                "get_candles: %s/%s — %d rows fetched", symbol, timeframe, len(df)
            )
            return df
        except SQLAlchemyError as exc:
            logger.error("get_candles failed (%s/%s): %s", symbol, timeframe, exc)
            raise

    def get_latest_candle_time(
        self, symbol: str, timeframe: str
    ) -> Optional[datetime]:
        """Return the most recent candle timestamp for *symbol*/*timeframe*.

        Args:
            symbol:    Instrument identifier.
            timeframe: Candle interval string.

        Returns:
            A ``datetime`` for the newest candle, or ``None`` if no rows exist.

        Raises:
            SQLAlchemyError: on any database error.
        """
        sql = text(
            """
            SELECT MAX(timestamp)
            FROM candles
            WHERE symbol    = :symbol
              AND timeframe = :timeframe
            """
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe})
                value = result.scalar()
            return value  # may be None
        except SQLAlchemyError as exc:
            logger.error(
                "get_latest_candle_time failed (%s/%s): %s", symbol, timeframe, exc
            )
            raise

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(self, trade_record: dict) -> int:
        """Insert a single trade record into the ``trades`` table.

        Args:
            trade_record: Mapping of column names to values.  All keys must
                          correspond to actual columns in the ``trades`` table.

        Returns:
            The auto-generated primary-key ``id`` of the inserted row.

        Raises:
            SQLAlchemyError: on any database error.
        """
        columns = ", ".join(trade_record.keys())
        placeholders = ", ".join(f":{k}" for k in trade_record.keys())
        sql = text(
            f"INSERT INTO trades ({columns}) VALUES ({placeholders}) RETURNING id"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, trade_record)
                inserted_id: int = result.scalar_one()
                conn.commit()
            logger.debug("insert_trade: inserted id=%d", inserted_id)
            return inserted_id
        except SQLAlchemyError as exc:
            logger.error("insert_trade failed: %s", exc)
            raise

    def update_trade(self, ticket: int, updates: dict) -> None:
        """Update one or more fields on the trade identified by *ticket*.

        Args:
            ticket:  The broker ticket number that uniquely identifies the row.
            updates: Mapping of column names to new values.

        Raises:
            ValueError:       if *updates* is empty.
            SQLAlchemyError:  on any database error.
        """
        if not updates:
            raise ValueError("update_trade: 'updates' dict must not be empty")

        set_clause = ", ".join(f"{col} = :{col}" for col in updates.keys())
        sql = text(f"UPDATE trades SET {set_clause} WHERE ticket = :_ticket")
        params = {**updates, "_ticket": ticket}
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
            logger.debug("update_trade: ticket=%d updated fields=%s", ticket, list(updates.keys()))
        except SQLAlchemyError as exc:
            logger.error("update_trade failed (ticket=%d): %s", ticket, exc)
            raise

    # ------------------------------------------------------------------
    # Position events
    # ------------------------------------------------------------------

    def insert_position_event(self, event: dict) -> None:
        """Insert a position lifecycle event into ``position_events``.

        Args:
            event: Mapping of column names to values.

        Raises:
            SQLAlchemyError: on any database error.
        """
        columns = ", ".join(event.keys())
        placeholders = ", ".join(f":{k}" for k in event.keys())
        sql = text(
            f"INSERT INTO position_events ({columns}) VALUES ({placeholders})"
        )
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, event)
                conn.commit()
            logger.debug("insert_position_event: inserted event type=%s", event.get("event_type"))
        except SQLAlchemyError as exc:
            logger.error("insert_position_event failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(self, signal: dict) -> int:
        """Insert a signal record into the ``signals`` table.

        Args:
            signal: Mapping of column names to values.

        Returns:
            The auto-generated primary-key ``id`` of the inserted row.

        Raises:
            SQLAlchemyError: on any database error.
        """
        columns = ", ".join(signal.keys())
        placeholders = ", ".join(f":{k}" for k in signal.keys())
        sql = text(
            f"INSERT INTO signals ({columns}) VALUES ({placeholders}) RETURNING id"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, signal)
                inserted_id: int = result.scalar_one()
                conn.commit()
            logger.debug("insert_signal: inserted id=%d", inserted_id)
            return inserted_id
        except SQLAlchemyError as exc:
            logger.error("insert_signal failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Regime states
    # ------------------------------------------------------------------

    def insert_regime_state(self, regime: dict) -> None:
        """Upsert a regime state record into ``regime_states``.

        The upsert key is ``(symbol, timeframe, timestamp)``.  If a matching
        row already exists its non-key columns are updated in place.

        Args:
            regime: Mapping of column names to values.  Must include
                    ``symbol``, ``timeframe``, and ``timestamp``.

        Raises:
            SQLAlchemyError: on any database error.
        """
        columns = ", ".join(regime.keys())
        placeholders = ", ".join(f":{k}" for k in regime.keys())
        non_key_cols = [
            k for k in regime.keys() if k not in ("symbol", "timeframe", "timestamp")
        ]
        if non_key_cols:
            update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in non_key_cols)
            conflict_action = f"DO UPDATE SET {update_clause}"
        else:
            conflict_action = "DO NOTHING"

        sql = text(
            f"""
            INSERT INTO regime_states ({columns})
            VALUES ({placeholders})
            ON CONFLICT (symbol, timeframe, timestamp) {conflict_action}
            """
        )
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, regime)
                conn.commit()
            logger.debug(
                "insert_regime_state: upserted symbol=%s timeframe=%s",
                regime.get("symbol"),
                regime.get("timeframe"),
            )
        except SQLAlchemyError as exc:
            logger.error("insert_regime_state failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Generic bulk insert
    # ------------------------------------------------------------------

    def bulk_insert(self, table: str, records: list[dict]) -> int:
        """Generic bulk insert into *table* with duplicate-row suppression.

        Uses ``ON CONFLICT DO NOTHING`` so rows that violate a unique
        constraint are skipped rather than raising an error.

        Args:
            table:   Name of the target table (not user-supplied at runtime;
                     assumed to be a trusted, hard-coded table name).
            records: List of dicts where every dict has the same keys.

        Returns:
            Number of rows actually inserted.

        Raises:
            ValueError:      if *records* is empty.
            SQLAlchemyError: on any database error.
        """
        if not records:
            raise ValueError("bulk_insert: 'records' list must not be empty")

        columns = ", ".join(records[0].keys())
        placeholders = ", ".join(f":{k}" for k in records[0].keys())
        sql = text(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, records)
                conn.commit()
            inserted = result.rowcount if result.rowcount >= 0 else len(records)
            logger.debug("bulk_insert: table=%s inserted=%d", table, inserted)
            return inserted
        except SQLAlchemyError as exc:
            logger.error("bulk_insert failed (table=%s): %s", table, exc)
            raise

    # ------------------------------------------------------------------
    # System events
    # ------------------------------------------------------------------

    def log_system_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        symbol: str = None,
        details: dict = None,
    ) -> None:
        """Insert a diagnostic record into the ``system_events`` table.

        Args:
            event_type: Short category label, e.g. ``"CONNECTION_ERROR"``.
            severity:   Severity level string, e.g. ``"INFO"``, ``"ERROR"``.
            message:    Human-readable description of the event.
            symbol:     Optional instrument symbol related to the event.
            details:    Optional free-form mapping serialised to JSON.

        Raises:
            SQLAlchemyError: on any database error.
        """
        sql = text(
            """
            INSERT INTO system_events (event_type, severity, message, symbol, details, created_at)
            VALUES (:event_type, :severity, :message, :symbol, :details, :created_at)
            """
        )
        params = {
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "symbol": symbol,
            "details": json.dumps(details) if details is not None else None,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
            logger.debug(
                "log_system_event: type=%s severity=%s", event_type, severity
            )
        except SQLAlchemyError as exc:
            logger.error("log_system_event failed: %s", exc)
            raise
