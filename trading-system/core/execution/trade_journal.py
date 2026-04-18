"""TradeJournal — logs every trade lifecycle event to PostgreSQL and Redis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import redis
from redis import RedisError

from core.utils.logger import get_logger

logger = get_logger(__name__)

# Contract size used for approximate PnL calculation when not otherwise known.
# For FX majors a standard lot is 100,000 units.
_DEFAULT_CONTRACT_SIZE = 100_000.0

# Redis key prefix and TTLs
_KEY_OPEN_POSITIONS = "trading:open_positions"
_KEY_DAILY_PNL = "trading:daily_pnl"
_KEY_CIRCUIT_BREAKER = "trading:circuit_breaker"
_KEY_LAST_SIGNAL_PREFIX = "trading:last_signal"

_TTL_DEFAULT = 300      # 5 minutes
_TTL_DAILY_PNL = 86400  # 24 hours
_TTL_CB = 3600          # 1 hour
_TTL_LAST_SIGNAL = 3600 # 1 hour


class TradeJournal:
    """Logs every trade lifecycle event to PostgreSQL and Redis.

    PostgreSQL: permanent record via DatabaseManager.
    Redis: live dashboard state (current positions, today PnL, CB status).

    Redis is optional — all Redis errors are swallowed and logged at WARNING
    level so that a Redis outage never disrupts trade execution.
    """

    def __init__(
        self,
        db_manager=None,
        redis_url: Optional[str] = None,
    ) -> None:
        """Connect to Redis. Store db_manager reference.

        Args:
            db_manager: Optional :class:`~core.data.db_manager.DatabaseManager`
                        for PostgreSQL persistence.
            redis_url:  Redis connection URL. Falls back to the ``REDIS_URL``
                        environment variable, then ``redis://localhost:6379``.
        """
        self._db_manager = db_manager

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")

        try:
            self._redis: Optional[redis.Redis] = redis.from_url(
                url, decode_responses=True, socket_connect_timeout=3
            )
            # Eagerly verify the connection
            self._redis.ping()
            logger.info("TradeJournal: Redis connected at %s", url)
        except RedisError as exc:
            logger.warning(
                "TradeJournal: could not connect to Redis (%s) — "
                "Redis logging will be skipped",
                exc,
            )
            self._redis = None
        except Exception as exc:
            logger.warning(
                "TradeJournal: unexpected error connecting to Redis (%s) — "
                "Redis logging will be skipped",
                exc,
            )
            self._redis = None

    # ------------------------------------------------------------------
    # Signal lifecycle
    # ------------------------------------------------------------------

    def on_signal_generated(self, signal) -> int:
        """Insert signal record to PostgreSQL.

        Args:
            signal: :class:`~core.signals.signal_router.SignalOutput` or any
                    object with the expected signal attributes.

        Returns:
            The database primary key of the inserted signal row,
            or -1 when no db_manager is configured.
        """
        if self._db_manager is None:
            logger.debug("TradeJournal.on_signal_generated: no db_manager — skipping")
            return -1

        try:
            regime_ctx = getattr(signal, "regime_context", None)
            regime_json: Optional[str] = None
            if regime_ctx is not None:
                try:
                    if hasattr(regime_ctx, "__dict__"):
                        regime_json = json.dumps(
                            {k: str(v) for k, v in regime_ctx.__dict__.items()}
                        )
                    else:
                        regime_json = json.dumps(str(regime_ctx))
                except (TypeError, ValueError):
                    regime_json = None

            record = {
                "magic_number": int(getattr(signal, "magic_number", 0)),
                "symbol": str(getattr(signal, "asset", "")),
                "timeframe": str(getattr(signal, "timeframe", "M15")),
                "timestamp": _utc_now_if_none(getattr(signal, "timestamp", None)),
                "direction": str(getattr(signal, "signal", "NO_TRADE")),
                "module": str(getattr(signal, "module", "UNKNOWN")),
                "confidence": float(getattr(signal, "confidence", 0.0)),
                "entry_price": float(getattr(signal, "entry_price", 0.0) or 0.0),
                "stop_loss": float(getattr(signal, "stop_loss", 0.0) or 0.0),
                "take_profit_1": float(getattr(signal, "take_profit_1", 0.0) or 0.0),
                "take_profit_2": float(getattr(signal, "take_profit_2", 0.0) or 0.0),
                "atr": float(getattr(signal, "atr", 0.0) or 0.0),
                "rr_ratio": float(getattr(signal, "rr_ratio", 0.0) or 0.0),
                "regime_context": regime_json,
                "was_executed": False,
            }

            signal_id = self._db_manager.insert_signal(record)
            logger.debug(
                "TradeJournal: signal inserted id=%d symbol=%s direction=%s",
                signal_id,
                record["symbol"],
                record["direction"],
            )
            return signal_id

        except Exception as exc:
            logger.error("TradeJournal.on_signal_generated failed: %s", exc)
            return -1

    def on_order_placed(self, ticket: int, trade_order) -> None:
        """Update signal record with ticket. Log order details.

        Args:
            ticket:      MT5 ticket of the placed order.
            trade_order: The :class:`~core.risk.TradeOrder` that was submitted.
        """
        logger.info(
            "TradeJournal: order placed ticket=%d %s %s "
            "entry=%.5f sl=%.5f lots=%.4f",
            ticket,
            getattr(trade_order, "symbol", ""),
            getattr(trade_order, "direction", ""),
            float(getattr(trade_order, "entry_price", 0.0)),
            float(getattr(trade_order, "stop_loss", 0.0)),
            float(getattr(trade_order, "lot_size", 0.0)),
        )

        if self._db_manager is None:
            return

        signal_id = getattr(trade_order, "signal_id", None)
        if signal_id and signal_id > 0:
            try:
                self._db_manager.update_trade(
                    ticket=signal_id,
                    updates={"order_ticket": ticket, "was_executed": True},
                )
            except Exception as exc:
                logger.error(
                    "TradeJournal.on_order_placed: failed to update signal "
                    "id=%s with ticket — %s",
                    signal_id,
                    exc,
                )

    def on_fill(
        self,
        ticket: int,
        fill_price: float,
        slippage_pips: float,
    ) -> None:
        """Insert/update trade record with entry details.

        Args:
            ticket:        MT5 ticket of the filled position.
            fill_price:    Actual execution price.
            slippage_pips: Slippage in pips (absolute value).
        """
        logger.info(
            "TradeJournal: fill ticket=%d price=%.5f slippage=%.1f pips",
            ticket,
            fill_price,
            slippage_pips,
        )

        if self._db_manager is None:
            return

        try:
            self._db_manager.update_trade(
                ticket=ticket,
                updates={
                    "entry_price": fill_price,
                    "slippage_pips": slippage_pips,
                    "entry_time": _utc_now_if_none(None),
                    "status": "OPEN",
                },
            )
        except Exception as exc:
            logger.error("TradeJournal.on_fill failed for ticket=%d: %s", ticket, exc)

    # ------------------------------------------------------------------
    # Position lifecycle events
    # ------------------------------------------------------------------

    def on_partial_close(
        self,
        ticket: int,
        volume: float,
        price: float,
        reason: str,
    ) -> None:
        """Log partial close event to the position_events table.

        Args:
            ticket: MT5 position ticket.
            volume: Volume closed (lots).
            price:  Execution price.
            reason: Human-readable close reason.
        """
        logger.info(
            "TradeJournal: partial close ticket=%d volume=%.4f "
            "price=%.5f reason=%s",
            ticket,
            volume,
            price,
            reason,
        )

        if self._db_manager is None:
            return

        try:
            event = {
                "ticket": ticket,
                "event_type": "partial_close",
                "volume_closed": volume,
                "close_price": price,
                "details": json.dumps({"reason": reason}),
            }
            self._db_manager.insert_position_event(event)

            # Also update trades table with partial close fields
            self._db_manager.update_trade(
                ticket=ticket,
                updates={
                    "partial_close_1_time": _utc_now_if_none(None),
                    "partial_close_1_price": price,
                    "partial_close_1_volume": volume,
                },
            )

        except Exception as exc:
            logger.error(
                "TradeJournal.on_partial_close failed for ticket=%d: %s", ticket, exc
            )

    def on_position_event(
        self,
        ticket: int,
        event_type: str,
        details: dict,
    ) -> None:
        """Log any position manager event (trail_update, stop_to_breakeven, etc.).

        Args:
            ticket:     MT5 position ticket.
            event_type: Event type string matching the position_events table enum.
            details:    Arbitrary key-value payload stored as JSONB.
        """
        logger.debug(
            "TradeJournal: position event ticket=%d type=%s details=%s",
            ticket,
            event_type,
            details,
        )

        if self._db_manager is None:
            return

        try:
            event: dict = {
                "ticket": ticket,
                "event_type": event_type,
                "details": json.dumps(details),
            }

            # Promote well-known fields to dedicated columns
            if "new_stop" in details:
                event["new_stop_loss"] = float(details["new_stop"])
            if "old_stop" in details:
                event["old_stop_loss"] = float(details["old_stop"])
            if "current_price" in details:
                event["current_price"] = float(details["current_price"])
            if "atr" in details:
                event["atr_at_event"] = float(details["atr"])

            self._db_manager.insert_position_event(event)

        except Exception as exc:
            logger.error(
                "TradeJournal.on_position_event failed for ticket=%d type=%s: %s",
                ticket,
                event_type,
                exc,
            )

    def on_close(
        self,
        ticket: int,
        exit_price: float,
        exit_reason: str,
        entry_price: Optional[float] = None,
        lot_size: Optional[float] = None,
        stop_loss: Optional[float] = None,
        direction: Optional[str] = None,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """Update trade record with exit details and computed performance metrics.

        Computes:
        - pnl_currency:  (exit_price - entry_price) * lot_size * contract_size
                         (sign-adjusted for SHORT trades)
        - pnl_percent:   pnl_currency / account_balance_at_entry * 100
        - r_multiple:    pnl_currency / risk_amount_currency
        - hold_time_candles: derived from entry_time to now in M15 bars
        - MAE/MFE are sourced from stored position_events if available

        Args:
            ticket:      MT5 position ticket.
            exit_price:  Market price at close.
            exit_reason: Reason code (e.g. ``"stop_hit"``, ``"tp2_hit"``).
            entry_price: Original entry price (used for PnL computation).
            lot_size:    Position size in lots (used for PnL computation).
            stop_loss:   Initial stop loss (used for R-multiple computation).
            direction:   ``"LONG"`` or ``"SHORT"``.
            entry_time:  UTC datetime of position open.
        """
        logger.info(
            "TradeJournal: close ticket=%d exit=%.5f reason=%s",
            ticket,
            exit_price,
            exit_reason,
        )

        if self._db_manager is None:
            return

        try:
            updates: dict = {
                "exit_time": _utc_now_if_none(None),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "status": "CLOSED",
            }

            # Compute performance metrics when we have enough data
            if (
                entry_price is not None
                and lot_size is not None
                and direction is not None
            ):
                price_diff = exit_price - entry_price
                if direction.upper() == "SHORT":
                    price_diff = -price_diff

                pnl_currency = price_diff * lot_size * _DEFAULT_CONTRACT_SIZE
                updates["pnl_currency"] = pnl_currency

                # pnl_percent — requires account balance at entry from DB
                try:
                    # We attempt a lightweight fetch; if it fails, skip gracefully
                    account_balance = self._fetch_account_balance_at_entry(ticket)
                    if account_balance and account_balance > 0:
                        updates["pnl_percent"] = (pnl_currency / account_balance) * 100
                except Exception:
                    pass  # pnl_percent is optional

                # R-multiple
                if stop_loss is not None:
                    risk_per_unit = abs(entry_price - stop_loss)
                    if risk_per_unit > 0 and lot_size > 0:
                        risk_currency = risk_per_unit * lot_size * _DEFAULT_CONTRACT_SIZE
                        if risk_currency > 0:
                            updates["r_multiple"] = pnl_currency / risk_currency

            # Hold time in M15 candles
            if entry_time is not None:
                now = datetime.now(timezone.utc)
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                elapsed_seconds = (now - entry_time).total_seconds()
                updates["hold_time_candles"] = int(elapsed_seconds / (15 * 60))

            # MAE / MFE from position_events
            try:
                mae, mfe = self._compute_mae_mfe(ticket)
                if mae is not None:
                    updates["mae"] = mae
                if mfe is not None:
                    updates["mfe"] = mfe
            except Exception:
                pass

            self._db_manager.update_trade(ticket=ticket, updates=updates)

            logger.debug(
                "TradeJournal: close recorded ticket=%d "
                "pnl=%.2f r_mult=%.2f",
                ticket,
                updates.get("pnl_currency", 0.0),
                updates.get("r_multiple", 0.0),
            )

        except Exception as exc:
            logger.error(
                "TradeJournal.on_close failed for ticket=%d: %s", ticket, exc
            )

    # ------------------------------------------------------------------
    # Redis live dashboard updates
    # ------------------------------------------------------------------

    def update_redis_positions(self, open_positions: list) -> None:
        """Write current open positions to Redis as JSON.

        Key: ``trading:open_positions``. TTL: 300 seconds.

        Args:
            open_positions: List of :class:`~core.execution.position_manager.PositionState`
                            objects (or any serialisable objects).
        """
        payload: list[dict] = []

        for pos in open_positions:
            try:
                if hasattr(pos, "__dict__"):
                    item = {
                        k: (v.isoformat() if isinstance(v, datetime) else v)
                        for k, v in pos.__dict__.items()
                    }
                else:
                    item = str(pos)
                payload.append(item)
            except Exception as exc:
                logger.debug(
                    "TradeJournal.update_redis_positions: could not serialise "
                    "position — %s",
                    exc,
                )

        self._safe_redis_set(
            _KEY_OPEN_POSITIONS,
            {"positions": payload, "count": len(payload)},
            ttl=_TTL_DEFAULT,
        )

    def update_redis_daily_pnl(
        self, pnl_currency: float, pnl_pct: float
    ) -> None:
        """Write today's PnL to Redis.

        Key: ``trading:daily_pnl``. TTL: 24 hours.

        Args:
            pnl_currency: Today's realised PnL in account currency.
            pnl_pct:      Today's PnL as a percentage of account balance.
        """
        self._safe_redis_set(
            _KEY_DAILY_PNL,
            {
                "pnl_currency": pnl_currency,
                "pnl_pct": pnl_pct,
                "updated_at": _utc_now_if_none(None).isoformat(),
            },
            ttl=_TTL_DAILY_PNL,
        )

    def update_redis_circuit_breaker(
        self, level: int, description: str
    ) -> None:
        """Write circuit breaker status to Redis.

        Key: ``trading:circuit_breaker``. TTL: 1 hour.

        Args:
            level:       Active circuit breaker level (0–4).
            description: Human-readable description of the trigger condition.
        """
        self._safe_redis_set(
            _KEY_CIRCUIT_BREAKER,
            {
                "level": level,
                "description": description,
                "updated_at": _utc_now_if_none(None).isoformat(),
            },
            ttl=_TTL_CB,
        )

    def update_redis_last_signal(self, symbol: str, signal) -> None:
        """Write last signal per asset to Redis.

        Key: ``trading:last_signal:{symbol}``. TTL: 1 hour.

        Args:
            symbol: Instrument symbol, e.g. ``"EURUSD"``.
            signal: :class:`~core.signals.signal_router.SignalOutput` (or similar).
        """
        try:
            payload: dict = {
                "symbol": symbol,
                "direction": str(getattr(signal, "signal", "NO_TRADE")),
                "module": str(getattr(signal, "module", "UNKNOWN")),
                "confidence": float(getattr(signal, "confidence", 0.0)),
                "entry_price": float(getattr(signal, "entry_price", 0.0) or 0.0),
                "timestamp": _utc_now_if_none(
                    getattr(signal, "timestamp", None)
                ).isoformat(),
            }
        except Exception as exc:
            logger.debug(
                "TradeJournal.update_redis_last_signal: serialisation error — %s",
                exc,
            )
            payload = {"symbol": symbol, "error": str(exc)}

        self._safe_redis_set(
            f"{_KEY_LAST_SIGNAL_PREFIX}:{symbol}",
            payload,
            ttl=_TTL_LAST_SIGNAL,
        )

    def _safe_redis_set(self, key: str, value: dict, ttl: int = 300) -> None:
        """Safely set a Redis key with JSON-serialised value and TTL.

        All :class:`redis.RedisError` exceptions are swallowed — Redis is
        optional infrastructure and must never disrupt trade execution.

        Args:
            key:   Redis key string.
            value: Dict payload to serialise to JSON.
            ttl:   Key expiry in seconds.
        """
        if self._redis is None:
            return

        try:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        except RedisError as exc:
            logger.warning(
                "TradeJournal: Redis set failed for key=%s — %s", key, exc
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_account_balance_at_entry(self, ticket: int) -> Optional[float]:
        """Retrieve the account balance recorded at trade entry from PostgreSQL.

        Args:
            ticket: MT5 position ticket.

        Returns:
            Account balance in currency, or None if unavailable.
        """
        if self._db_manager is None:
            return None

        try:
            # DatabaseManager uses SQLAlchemy text queries; we call a simple
            # fetch via a method that accepts raw SQL if available, otherwise
            # fall back to a get_candles-style approach.
            # The simplest approach: use the engine directly via execute on a
            # text query.  DatabaseManager exposes _engine publicly for this.
            from sqlalchemy import text  # noqa: PLC0415

            engine = getattr(self._db_manager, "_engine", None)
            if engine is None:
                return None

            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT account_balance_at_entry FROM trades "
                        "WHERE ticket = :ticket LIMIT 1"
                    ),
                    {"ticket": ticket},
                )
                row = result.fetchone()

            if row and row[0] is not None:
                return float(row[0])

        except Exception as exc:
            logger.debug(
                "TradeJournal._fetch_account_balance_at_entry: %s", exc
            )

        return None

    def _compute_mae_mfe(
        self, ticket: int
    ) -> tuple[Optional[float], Optional[float]]:
        """Derive max adverse excursion (MAE) and max favourable excursion (MFE)
        from stored position_events for the given ticket.

        Both values are returned as absolute price differences (not pips).

        Args:
            ticket: MT5 position ticket.

        Returns:
            Tuple of (mae, mfe) as floats, or (None, None) if unavailable.
        """
        if self._db_manager is None:
            return None, None

        try:
            from sqlalchemy import text  # noqa: PLC0415

            engine = getattr(self._db_manager, "_engine", None)
            if engine is None:
                return None, None

            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT current_price FROM position_events "
                        "WHERE ticket = :ticket "
                        "ORDER BY timestamp ASC"
                    ),
                    {"ticket": ticket},
                )
                rows = result.fetchall()

            if not rows:
                return None, None

            prices = [float(r[0]) for r in rows if r[0] is not None]
            if not prices:
                return None, None

            # entry_price needed for MAE/MFE calculation
            entry_price: Optional[float] = None
            try:
                with engine.connect() as conn:
                    result = conn.execute(
                        text(
                            "SELECT entry_price, direction FROM trades "
                            "WHERE ticket = :ticket LIMIT 1"
                        ),
                        {"ticket": ticket},
                    )
                    row = result.fetchone()
                if row:
                    entry_price = float(row[0]) if row[0] else None
                    direction = str(row[1]) if row[1] else "LONG"
            except Exception:
                return None, None

            if entry_price is None:
                return None, None

            if direction.upper() == "LONG":
                adverse = [entry_price - p for p in prices if p < entry_price]
                favourable = [p - entry_price for p in prices if p > entry_price]
            else:
                adverse = [p - entry_price for p in prices if p > entry_price]
                favourable = [entry_price - p for p in prices if p < entry_price]

            mae = max(adverse) if adverse else 0.0
            mfe = max(favourable) if favourable else 0.0

            return mae, mfe

        except Exception as exc:
            logger.debug("TradeJournal._compute_mae_mfe: %s", exc)
            return None, None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utc_now_if_none(dt: Optional[datetime]) -> datetime:
    """Return *dt* if provided, otherwise the current UTC datetime.

    Args:
        dt: Datetime value or None.

    Returns:
        A timezone-aware UTC datetime.
    """
    if dt is not None and isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.now(timezone.utc)
