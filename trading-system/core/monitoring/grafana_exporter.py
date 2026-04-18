"""GrafanaExporter — pushes live metrics to Redis for Grafana dashboards."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import redis
import redis.exceptions

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Key prefix and TTLs
# ---------------------------------------------------------------------------

_KEY_PREFIX = "trading:"
_TTL_REGIME = 120      # seconds — regime state is refreshed every bar (~15 s)
_TTL_SIGNAL = 60       # seconds — signal is only relevant until next bar
_TTL_ACCOUNT = 30      # seconds — account summary refreshed very frequently
_TTL_MODEL_HEALTH = 3600  # seconds — model metrics are stable between daily runs
_TTL_SYSTEM_STATUS = 60   # seconds


# ---------------------------------------------------------------------------
# GrafanaExporter
# ---------------------------------------------------------------------------


class GrafanaExporter:
    """Pushes live trading metrics to Redis for Grafana dashboards.

    All keys are prefixed with ``trading:`` for the Grafana Redis data-source.

    Redis is treated as optional infrastructure: if the server is unreachable
    at construction time, or if any individual write fails, the error is logged
    at WARNING level and execution continues normally.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        """Connect to Redis.  Gracefully degrade if unavailable.

        Args:
            redis_url: Redis connection URL.  Falls back to the ``REDIS_URL``
                       environment variable, then ``redis://localhost:6379``.
        """
        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[redis.Redis] = None

        try:
            client: redis.Redis = redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            client.ping()
            self._redis = client
            logger.info("GrafanaExporter: Redis connected at %s", url)
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "GrafanaExporter: could not connect to Redis (%s) — "
                "dashboard exports will be skipped",
                exc,
            )

    # ------------------------------------------------------------------
    # Public export methods
    # ------------------------------------------------------------------

    def export_regime_state(self, symbol: str, regime_state: object) -> None:
        """Write the current regime state for *symbol* to Redis.

        Key: ``trading:regime:{symbol}``. TTL: 120 s.

        Args:
            symbol:       Instrument identifier, e.g. ``"EURUSD"``.
            regime_state: Any object that is JSON-serialisable or exposes
                          ``__dict__``.  Dataclasses, plain dicts, and named-
                          tuples are all accepted.
        """
        key = f"{_KEY_PREFIX}regime:{symbol}"
        payload = _to_dict(regime_state)
        payload["symbol"] = symbol
        payload["exported_at"] = _utc_iso()
        self._push(key, payload, ttl=_TTL_REGIME)

    def export_signal(self, signal: object) -> None:
        """Write the latest signal details to Redis.

        Key: ``trading:signal:{symbol}``. TTL: 60 s.

        The symbol is read from ``signal.asset`` or ``signal.symbol``; falls
        back to ``"UNKNOWN"`` when neither attribute is present.

        Args:
            signal: Signal object (e.g. ``SignalOutput``) with attributes
                    ``asset`` / ``symbol``, ``signal``, ``module``,
                    ``confidence``, ``entry_price``, ``stop_loss``,
                    ``take_profit_1``, ``take_profit_2``, ``rr_ratio``.
        """
        symbol: str = str(
            getattr(signal, "asset", None)
            or getattr(signal, "symbol", "UNKNOWN")
        )
        key = f"{_KEY_PREFIX}signal:{symbol}"

        payload: dict = {
            "symbol": symbol,
            "direction": str(getattr(signal, "signal", "NO_TRADE")),
            "module": str(getattr(signal, "module", "UNKNOWN")),
            "confidence": _safe_float(getattr(signal, "confidence", 0.0)),
            "entry_price": _safe_float(getattr(signal, "entry_price", 0.0)),
            "stop_loss": _safe_float(getattr(signal, "stop_loss", 0.0)),
            "take_profit_1": _safe_float(getattr(signal, "take_profit_1", 0.0)),
            "take_profit_2": _safe_float(getattr(signal, "take_profit_2", 0.0)),
            "rr_ratio": _safe_float(getattr(signal, "rr_ratio", 0.0)),
            "exported_at": _utc_iso(),
        }

        ts = getattr(signal, "timestamp", None)
        if ts is not None:
            payload["signal_timestamp"] = str(ts)

        self._push(key, payload, ttl=_TTL_SIGNAL)

    def export_account(self, account_state: dict) -> None:
        """Write the current account summary to Redis.

        Key: ``trading:account``. TTL: 30 s.

        Args:
            account_state: Account metrics dict (balance, equity, daily_pnl, etc.).
        """
        key = f"{_KEY_PREFIX}account"
        payload = dict(account_state)
        payload["exported_at"] = _utc_iso()
        self._push(key, payload, ttl=_TTL_ACCOUNT)

    def export_model_health(self, symbol: str, metrics: dict) -> None:
        """Write the latest model performance metrics for *symbol* to Redis.

        Key: ``trading:model_health:{symbol}``. TTL: 3600 s.

        Args:
            symbol:  Instrument identifier.
            metrics: Dict of model-level metrics (win_rate, sharpe, etc.).
        """
        key = f"{_KEY_PREFIX}model_health:{symbol}"
        payload = dict(metrics)
        payload["symbol"] = symbol
        payload["exported_at"] = _utc_iso()
        self._push(key, payload, ttl=_TTL_MODEL_HEALTH)

    def export_system_status(self, status: dict) -> None:
        """Write overall system status to Redis.

        Key: ``trading:system_status``. TTL: 60 s.

        Args:
            status: Dict describing system health (e.g. circuit_breaker_level,
                    mt5_connected, data_feed_ok, etc.).
        """
        key = f"{_KEY_PREFIX}system_status"
        payload = dict(status)
        payload["exported_at"] = _utc_iso()
        self._push(key, payload, ttl=_TTL_SYSTEM_STATUS)

    # ------------------------------------------------------------------
    # Internal push helper
    # ------------------------------------------------------------------

    def _push(self, key: str, value: dict, ttl: int) -> None:
        """Serialise *value* to JSON and write it to Redis with a TTL.

        All :class:`redis.exceptions.RedisError` exceptions are swallowed so
        that a Redis outage never disrupts trade execution.

        Args:
            key:   Full Redis key string (including ``trading:`` prefix).
            value: Dict payload to serialise.
            ttl:   Key expiry in seconds.
        """
        if self._redis is None:
            return

        try:
            serialised = json.dumps(value, default=str)
            self._redis.setex(key, ttl, serialised)
            logger.debug("GrafanaExporter._push: key=%s ttl=%ds", key, ttl)
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "GrafanaExporter._push: Redis write failed for key=%s — %s", key, exc
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_dict(obj: object) -> dict:
    """Convert *obj* to a plain dict suitable for JSON serialisation.

    Handles plain dicts, objects with ``__dict__``, and falls back to
    ``{"value": str(obj)}`` for anything else.

    Args:
        obj: Any object.

    Returns:
        A plain dict.
    """
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        result: dict = {}
        for k, v in obj.__dict__.items():
            result[k] = v.isoformat() if isinstance(v, datetime) else v
        return result
    return {"value": str(obj)}


def _safe_float(value: object) -> float:
    """Convert *value* to float, returning 0.0 on failure.

    Args:
        value: Numeric or None.

    Returns:
        Float representation, or 0.0 if conversion fails.
    """
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        UTC datetime string, e.g. ``"2026-04-16T22:00:00.000000"``.
    """
    return datetime.now(timezone.utc).isoformat()
