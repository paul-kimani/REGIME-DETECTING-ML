"""MT5Connector — direct wrapper around the MetaTrader5 Python library.

This is the ONLY file in the entire codebase that imports MetaTrader5.
All other modules use MT5Connector exclusively.

On Windows: MetaTrader5 is available natively and connects directly to the
running MT5 terminal on the same machine. No HTTP layer, no bridge process.

On Mac: MetaTrader5 is not installable. The import is guarded so the rest
of the codebase can still be imported and tested using a mock MT5Connector.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Protected import — MetaTrader5 only available on Windows
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as _mt5_lib
    _MT5_AVAILABLE = True
except ImportError:
    _mt5_lib = None
    _MT5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom exception (same name as the old HTTP bridge used)
# ---------------------------------------------------------------------------

class MT5ConnectionError(Exception):
    """Raised when MT5 is not connected or a terminal call fails."""


# ---------------------------------------------------------------------------
# MT5Connector
# ---------------------------------------------------------------------------

class MT5Connector:
    """Direct wrapper around the MetaTrader5 Python library.

    Exposes timeframe and order-type constants as class attributes so that
    no other module needs to import MetaTrader5.  The integer values are the
    official MT5 API constants and are hardcoded here so they work even when
    MetaTrader5 is not installed (e.g. on Mac for testing).

    External API is intentionally identical to the old MT5Client (HTTP) so
    that modules only need their import line changed — no logic rewrites.
    """

    # ------------------------------------------------------------------ #
    # Timeframe constants                                                  #
    # ------------------------------------------------------------------ #
    TIMEFRAME_M1  = 1
    TIMEFRAME_M5  = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1  = 16385
    TIMEFRAME_H4  = 16388
    TIMEFRAME_D1  = 16408
    TIMEFRAME_W1  = 32769
    TIMEFRAME_MN1 = 49153

    # ------------------------------------------------------------------ #
    # Order-type constants                                                 #
    # ------------------------------------------------------------------ #
    ORDER_TYPE_BUY             = 0
    ORDER_TYPE_SELL            = 1
    ORDER_TYPE_BUY_LIMIT       = 2
    ORDER_TYPE_SELL_LIMIT      = 3
    ORDER_TYPE_BUY_STOP        = 4
    ORDER_TYPE_SELL_STOP       = 5
    ORDER_TYPE_BUY_STOP_LIMIT  = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7

    # ------------------------------------------------------------------ #
    # Trade-action constants                                               #
    # ------------------------------------------------------------------ #
    TRADE_ACTION_DEAL    = 1   # Immediate market execution
    TRADE_ACTION_PENDING = 5   # Place a pending order
    TRADE_ACTION_SLTP    = 6   # Modify SL/TP on an open position

    # ------------------------------------------------------------------ #
    # Return-code constants                                                #
    # ------------------------------------------------------------------ #
    TRADE_RETCODE_DONE = 10009

    # ------------------------------------------------------------------ #
    # Retry settings                                                       #
    # ------------------------------------------------------------------ #
    _MAX_RETRIES   = 3
    _RETRY_BACKOFF = 2.0  # seconds between retries

    def __init__(self) -> None:
        self._connected: bool = False
        self._login:     Optional[int]  = None
        self._password:  Optional[str]  = None
        self._server:    Optional[str]  = None
        self._path:      Optional[str]  = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        path:     Optional[str] = None,
        login:    Optional[int] = None,
        password: Optional[str] = None,
        server:   Optional[str] = None,
    ) -> bool:
        """Initialise the MT5 terminal connection.

        Args:
            path:     Full path to terminal64.exe (optional — auto-detected
                      by MT5 library if not supplied).
            login:    MT5 account number.  If None, uses the account that is
                      already logged in to the running terminal.
            password: MT5 account password.
            server:   Broker server name (e.g. ``"ICMarkets-Demo"``).

        Returns:
            True on success.

        Raises:
            MT5ConnectionError: If MetaTrader5 is not installed or the
                terminal cannot be reached.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError(
                "MetaTrader5 Python package is not installed. "
                "Install it with: pip install MetaTrader5  (Windows only)"
            )

        self._path     = path
        self._login    = login
        self._password = password
        self._server   = server

        kwargs: dict = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs["login"] = int(login)
        if password:
            kwargs["password"] = password
        if server:
            kwargs["server"] = server

        ok = _mt5_lib.initialize(**kwargs) if kwargs else _mt5_lib.initialize()
        if not ok:
            err = _mt5_lib.last_error()
            raise MT5ConnectionError(f"MT5 initialize() failed: {err}")

        self._connected = True
        info = _mt5_lib.account_info()
        if info:
            logger.info(
                "MT5Connector connected — server=%s account=%s",
                getattr(info, "server", "?"),
                getattr(info, "login",  "?"),
            )
        return True

    def shutdown(self) -> None:
        """Shutdown the MT5 connection gracefully."""
        if _MT5_AVAILABLE and self._connected:
            _mt5_lib.shutdown()
        self._connected = False
        logger.info("MT5Connector shutdown")

    def is_connected(self) -> bool:
        """Return True if the terminal is reachable."""
        if not _MT5_AVAILABLE or not self._connected:
            return False
        info = _mt5_lib.terminal_info()
        return info is not None and getattr(info, "connected", False)

    def _reconnect(self) -> bool:
        """Attempt to re-initialise after a lost connection."""
        logger.warning("MT5Connector attempting reconnect …")
        try:
            return self.initialize(
                path=self._path,
                login=self._login,
                password=self._password,
                server=self._server,
            )
        except MT5ConnectionError as exc:
            logger.error("MT5Connector reconnect failed: %s", exc)
            return False

    def _retry(self, fn, *args, **kwargs):
        """Call *fn* with retry/reconnect on failure.

        Returns the function's return value or raises MT5ConnectionError.
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                result = fn(*args, **kwargs)
                if result is not None:
                    return result
                # MT5 returns None on failure; check last error
                err = _mt5_lib.last_error() if _MT5_AVAILABLE else (-1, "MT5 unavailable")
                if err[0] in (-10004, -10003):  # NOT_CONNECTED / COMMON_ERROR
                    logger.warning(
                        "MT5 not connected (attempt %d/%d) — reconnecting",
                        attempt, self._MAX_RETRIES,
                    )
                    self._reconnect()
                else:
                    logger.warning("MT5 call returned None: %s (attempt %d)", err, attempt)
            except Exception as exc:
                logger.warning("MT5 call exception (attempt %d): %s", attempt, exc)
            if attempt < self._MAX_RETRIES:
                time.sleep(self._RETRY_BACKOFF)
        raise MT5ConnectionError(
            f"MT5 call failed after {self._MAX_RETRIES} attempts: {fn.__name__}"
        )

    # ------------------------------------------------------------------ #
    # Account / health                                                     #
    # ------------------------------------------------------------------ #

    def health_check(self) -> dict:
        """Return a connection health summary.

        Returns:
            Dict with keys ``mt5_connected`` (bool) and ``account_server`` (str).
        """
        if not _MT5_AVAILABLE or not self._connected:
            return {"mt5_connected": False, "account_server": ""}
        info = _mt5_lib.account_info()
        if info is None:
            return {"mt5_connected": False, "account_server": ""}
        return {
            "mt5_connected":  True,
            "account_server": getattr(info, "server", ""),
        }

    def get_account_info(self) -> dict:
        """Return account info as a plain dict.

        Returns:
            Dict with keys: login, balance, equity, margin, free_margin,
            currency, leverage, server, name.

        Raises:
            MT5ConnectionError: If terminal is unreachable.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        info = self._retry(_mt5_lib.account_info)
        return {
            "login":        getattr(info, "login",       0),
            "balance":      getattr(info, "balance",     0.0),
            "equity":       getattr(info, "equity",      0.0),
            "margin":       getattr(info, "margin",      0.0),
            "free_margin":  getattr(info, "margin_free", 0.0),
            "currency":     getattr(info, "currency",    ""),
            "leverage":     getattr(info, "leverage",    1),
            "server":       getattr(info, "server",      ""),
            "name":         getattr(info, "name",        ""),
        }

    # Alias used in run_live.py candle callback
    def account_info(self) -> dict:
        """Alias for :meth:`get_account_info`."""
        return self.get_account_info()

    # ------------------------------------------------------------------ #
    # Symbol info                                                          #
    # ------------------------------------------------------------------ #

    def get_symbol_info(self, symbol: str) -> dict:
        """Return symbol specification as a plain dict.

        Args:
            symbol: Instrument symbol (e.g. ``"XAUUSD"``).

        Returns:
            Dict with keys: name, bid, ask, spread, digits, point,
            volume_min, volume_max, volume_step, contract_size,
            trade_stops_level, trade_mode.

        Raises:
            MT5ConnectionError: If symbol is not found or terminal unreachable.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        info = self._retry(_mt5_lib.symbol_info, symbol)
        return {
            "name":               getattr(info, "name",               symbol),
            "bid":                getattr(info, "bid",                 0.0),
            "ask":                getattr(info, "ask",                 0.0),
            "spread":             getattr(info, "spread",              0),
            "digits":             getattr(info, "digits",              5),
            "point":              getattr(info, "point",               0.00001),
            "volume_min":         getattr(info, "volume_min",          0.01),
            "volume_max":         getattr(info, "volume_max",          100.0),
            "volume_step":        getattr(info, "volume_step",         0.01),
            "contract_size":      getattr(info, "trade_contract_size", 100_000.0),
            "trade_stops_level":  getattr(info, "trade_stops_level",   0),
            "trade_mode":         getattr(info, "trade_mode",          0),
        }

    # Alias
    def symbol_info(self, symbol: str) -> dict:
        """Alias for :meth:`get_symbol_info`."""
        return self.get_symbol_info(symbol)

    # ------------------------------------------------------------------ #
    # Tick                                                                 #
    # ------------------------------------------------------------------ #

    def symbol_info_tick(self, symbol: str) -> dict:
        """Return the latest bid/ask tick for *symbol*.

        Args:
            symbol: Instrument symbol.

        Returns:
            Dict with keys: symbol, bid, ask, last, volume, time (ISO string).

        Raises:
            MT5ConnectionError: On failure.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        tick = self._retry(_mt5_lib.symbol_info_tick, symbol)
        return {
            "symbol": symbol,
            "bid":    getattr(tick, "bid",    0.0),
            "ask":    getattr(tick, "ask",    0.0),
            "last":   getattr(tick, "last",   0.0),
            "volume": getattr(tick, "volume", 0),
            "time":   datetime.fromtimestamp(
                          getattr(tick, "time", 0), tz=timezone.utc
                      ).isoformat(),
        }

    # Alias
    def get_tick(self, symbol: str) -> dict:
        """Alias for :meth:`symbol_info_tick`."""
        return self.symbol_info_tick(symbol)

    # ------------------------------------------------------------------ #
    # OHLCV data                                                           #
    # ------------------------------------------------------------------ #

    def copy_rates_from_pos(
        self,
        symbol:    str,
        timeframe: int,
        start_pos: int,
        count:     int,
    ) -> list[dict]:
        """Fetch *count* bars starting at *start_pos* from the end.

        Args:
            symbol:    Instrument symbol.
            timeframe: MT5 timeframe constant (e.g. ``MT5Connector.TIMEFRAME_M15``).
            start_pos: Bar index from the current bar (0 = latest).
            count:     Number of bars to fetch.

        Returns:
            List of dicts with keys: time (ISO string), open, high, low,
            close, tick_volume, spread, real_volume.

        Raises:
            MT5ConnectionError: On failure.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        rates = self._retry(
            _mt5_lib.copy_rates_from_pos, symbol, timeframe, start_pos, count
        )
        return self._rates_to_dicts(rates)

    def copy_rates_range(
        self,
        symbol:    str,
        timeframe: int,
        date_from: datetime,
        date_to:   datetime,
    ) -> list[dict]:
        """Fetch bars between two datetimes.

        Args:
            symbol:    Instrument symbol.
            timeframe: MT5 timeframe constant.
            date_from: Start of range (UTC-aware datetime).
            date_to:   End of range (UTC-aware datetime).

        Returns:
            List of rate dicts (same schema as :meth:`copy_rates_from_pos`).

        Raises:
            MT5ConnectionError: On failure.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        rates = self._retry(
            _mt5_lib.copy_rates_range, symbol, timeframe, date_from, date_to
        )
        return self._rates_to_dicts(rates)

    @staticmethod
    def _rates_to_dicts(rates) -> list[dict]:
        """Convert MT5 numpy structured array of rates to a list of plain dicts.

        MT5 returns a numpy structured array.  We convert each row to a plain
        dict, converting the Unix timestamp to an ISO-8601 string so downstream
        code (data_pipeline._convert_mt5_rates) can parse it uniformly.
        """
        if rates is None:
            return []
        result = []
        for row in rates:
            result.append({
                "time":        datetime.fromtimestamp(
                                   int(row["time"]), tz=timezone.utc
                               ).isoformat(),
                "open":        float(row["open"]),
                "high":        float(row["high"]),
                "low":         float(row["low"]),
                "close":       float(row["close"]),
                "tick_volume": int(row["tick_volume"]),
                "spread":      int(row["spread"]),
                "real_volume": int(row["real_volume"]) if "real_volume" in row.dtype.names else 0,
            })
        return result

    # ------------------------------------------------------------------ #
    # Positions and orders                                                 #
    # ------------------------------------------------------------------ #

    def get_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """Return open positions as a list of plain dicts.

        Args:
            symbol: Filter to a specific symbol, or None for all.

        Returns:
            List of position dicts with keys: ticket, symbol, type, volume,
            price_open, sl, tp, profit, magic, comment.

        Raises:
            MT5ConnectionError: On failure.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        if symbol:
            raw = _mt5_lib.positions_get(symbol=symbol)
        else:
            raw = _mt5_lib.positions_get()
        if raw is None:
            err = _mt5_lib.last_error()
            if err[0] == 0:   # no error, just no positions
                return []
            raise MT5ConnectionError(f"positions_get failed: {err}")
        return [self._position_to_dict(p) for p in raw]

    # Alias matching old MT5Client method name
    def positions_get(self, symbol: Optional[str] = None) -> list[dict]:
        """Alias for :meth:`get_positions`."""
        return self.get_positions(symbol)

    def get_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Return pending orders as a list of plain dicts.

        Args:
            symbol: Filter to a specific symbol, or None for all.

        Returns:
            List of order dicts with keys: ticket, symbol, type, volume_initial,
            volume_current, price_open, sl, tp, magic, comment, state.

        Raises:
            MT5ConnectionError: On failure.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        if symbol:
            raw = _mt5_lib.orders_get(symbol=symbol)
        else:
            raw = _mt5_lib.orders_get()
        if raw is None:
            err = _mt5_lib.last_error()
            if err[0] == 0:
                return []
            raise MT5ConnectionError(f"orders_get failed: {err}")
        return [self._order_to_dict(o) for o in raw]

    # Alias
    def orders_get(self, symbol: Optional[str] = None) -> list[dict]:
        """Alias for :meth:`get_orders`."""
        return self.get_orders(symbol)

    # ------------------------------------------------------------------ #
    # Order execution                                                      #
    # ------------------------------------------------------------------ #

    def order_send(self, request: dict) -> dict:
        """Send an order request to MT5.

        The caller is responsible for building the complete request dict
        with keys: action, symbol, volume, type, price, sl, tp, deviation,
        magic, comment.  SL and TP should always be 0 — stop management
        is handled by PositionManager in Python.

        Args:
            request: MT5 order request dict.

        Returns:
            Dict with keys: retcode (int), ticket (int), comment (str),
            request_id (int), deal (int).

        Raises:
            MT5ConnectionError: If terminal is unreachable.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        result = _mt5_lib.order_send(request)
        if result is None:
            err = _mt5_lib.last_error()
            raise MT5ConnectionError(f"order_send returned None: {err}")
        return {
            "retcode":    getattr(result, "retcode",    -1),
            "ticket":     getattr(result, "order",      0),
            "deal":       getattr(result, "deal",       0),
            "comment":    getattr(result, "comment",    ""),
            "request_id": getattr(result, "request_id", 0),
        }

    def order_check(self, request: dict) -> dict:
        """Check an order request without actually sending it.

        Args:
            request: MT5 order request dict.

        Returns:
            Dict with keys: retcode (int), balance, equity, profit, margin,
            margin_free, margin_level, comment.

        Raises:
            MT5ConnectionError: If terminal is unreachable.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        result = _mt5_lib.order_check(request)
        if result is None:
            err = _mt5_lib.last_error()
            raise MT5ConnectionError(f"order_check returned None: {err}")
        return {
            "retcode":      getattr(result, "retcode",      -1),
            "balance":      getattr(result, "balance",      0.0),
            "equity":       getattr(result, "equity",       0.0),
            "profit":       getattr(result, "profit",       0.0),
            "margin":       getattr(result, "margin",       0.0),
            "margin_free":  getattr(result, "margin_free",  0.0),
            "margin_level": getattr(result, "margin_level", 0.0),
            "comment":      getattr(result, "comment",      ""),
        }

    def order_cancel(self, ticket: int) -> dict:
        """Cancel a pending order by ticket number.

        Args:
            ticket: MT5 order ticket to cancel.

        Returns:
            Dict with keys: retcode (int), ticket (int), comment (str).

        Raises:
            MT5ConnectionError: If terminal is unreachable.
        """
        if not _MT5_AVAILABLE:
            raise MT5ConnectionError("MetaTrader5 not available")
        request = {
            "action": 8,  # TRADE_ACTION_REMOVE = 8 (cancel pending order)
            "order":  ticket,
        }
        result = _mt5_lib.order_send(request)
        if result is None:
            err = _mt5_lib.last_error()
            raise MT5ConnectionError(f"order_cancel (order_send REMOVE) returned None: {err}")
        return {
            "retcode": getattr(result, "retcode", -1),
            "ticket":  ticket,
            "comment": getattr(result, "comment", ""),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _position_to_dict(p) -> dict:
        """Convert an MT5 position namedtuple to a plain dict."""
        return {
            "ticket":      getattr(p, "ticket",      0),
            "symbol":      getattr(p, "symbol",      ""),
            "type":        getattr(p, "type",         0),
            "volume":      getattr(p, "volume",       0.0),
            "price_open":  getattr(p, "price_open",   0.0),
            "price_current": getattr(p, "price_current", 0.0),
            "sl":          getattr(p, "sl",           0.0),
            "tp":          getattr(p, "tp",           0.0),
            "profit":      getattr(p, "profit",       0.0),
            "magic":       getattr(p, "magic",        0),
            "comment":     getattr(p, "comment",      ""),
            "time":        getattr(p, "time",         0),
        }

    @staticmethod
    def _order_to_dict(o) -> dict:
        """Convert an MT5 order namedtuple to a plain dict."""
        return {
            "ticket":         getattr(o, "ticket",         0),
            "symbol":         getattr(o, "symbol",         ""),
            "type":           getattr(o, "type",            0),
            "volume_initial": getattr(o, "volume_initial",  0.0),
            "volume_current": getattr(o, "volume_current",  0.0),
            "price_open":     getattr(o, "price_open",      0.0),
            "sl":             getattr(o, "sl",              0.0),
            "tp":             getattr(o, "tp",              0.0),
            "magic":          getattr(o, "magic",           0),
            "comment":        getattr(o, "comment",         ""),
            "state":          getattr(o, "state",           0),
            "time_setup":     getattr(o, "time_setup",      0),
        }
