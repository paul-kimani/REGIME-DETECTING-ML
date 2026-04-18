"""MT5Client — HTTP client bridging to the Windows MT5 bridge server."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MT5ConnectionError(Exception):
    """Raised when all retry attempts to reach the MT5 bridge server fail."""


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------

def _should_retry(exc: BaseException) -> bool:
    """Return True for transient errors that warrant a retry.

    Retries on:
    - Any :class:`httpx.RequestError` (network-level failures, timeouts, etc.)
    - :class:`httpx.HTTPStatusError` with a 5xx status code (server errors).
    """
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


# ---------------------------------------------------------------------------
# MT5Client
# ---------------------------------------------------------------------------

class MT5Client:
    """HTTP client that mirrors the MetaTrader5 Python library API.

    Communicates with the Windows MT5 bridge server (FastAPI) so the rest
    of the system can interact with MetaTrader 5 without platform-specific
    dependencies.

    Args:
        base_url: Base URL of the MT5 bridge server.  Falls back to the
            ``MT5_BRIDGE_URL`` environment variable.
        api_key: API key sent in the ``X-API-Key`` header on every request.
            Falls back to the ``MT5_BRIDGE_API_KEY`` environment variable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._base_url: str = (
            base_url or os.environ.get("MT5_BRIDGE_URL", "http://localhost:8000")
        ).rstrip("/")
        self._api_key: str = api_key or os.environ.get("MT5_BRIDGE_API_KEY", "")
        self._client = httpx.Client(
            timeout=10.0,
            headers={"X-API-Key": self._api_key},
        )
        logger.debug(
            "MT5Client initialised — bridge URL: %s", self._base_url
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Execute an HTTP request against the bridge server.

        Applies retry logic (up to 3 attempts with exponential back-off)
        for transient network and 5xx server errors.  Logs every attempt at
        DEBUG level including elapsed time.

        Args:
            method: HTTP method string (``"GET"``, ``"POST"``, …).
            path: URL path relative to *base_url* (e.g. ``"/health"``).
            **kwargs: Extra keyword arguments forwarded to
                :meth:`httpx.Client.request` (e.g. ``params``, ``json``).

        Returns:
            Parsed JSON response body as a :class:`dict`.

        Raises:
            MT5ConnectionError: If all retry attempts are exhausted.
        """
        url = f"{self._base_url}{path}"

        @retry(
            retry=retry_if_exception(_should_retry),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=False,
        )
        def _do_request() -> dict:
            t0 = time.monotonic()
            try:
                response = self._client.request(method, url, **kwargs)
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "%s %s — status %d — %.1f ms",
                    method.upper(),
                    url,
                    response.status_code,
                    elapsed_ms,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "%s %s — HTTP error %d — %.1f ms",
                    method.upper(),
                    url,
                    exc.response.status_code,
                    elapsed_ms,
                )
                raise
            except httpx.RequestError as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "%s %s — request error: %s — %.1f ms",
                    method.upper(),
                    url,
                    exc,
                    elapsed_ms,
                )
                raise
            except json.JSONDecodeError as exc:
                logger.debug(
                    "%s %s — JSON decode error: %s",
                    method.upper(),
                    url,
                    exc,
                )
                raise MT5ConnectionError(
                    f"MT5 bridge returned non-JSON response for {method.upper()} {url}: {exc}"
                ) from exc

        try:
            return _do_request()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise MT5ConnectionError(
                f"MT5 bridge unreachable after 3 attempts — {method.upper()} {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API — mirrors MetaTrader5 Python library naming
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Return the bridge server health status.

        Returns:
            A dict containing at minimum ``mt5_connected`` (bool) and other
            diagnostic fields returned by the bridge.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("GET", "/health")

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: str,
        start_pos: int,
        count: int,
    ) -> list[dict]:
        """Fetch historical OHLCV candles starting from a bar offset.

        Args:
            symbol: Instrument symbol (e.g. ``"EURUSD"``).
            timeframe: Timeframe string recognised by the bridge
                (e.g. ``"M1"``, ``"H1"``, ``"D1"``).
            start_pos: Index of the first bar to return (0 = current bar).
            count: Number of bars to return.

        Returns:
            List of OHLCV dicts, each containing ``time``, ``open``,
            ``high``, ``low``, ``close``, and ``tick_volume`` keys.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request(
            "GET",
            f"/candles/{symbol}/{timeframe}/{count}",
            params={"start_pos": start_pos},
        )

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> list[dict]:
        """Fetch historical OHLCV candles within a date range.

        Args:
            symbol: Instrument symbol (e.g. ``"EURUSD"``).
            timeframe: Timeframe string (e.g. ``"M15"``, ``"H4"``).
            date_from: Start of the range (inclusive).
            date_to: End of the range (inclusive).

        Returns:
            List of OHLCV dicts ordered by ascending time.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request(
            "GET",
            f"/candles/{symbol}/{timeframe}/range",
            params={
                "from_date": date_from.isoformat(),
                "to_date": date_to.isoformat(),
            },
        )

    def symbol_info_tick(self, symbol: str) -> dict:
        """Return the latest tick for *symbol*.

        Args:
            symbol: Instrument symbol (e.g. ``"GBPUSD"``).

        Returns:
            Tick dict containing at minimum ``bid``, ``ask``, ``spread``,
            and ``time`` fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("GET", f"/tick/{symbol}")

    def account_info(self) -> dict:
        """Return the current trading account details.

        Returns:
            Account dict containing balance, equity, margin, free margin,
            leverage, currency, and other broker-specific fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("GET", "/account")

    def positions_get(self, symbol: Optional[str] = None) -> list[dict]:
        """Return all open positions, optionally filtered by symbol.

        Args:
            symbol: If provided, only positions for this instrument are
                returned (client-side filter).

        Returns:
            List of position dicts.  Each dict contains at minimum
            ``ticket``, ``symbol``, ``type``, ``volume``, ``open_price``,
            ``current_price``, and ``profit`` fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        positions: list[dict] = self._request("GET", "/positions")
        if symbol is not None:
            positions = [p for p in positions if p.get("symbol") == symbol]
        return positions

    def orders_get(self, symbol: Optional[str] = None) -> list[dict]:
        """Return all pending orders, optionally filtered by symbol.

        Args:
            symbol: If provided, only orders for this instrument are
                returned (client-side filter).

        Returns:
            List of order dicts.  Each dict contains at minimum
            ``ticket``, ``symbol``, ``type``, ``volume``, ``open_price``,
            and ``state`` fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        orders: list[dict] = self._request("GET", "/orders")
        if symbol is not None:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders

    def symbol_info(self, symbol: str) -> dict:
        """Return static and market-session information for *symbol*.

        Args:
            symbol: Instrument symbol (e.g. ``"USDJPY"``).

        Returns:
            Symbol info dict with fields such as ``digits``, ``point``,
            ``trade_contract_size``, ``volume_min``, ``volume_max``,
            ``volume_step``, and ``spread``.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("GET", f"/symbol/{symbol}")

    def order_send(self, request: dict) -> dict:
        """Place a new order or modify an existing one.

        Args:
            request: Order request dict compatible with the MT5
                ``MqlTradeRequest`` structure.  Must contain at minimum
                ``action``, ``symbol``, ``volume``, and ``type``.

        Returns:
            Result dict containing at minimum ``retcode``, ``ticket``,
            and ``comment`` fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("POST", "/order/place", json=request)

    def order_cancel(self, ticket: int) -> dict:
        """Cancel a pending order by ticket number.

        Args:
            ticket: Unique order ticket identifier.

        Returns:
            Result dict containing ``retcode`` and ``comment`` fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("POST", "/order/cancel", json={"ticket": ticket})

    def position_close(self, ticket: int) -> dict:
        """Close an open position at market price.

        Args:
            ticket: Unique position ticket identifier.

        Returns:
            Result dict containing ``retcode``, ``ticket``, and ``comment``
            fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request("POST", "/position/close", json={"ticket": ticket})

    def position_partial_close(self, ticket: int, volume: float) -> dict:
        """Partially close an open position.

        Args:
            ticket: Unique position ticket identifier.
            volume: Lot volume to close (must be less than the full position
                volume and a valid multiple of the symbol's volume step).

        Returns:
            Result dict containing ``retcode``, ``ticket``, and ``comment``
            fields.

        Raises:
            MT5ConnectionError: If the server cannot be reached.
        """
        return self._request(
            "POST",
            "/position/partial_close",
            json={"ticket": ticket, "volume": volume},
        )

    def is_connected(self) -> bool:
        """Check whether the bridge server is reachable and MT5 is connected.

        Calls :meth:`health_check` and inspects the ``mt5_connected`` flag.
        All exceptions are swallowed; a return value of ``False`` is used as
        the catch-all failure signal.

        Returns:
            ``True`` if the bridge responded and reported ``mt5_connected``
            as truthy, ``False`` in all other cases.
        """
        try:
            result = self.health_check()
            return bool(result.get("mt5_connected", False))
        except MT5ConnectionError:
            return False
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False
