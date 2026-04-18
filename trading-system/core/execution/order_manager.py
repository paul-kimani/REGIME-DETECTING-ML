"""OrderManager — places, monitors, and cancels limit/market/stop-limit orders."""

from __future__ import annotations

import time
from typing import Optional

from core.utils.logger import get_logger
from core.risk import TradeOrder
from core.execution.mt5_connector import MT5Connector as MT5Client, MT5ConnectionError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MT5 constants (integer codes matching MetaTrader 5 API)
# ---------------------------------------------------------------------------
ORDER_TYPE_BUY            = 0
ORDER_TYPE_SELL           = 1
ORDER_TYPE_BUY_LIMIT      = 2
ORDER_TYPE_SELL_LIMIT     = 3
ORDER_TYPE_BUY_STOP       = 4
ORDER_TYPE_SELL_STOP      = 5
ORDER_TYPE_BUY_STOP_LIMIT = 6
ORDER_TYPE_SELL_STOP_LIMIT = 7

TRADE_ACTION_DEAL    = 1   # Immediate execution (market order)
TRADE_ACTION_PENDING = 5   # Place a pending order

TRADE_RETCODE_DONE = 10009  # MT5 success retcode

# Poll interval for monitor_pending (seconds)
_POLL_INTERVAL_SECONDS = 5

# M15 candle duration in seconds
_M15_CANDLE_SECONDS = 15 * 60


class OrderManager:
    """Places, monitors, and cancels limit/market/stop-limit orders via MT5.

    IMPORTANT: Orders are NEVER sent with SL or TP to MT5 (sl=0, tp=0 always).
    All stop management is handled by PositionManager in Python.
    """

    def __init__(
        self,
        mt5_client: MT5Client,
        trade_journal: Optional[object] = None,
    ) -> None:
        """Store client reference and optional journal.

        Args:
            mt5_client:    Connected MT5 bridge client.
            trade_journal: Optional TradeJournal for lifecycle logging.
        """
        self._mt5 = mt5_client
        self._journal = trade_journal

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_limit_order(self, trade_order: TradeOrder) -> Optional[int]:
        """Place a limit order for the given TradeOrder.

        Builds an MT5 pending-order request with sl=0 / tp=0 (stop management
        is handled entirely in Python by PositionManager).

        Args:
            trade_order: Fully-sized, validated order to execute.

        Returns:
            MT5 ticket integer on success, or ``None`` on failure.
        """
        order_type = self._build_order_type(trade_order.direction, "LIMIT")
        request = {
            "action":    TRADE_ACTION_PENDING,
            "symbol":    trade_order.symbol,
            "volume":    trade_order.lot_size,
            "type":      order_type,
            "price":     trade_order.entry_price,
            "sl":        0,
            "tp":        0,
            "deviation": 20,
            "magic":     trade_order.magic_number,
            "comment":   f"trading_system_{trade_order.module}",
        }

        logger.info(
            "OrderManager [%s %s]: placing LIMIT order — "
            "price=%.5f lots=%.4f magic=%d",
            trade_order.symbol,
            trade_order.direction,
            trade_order.entry_price,
            trade_order.lot_size,
            trade_order.magic_number,
        )

        try:
            result = self._mt5.order_send(request)
        except MT5ConnectionError as exc:
            logger.error(
                "OrderManager [%s %s]: MT5ConnectionError placing limit "
                "order — %s",
                trade_order.symbol,
                trade_order.direction,
                exc,
            )
            return None

        retcode = result.get("retcode")
        ticket = result.get("ticket")

        if retcode == TRADE_RETCODE_DONE and ticket:
            logger.info(
                "OrderManager [%s %s]: LIMIT order placed — ticket=%d",
                trade_order.symbol,
                trade_order.direction,
                ticket,
            )
            return int(ticket)

        logger.error(
            "OrderManager [%s %s]: LIMIT order failed — "
            "retcode=%s comment=%s",
            trade_order.symbol,
            trade_order.direction,
            retcode,
            result.get("comment", ""),
        )
        return None

    def place_market_order(self, trade_order: TradeOrder) -> Optional[int]:
        """Place a market order (fallback after limit expiry).

        Validates that the current market price is still within 0.5 ATR of
        the original entry price before sending.  Uses the live ask (LONG) or
        bid (SHORT) as the execution price.  sl=0 / tp=0 always.

        Args:
            trade_order: Original order; entry_price and atr used for
                         validation.

        Returns:
            MT5 ticket integer on success, or ``None`` on failure.
        """
        symbol = trade_order.symbol
        direction = trade_order.direction

        # Fetch live tick for price validation and execution price
        try:
            tick = self._mt5.symbol_info_tick(symbol)
        except MT5ConnectionError as exc:
            logger.error(
                "OrderManager [%s %s]: MT5ConnectionError fetching tick for "
                "market order — %s",
                symbol,
                direction,
                exc,
            )
            return None

        try:
            ask = float(tick["ask"])
            bid = float(tick["bid"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                "OrderManager [%s %s]: malformed tick data — %s",
                symbol,
                direction,
                exc,
            )
            return None

        exec_price = ask if direction == "LONG" else bid
        atr_threshold = trade_order.atr * 0.5

        if abs(exec_price - trade_order.entry_price) > atr_threshold:
            logger.warning(
                "OrderManager [%s %s]: market order aborted — current price "
                "%.5f deviates %.5f from entry %.5f (threshold %.5f / 0.5×ATR)",
                symbol,
                direction,
                exec_price,
                abs(exec_price - trade_order.entry_price),
                trade_order.entry_price,
                atr_threshold,
            )
            return None

        order_type = self._build_order_type(direction, "MARKET")
        request = {
            "action":    TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    trade_order.lot_size,
            "type":      order_type,
            "price":     exec_price,
            "sl":        0,
            "tp":        0,
            "deviation": 20,
            "magic":     trade_order.magic_number,
            "comment":   f"trading_system_{trade_order.module}",
        }

        logger.info(
            "OrderManager [%s %s]: placing MARKET order — "
            "price=%.5f lots=%.4f magic=%d",
            symbol,
            direction,
            exec_price,
            trade_order.lot_size,
            trade_order.magic_number,
        )

        try:
            result = self._mt5.order_send(request)
        except MT5ConnectionError as exc:
            logger.error(
                "OrderManager [%s %s]: MT5ConnectionError placing market "
                "order — %s",
                symbol,
                direction,
                exc,
            )
            return None

        retcode = result.get("retcode")
        ticket = result.get("ticket")

        if retcode == TRADE_RETCODE_DONE and ticket:
            logger.info(
                "OrderManager [%s %s]: MARKET order placed — ticket=%d "
                "exec_price=%.5f",
                symbol,
                direction,
                ticket,
                exec_price,
            )
            return int(ticket)

        logger.error(
            "OrderManager [%s %s]: MARKET order failed — "
            "retcode=%s comment=%s",
            symbol,
            direction,
            retcode,
            result.get("comment", ""),
        )
        return None

    def place_stop_limit_order(self, trade_order: TradeOrder) -> Optional[int]:
        """Place a stop-limit order for breakout entries.

        The ``price`` field acts as the stop trigger; MT5 converts the order
        to a limit at the same price when triggered.  sl=0 / tp=0 always.

        Args:
            trade_order: Order with entry_price as the stop trigger level.

        Returns:
            MT5 ticket integer on success, or ``None`` on failure.
        """
        order_type = self._build_order_type(trade_order.direction, "STOP_LIMIT")
        request = {
            "action":     TRADE_ACTION_PENDING,
            "symbol":     trade_order.symbol,
            "volume":     trade_order.lot_size,
            "type":       order_type,
            "price":      trade_order.entry_price,   # stop trigger
            "stoplimit":  trade_order.entry_price,   # limit price at trigger
            "sl":         0,
            "tp":         0,
            "deviation":  20,
            "magic":      trade_order.magic_number,
            "comment":    f"trading_system_{trade_order.module}",
        }

        logger.info(
            "OrderManager [%s %s]: placing STOP_LIMIT order — "
            "trigger=%.5f lots=%.4f magic=%d",
            trade_order.symbol,
            trade_order.direction,
            trade_order.entry_price,
            trade_order.lot_size,
            trade_order.magic_number,
        )

        try:
            result = self._mt5.order_send(request)
        except MT5ConnectionError as exc:
            logger.error(
                "OrderManager [%s %s]: MT5ConnectionError placing stop-limit "
                "order — %s",
                trade_order.symbol,
                trade_order.direction,
                exc,
            )
            return None

        retcode = result.get("retcode")
        ticket = result.get("ticket")

        if retcode == TRADE_RETCODE_DONE and ticket:
            logger.info(
                "OrderManager [%s %s]: STOP_LIMIT order placed — ticket=%d",
                trade_order.symbol,
                trade_order.direction,
                ticket,
            )
            return int(ticket)

        logger.error(
            "OrderManager [%s %s]: STOP_LIMIT order failed — "
            "retcode=%s comment=%s",
            trade_order.symbol,
            trade_order.direction,
            retcode,
            result.get("comment", ""),
        )
        return None

    def cancel_order(self, ticket: int) -> bool:
        """Cancel a pending order by ticket number.

        Args:
            ticket: MT5 order ticket to cancel.

        Returns:
            ``True`` on success (retcode 10009), ``False`` otherwise.
        """
        logger.info("OrderManager: cancelling order ticket=%d", ticket)
        try:
            result = self._mt5.order_cancel(ticket)
        except MT5ConnectionError as exc:
            logger.error(
                "OrderManager: MT5ConnectionError cancelling ticket=%d — %s",
                ticket,
                exc,
            )
            return False

        retcode = result.get("retcode")
        if retcode == TRADE_RETCODE_DONE:
            logger.info(
                "OrderManager: order ticket=%d cancelled successfully", ticket
            )
            return True

        logger.error(
            "OrderManager: cancel failed for ticket=%d — "
            "retcode=%s comment=%s",
            ticket,
            retcode,
            result.get("comment", ""),
        )
        return False

    # ------------------------------------------------------------------
    # Order monitoring
    # ------------------------------------------------------------------

    def monitor_pending(
        self,
        trade_order: TradeOrder,
        ticket: int,
        expiry_candles: int = 3,
    ) -> str:
        """Poll until the pending order fills, expires, or is cancelled.

        Polls every 5 seconds.  Expiry is measured in M15 candles elapsed
        since this method was called.

        For ``momentum`` / ``breakout`` modules a market-order fallback is
        attempted when the order expires and the price is still within
        0.5 ATR.  For ``mean_reversion`` the order simply expires without
        a fallback (no_market_fallback).

        Args:
            trade_order:    Original order (symbol, module, atr used).
            ticket:         MT5 ticket of the pending order to watch.
            expiry_candles: Number of M15 candles after which the order is
                            considered stale and cancelled.  Defaults to 3.

        Returns:
            One of ``"filled"``, ``"expired"``, ``"cancelled"``, ``"error"``.
        """
        symbol = trade_order.symbol
        module = trade_order.module
        expiry_seconds = expiry_candles * _M15_CANDLE_SECONDS
        start_time = time.monotonic()

        logger.info(
            "OrderManager [%s]: monitoring ticket=%d module=%s "
            "expiry=%d candles",
            symbol,
            ticket,
            module,
            expiry_candles,
        )

        while True:
            time.sleep(_POLL_INTERVAL_SECONDS)
            elapsed = time.monotonic() - start_time

            try:
                # --------------------------------------------------------
                # Check 1: Has this ticket moved to open positions?
                # --------------------------------------------------------
                positions = self._mt5.positions_get(symbol)
                position_tickets = {p.get("ticket") for p in positions}
                if ticket in position_tickets:
                    logger.info(
                        "OrderManager [%s]: ticket=%d FILLED after %.0fs",
                        symbol,
                        ticket,
                        elapsed,
                    )
                    return "filled"

                # --------------------------------------------------------
                # Check 2: Is the ticket still in pending orders?
                # --------------------------------------------------------
                pending = self._mt5.orders_get(symbol)
                pending_tickets = {o.get("ticket") for o in pending}
                if ticket not in pending_tickets:
                    logger.info(
                        "OrderManager [%s]: ticket=%d no longer in pending "
                        "orders — CANCELLED externally after %.0fs",
                        symbol,
                        ticket,
                        elapsed,
                    )
                    return "cancelled"

            except MT5ConnectionError as exc:
                logger.error(
                    "OrderManager [%s]: MT5ConnectionError while monitoring "
                    "ticket=%d — %s",
                    symbol,
                    ticket,
                    exc,
                )
                return "error"

            # ----------------------------------------------------------------
            # Check 3: Expiry
            # ----------------------------------------------------------------
            elapsed_candles = elapsed / _M15_CANDLE_SECONDS
            if elapsed_candles >= expiry_candles:
                logger.info(
                    "OrderManager [%s]: ticket=%d expired after %.1f candles "
                    "(%.0fs) — cancelling",
                    symbol,
                    ticket,
                    elapsed_candles,
                    elapsed,
                )
                self.cancel_order(ticket)

                # Mean-reversion: never fall back to market
                no_fallback = module == "mean_reversion"
                if no_fallback:
                    logger.info(
                        "OrderManager [%s]: mean_reversion module — "
                        "no market fallback, returning 'expired'",
                        symbol,
                    )
                    return "expired"

                # Momentum / breakout: attempt market fallback
                logger.info(
                    "OrderManager [%s]: attempting market fallback for "
                    "module=%s",
                    symbol,
                    module,
                )
                fallback_ticket = self.place_market_order(trade_order)
                if fallback_ticket is not None:
                    logger.info(
                        "OrderManager [%s]: market fallback placed — "
                        "new ticket=%d",
                        symbol,
                        fallback_ticket,
                    )
                    # Update the caller's view: the new ticket is now "filled"
                    # pending confirmation, but we return "expired" so the
                    # caller can route through on_fill with the new ticket.
                    return "expired"

                logger.warning(
                    "OrderManager [%s]: market fallback rejected (price moved "
                    "too far) — returning 'expired'",
                    symbol,
                )
                return "expired"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_order_type(self, direction: str, order_type: str) -> int:
        """Map direction + order_type string to MT5 ORDER_TYPE constant.

        Args:
            direction:  ``"LONG"`` or ``"SHORT"``.
            order_type: ``"MARKET"``, ``"LIMIT"``, ``"STOP"``, or
                        ``"STOP_LIMIT"``.

        Returns:
            Integer MT5 order type code.

        Raises:
            ValueError: If direction or order_type is not recognised.
        """
        mapping: dict[tuple[str, str], int] = {
            ("LONG",  "MARKET"):     ORDER_TYPE_BUY,
            ("SHORT", "MARKET"):     ORDER_TYPE_SELL,
            ("LONG",  "LIMIT"):      ORDER_TYPE_BUY_LIMIT,
            ("SHORT", "LIMIT"):      ORDER_TYPE_SELL_LIMIT,
            ("LONG",  "STOP"):       ORDER_TYPE_BUY_STOP,
            ("SHORT", "STOP"):       ORDER_TYPE_SELL_STOP,
            ("LONG",  "STOP_LIMIT"): ORDER_TYPE_BUY_STOP_LIMIT,
            ("SHORT", "STOP_LIMIT"): ORDER_TYPE_SELL_STOP_LIMIT,
        }
        key = (direction.upper(), order_type.upper())
        if key not in mapping:
            raise ValueError(
                f"Unknown order type combination: direction={direction!r} "
                f"order_type={order_type!r}"
            )
        return mapping[key]
