"""FillMonitor — handles fill, partial-fill, and rejection callbacks from MT5."""

from __future__ import annotations

from typing import Optional

from core.utils.logger import get_logger
from core.utils.config import get_config
from core.risk import TradeOrder
from core.execution.mt5_connector import MT5Connector as MT5Client, MT5ConnectionError

logger = get_logger(__name__)

# Minimum fill ratio to accept a partial fill (70 %)
_MIN_PARTIAL_FILL_RATIO: float = 0.70


class FillMonitor:
    """Handles fill, partial-fill, and rejection events from MT5.

    Called by OrderManager after polling detects a position opened.
    Adjusts TP1/TP2 for slippage, then hands off to PositionManager.
    """

    def __init__(
        self,
        mt5_client: MT5Client,
        position_manager: Optional[object] = None,
        trade_journal: Optional[object] = None,
    ) -> None:
        """Store references.

        ``position_manager`` is injected after construction to break the
        circular dependency between FillMonitor and PositionManager.

        Args:
            mt5_client:       Connected MT5 bridge client.
            position_manager: PositionManager instance (may be set later).
            trade_journal:    Optional TradeJournal for audit logging.
        """
        self._mt5 = mt5_client
        self.position_manager = position_manager   # public so caller can set
        self._journal = trade_journal

        try:
            self._cfg = get_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FillMonitor: could not load config (%s) — "
                "pip size defaults will be used",
                exc,
            )
            self._cfg = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_fill(self, ticket: int, trade_order: TradeOrder) -> None:
        """Handle a confirmed fill event.

        Steps:
        1. Fetch live position details to retrieve the actual fill price.
        2. Calculate slippage in pips relative to the requested entry price.
        3. Adjust TP1/TP2 to preserve R:R: shift targets in the direction of
           the fill price.
        4. Notify PositionManager if set.
        5. Notify TradeJournal if set.
        6. Log fill summary.

        Args:
            ticket:      MT5 position ticket that was just opened.
            trade_order: Original TradeOrder (mutated in-place for TPs).
        """
        symbol = trade_order.symbol
        direction = trade_order.direction

        # ----------------------------------------------------------------
        # Step 1: Fetch actual fill price from live positions
        # ----------------------------------------------------------------
        fill_price: float = trade_order.entry_price  # safe fallback
        try:
            positions = self._mt5.positions_get(symbol)
            matched = [p for p in positions if p.get("ticket") == ticket]
            if matched:
                fill_price = float(matched[0].get("open_price", trade_order.entry_price))
            else:
                logger.warning(
                    "FillMonitor [%s]: ticket=%d not found in positions — "
                    "using requested entry price %.5f as fill price",
                    symbol,
                    ticket,
                    trade_order.entry_price,
                )
        except MT5ConnectionError as exc:
            logger.warning(
                "FillMonitor [%s]: MT5ConnectionError fetching fill price "
                "for ticket=%d — %s; using requested entry",
                symbol,
                ticket,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FillMonitor [%s]: unexpected error fetching fill price for "
                "ticket=%d — %s; using requested entry",
                symbol,
                ticket,
                exc,
            )

        # ----------------------------------------------------------------
        # Step 2: Calculate slippage
        # ----------------------------------------------------------------
        pip_size = self._get_pip_size(symbol)
        slippage_price = fill_price - trade_order.entry_price   # signed
        slippage_pips = abs(slippage_price) / pip_size if pip_size > 0 else 0.0

        # ----------------------------------------------------------------
        # Step 3: Adjust TP1 / TP2 to preserve R:R
        #
        # LONG: if fill > entry (paid more), TPs shift up by slippage so
        #       the profit distance from fill stays the same as planned.
        # SHORT: if fill < entry (received less), TPs shift down.
        # In both cases: tp_adjusted = tp_original + slippage_price_signed
        # which correctly handles adverse AND favourable fills.
        # ----------------------------------------------------------------
        if direction == "LONG":
            tp1_adjusted = trade_order.take_profit_1 + slippage_price
            tp2_adjusted = trade_order.take_profit_2 + slippage_price
        else:
            # SHORT: slippage_price is negative when fill < entry (worse fill)
            # shifting TPs down preserves the profit distance from fill price.
            tp1_adjusted = trade_order.take_profit_1 - slippage_price
            tp2_adjusted = trade_order.take_profit_2 - slippage_price

        trade_order.take_profit_1 = tp1_adjusted
        trade_order.take_profit_2 = tp2_adjusted

        logger.info(
            "FillMonitor [%s %s]: ticket=%d fill_price=%.5f "
            "requested_entry=%.5f slippage=%.1f pips "
            "tp1_adj=%.5f tp2_adj=%.5f",
            symbol,
            direction,
            ticket,
            fill_price,
            trade_order.entry_price,
            slippage_pips,
            tp1_adjusted,
            tp2_adjusted,
        )

        # ----------------------------------------------------------------
        # Step 4: Hand off to PositionManager
        # ----------------------------------------------------------------
        if self.position_manager is not None:
            try:
                self.position_manager.add_position(ticket, trade_order, fill_price)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "FillMonitor [%s]: PositionManager.add_position raised "
                    "an unexpected error for ticket=%d — %s",
                    symbol,
                    ticket,
                    exc,
                )

        # ----------------------------------------------------------------
        # Step 5: Notify TradeJournal
        # ----------------------------------------------------------------
        if self._journal is not None:
            try:
                self._journal.on_fill(ticket, fill_price, slippage_pips)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FillMonitor [%s]: TradeJournal.on_fill raised an "
                    "error for ticket=%d — %s",
                    symbol,
                    ticket,
                    exc,
                )

    def on_partial_fill(
        self,
        ticket: int,
        filled_volume: float,
        trade_order: TradeOrder,
    ) -> None:
        """Handle a partial fill event.

        Accepts the fill when ``filled_volume`` >= 70 % of the planned lot
        size, cancels the unfilled remainder, adjusts the order's lot_size
        to the actual filled amount, then delegates to :meth:`on_fill`.

        Rejects (cancels everything) when filled_volume < 70 %.

        Args:
            ticket:        MT5 ticket of the partially-filled order.
            filled_volume: Volume that was actually filled (lots).
            trade_order:   Original TradeOrder (lot_size mutated on accept).
        """
        symbol = trade_order.symbol
        min_acceptable = _MIN_PARTIAL_FILL_RATIO * trade_order.lot_size

        if filled_volume < min_acceptable:
            logger.warning(
                "FillMonitor [%s]: partial fill too small — "
                "filled=%.4f lots < min=%.4f lots (%.0f%% of planned) — "
                "cancelling ticket=%d",
                symbol,
                filled_volume,
                min_acceptable,
                _MIN_PARTIAL_FILL_RATIO * 100,
                ticket,
            )
            try:
                self._mt5.order_cancel(ticket)
            except MT5ConnectionError as exc:
                logger.error(
                    "FillMonitor [%s]: MT5ConnectionError cancelling "
                    "under-filled ticket=%d — %s",
                    symbol,
                    ticket,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "FillMonitor [%s]: unexpected error cancelling "
                    "under-filled ticket=%d — %s",
                    symbol,
                    ticket,
                    exc,
                )
            return

        # Accept: cancel any unfilled remainder
        logger.info(
            "FillMonitor [%s]: partial fill accepted — "
            "filled=%.4f / planned=%.4f lots (%.1f%%) — "
            "cancelling remainder for ticket=%d",
            symbol,
            filled_volume,
            trade_order.lot_size,
            (filled_volume / trade_order.lot_size) * 100,
            ticket,
        )

        try:
            self._mt5.order_cancel(ticket)
        except MT5ConnectionError as exc:
            logger.warning(
                "FillMonitor [%s]: MT5ConnectionError cancelling remainder "
                "of ticket=%d — %s; continuing with partial fill",
                symbol,
                ticket,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FillMonitor [%s]: unexpected error cancelling remainder of "
                "ticket=%d — %s; continuing with partial fill",
                symbol,
                ticket,
                exc,
            )

        # Adjust lot_size to the actual filled amount
        trade_order.lot_size = filled_volume

        # Delegate to normal fill handling
        self.on_fill(ticket, trade_order)

    def on_rejection(
        self,
        ticket: int,
        retcode: int,
        trade_order: TradeOrder,
    ) -> None:
        """Handle an MT5 order rejection.

        Logs the rejection details and does NOT retry — retries are a
        policy decision for the calling layer.

        Args:
            ticket:      MT5 ticket that was rejected.
            retcode:     MT5 return code explaining the rejection.
            trade_order: The order that was rejected.
        """
        logger.error(
            "FillMonitor [%s %s]: ORDER REJECTED — "
            "ticket=%d retcode=%d module=%s",
            trade_order.symbol,
            trade_order.direction,
            ticket,
            retcode,
            trade_order.module,
        )

        if self._journal is not None:
            try:
                if hasattr(self._journal, "on_rejection"):
                    self._journal.on_rejection(ticket, retcode, trade_order)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FillMonitor [%s]: TradeJournal.on_rejection raised an "
                    "error for ticket=%d — %s",
                    trade_order.symbol,
                    ticket,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pip_size(self, symbol: str) -> float:
        """Look up pip size for symbol from assets config.

        Args:
            symbol: Instrument symbol (e.g. ``"EURUSD"``).

        Returns:
            Pip size as a float.  Defaults to ``0.0001`` when the symbol
            is not found or config is unavailable.
        """
        if self._cfg is None:
            return 0.0001
        try:
            for asset in self._cfg.assets:
                if getattr(asset, "symbol", None) == symbol:
                    return float(asset.pip_size)
        except Exception:  # noqa: BLE001
            pass
        return 0.0001
