"""Final pre-execution validation immediately before sending orders to MT5."""

from __future__ import annotations

from typing import Optional

from core.utils.logger import get_logger
from core.utils.config import get_config
from core.risk import TradeOrder
from core.execution.mt5_connector import MT5Connector as MT5Client, MT5ConnectionError

logger = get_logger(__name__)

# Maximum distance between requested entry and current bid/ask before the
# price is considered stale (expressed in pips).
_MAX_ENTRY_DEVIATION_PIPS: float = 50.0

# Spread tolerance multiplier — wider than pre-trade because this runs on
# real-time tick data immediately before placement.
_SPREAD_TOLERANCE_MULTIPLIER: float = 2.0


class PreExecutionValidator:
    """Final validation gate immediately before an order touches MT5.

    Runs synchronously at order-placement time. Any failure prevents
    the order from being sent. Unlike PreTradeChecker (which runs on
    signal generation), this runs on the current live market state.
    """

    def __init__(self, mt5_client: MT5Client) -> None:
        """Store mt5_client reference and load config.

        Args:
            mt5_client: Connected MT5 bridge client instance.
        """
        self._mt5 = mt5_client
        try:
            self._cfg = get_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PreExecutionValidator: could not load config (%s) — "
                "asset defaults will be used",
                exc,
            )
            self._cfg = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        trade_order: TradeOrder,
        regime_state: Optional[object] = None,
    ) -> tuple[bool, list[str]]:
        """Run all final checks before the order is sent to MT5.

        Checks (in order):
        1. mt5_connection        — bridge must be connected.
        2. price_still_valid     — entry price within 50 pips of current tick.
        3. spread_still_ok       — spread <= typical_spread * 2.0.
        4. no_conflicting_orders — no pending order in same direction.
        5. no_conflicting_positions — no open position in same direction.
        6. regime_still_intact   — regime sizing multiplier > 0 (if provided).

        Args:
            trade_order:  The fully-sized order about to be placed.
            regime_state: Optional live regime context; check 6 is skipped
                          when ``None``.

        Returns:
            ``(True, [])`` when all checks pass, or
            ``(False, [failure_name, ...])`` on the first (or multiple)
            failures.

        Raises:
            Nothing — MT5ConnectionError is caught internally.
        """
        failures: list[str] = []
        symbol = trade_order.symbol
        direction = trade_order.direction

        # ----------------------------------------------------------------
        # Check 1: MT5 connection
        # ----------------------------------------------------------------
        try:
            if not self._mt5.is_connected():
                logger.error(
                    "PreExecutionValidator [%s]: MT5 not connected", symbol
                )
                return False, ["mt5_connection"]
        except MT5ConnectionError:
            logger.error(
                "PreExecutionValidator [%s]: MT5ConnectionError during "
                "connection check",
                symbol,
            )
            return False, ["mt5_connection_failed"]

        # ----------------------------------------------------------------
        # Fetch live tick once — used by checks 2 and 3
        # ----------------------------------------------------------------
        tick: Optional[dict] = None
        try:
            tick = self._mt5.symbol_info_tick(symbol)
        except MT5ConnectionError as exc:
            logger.error(
                "PreExecutionValidator [%s]: MT5ConnectionError fetching "
                "tick — %s",
                symbol,
                exc,
            )
            return False, ["mt5_connection_failed"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PreExecutionValidator [%s]: unexpected error fetching tick "
                "— %s; skipping price/spread checks",
                symbol,
                exc,
            )

        # ----------------------------------------------------------------
        # Check 2: Price still valid
        # ----------------------------------------------------------------
        if tick is not None:
            price_ok, price_reason = self._check_price_valid(trade_order, tick)
            if not price_ok:
                logger.warning(
                    "PreExecutionValidator [%s]: price check failed — %s",
                    symbol,
                    price_reason,
                )
                failures.append("price_still_valid")

            # Check 3: Spread still OK
            spread_ok, spread_reason = self._check_spread(trade_order, tick)
            if not spread_ok:
                logger.warning(
                    "PreExecutionValidator [%s]: spread check failed — %s",
                    symbol,
                    spread_reason,
                )
                failures.append("spread_still_ok")

        # ----------------------------------------------------------------
        # Check 4: No conflicting pending orders
        # ----------------------------------------------------------------
        try:
            pending_orders = self._mt5.orders_get(symbol)
            direction_type = self._direction_to_order_types(direction)
            conflicting = [
                o for o in pending_orders
                if o.get("type") in direction_type
            ]
            if conflicting:
                tickets = [o.get("ticket") for o in conflicting]
                logger.warning(
                    "PreExecutionValidator [%s]: conflicting pending order(s) "
                    "in same direction (%s): %s",
                    symbol,
                    direction,
                    tickets,
                )
                failures.append("no_conflicting_orders")
        except MT5ConnectionError as exc:
            logger.warning(
                "PreExecutionValidator [%s]: MT5ConnectionError checking "
                "pending orders — %s; skipping",
                symbol,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PreExecutionValidator [%s]: unexpected error checking "
                "pending orders — %s; skipping",
                symbol,
                exc,
            )

        # ----------------------------------------------------------------
        # Check 5: No conflicting open positions
        # ----------------------------------------------------------------
        try:
            open_positions = self._mt5.positions_get(symbol)
            # MT5 position type: 0 = BUY, 1 = SELL
            target_pos_type = 0 if direction == "LONG" else 1
            conflicting_pos = [
                p for p in open_positions
                if p.get("type") == target_pos_type
            ]
            if conflicting_pos:
                pos_tickets = [p.get("ticket") for p in conflicting_pos]
                logger.warning(
                    "PreExecutionValidator [%s]: conflicting open position(s) "
                    "in same direction (%s): %s",
                    symbol,
                    direction,
                    pos_tickets,
                )
                failures.append("no_conflicting_positions")
        except MT5ConnectionError as exc:
            logger.warning(
                "PreExecutionValidator [%s]: MT5ConnectionError checking "
                "open positions — %s; skipping",
                symbol,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PreExecutionValidator [%s]: unexpected error checking "
                "open positions — %s; skipping",
                symbol,
                exc,
            )

        # ----------------------------------------------------------------
        # Check 6: Regime still intact
        # ----------------------------------------------------------------
        if regime_state is not None:
            try:
                sizing_mult = float(
                    getattr(regime_state, "final_sizing_multiplier", 1.0)
                )
                if sizing_mult <= 0:
                    logger.warning(
                        "PreExecutionValidator [%s]: regime sizing multiplier "
                        "is %.4f — regime no longer intact",
                        symbol,
                        sizing_mult,
                    )
                    failures.append("regime_still_intact")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PreExecutionValidator [%s]: could not read regime "
                    "sizing multiplier — %s; skipping",
                    symbol,
                    exc,
                )

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        if failures:
            logger.info(
                "PreExecutionValidator [%s %s]: FAILED checks: %s",
                symbol,
                direction,
                failures,
            )
            return False, failures

        logger.info(
            "PreExecutionValidator [%s %s]: all checks passed",
            symbol,
            direction,
        )
        return True, []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_price_valid(
        self,
        trade_order: TradeOrder,
        tick: dict,
    ) -> tuple[bool, str]:
        """Check entry_price is within 50 pips of current bid/ask.

        For LONG orders the reference price is the ask; for SHORT orders
        it is the bid.  Distance is measured in pips using the asset's
        pip_size from config.

        Args:
            trade_order: Order containing the requested entry price.
            tick:        Current live tick dict with ``bid`` and ``ask``.

        Returns:
            ``(True, "")`` when the price is still valid, or
            ``(False, reason_string)`` when it has moved too far.
        """
        symbol = trade_order.symbol
        pip_size = self._get_pip_size(symbol)

        try:
            if trade_order.direction == "LONG":
                ref_price = float(tick["ask"])
            else:
                ref_price = float(tick["bid"])
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"tick missing bid/ask: {exc}"

        deviation_price = abs(trade_order.entry_price - ref_price)
        if pip_size > 0:
            deviation_pips = deviation_price / pip_size
        else:
            deviation_pips = 0.0

        if deviation_pips > _MAX_ENTRY_DEVIATION_PIPS:
            return (
                False,
                f"entry {trade_order.entry_price:.5f} is {deviation_pips:.1f} pips "
                f"from current {'ask' if trade_order.direction == 'LONG' else 'bid'} "
                f"{ref_price:.5f} (max {_MAX_ENTRY_DEVIATION_PIPS:.0f} pips)",
            )

        return True, ""

    def _check_spread(
        self,
        trade_order: TradeOrder,
        tick: dict,
    ) -> tuple[bool, str]:
        """Check spread <= typical_spread * 2.0.

        Typical spread is looked up from the assets config for the
        order's symbol.  Falls back to a generous 10-pip default when the
        symbol is not found so as not to block the order unnecessarily.

        Args:
            trade_order: Order (symbol used for config lookup).
            tick:        Current live tick dict with ``spread``, or
                         alternatively ``ask`` and ``bid`` so the spread
                         can be derived.

        Returns:
            ``(True, "")`` when spread is acceptable, or
            ``(False, reason_string)`` when it is too wide.
        """
        symbol = trade_order.symbol
        typical_spread = self._get_typical_spread(symbol)
        max_spread = typical_spread * _SPREAD_TOLERANCE_MULTIPLIER

        # Prefer the explicit spread field; fall back to ask - bid
        try:
            current_spread = float(tick.get("spread", 0.0))
            if current_spread == 0.0:
                ask = float(tick.get("ask", 0.0))
                bid = float(tick.get("bid", 0.0))
                current_spread = ask - bid
        except (TypeError, ValueError) as exc:
            return False, f"could not read spread from tick: {exc}"

        if current_spread > max_spread:
            return (
                False,
                f"current spread {current_spread:.5f} exceeds max allowed "
                f"{max_spread:.5f} (typical {typical_spread:.5f} × "
                f"{_SPREAD_TOLERANCE_MULTIPLIER})",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Config lookups
    # ------------------------------------------------------------------

    def _get_pip_size(self, symbol: str) -> float:
        """Return pip_size for symbol from assets config. Default 0.0001."""
        if self._cfg is None:
            return 0.0001
        try:
            for asset in self._cfg.assets:
                if getattr(asset, "symbol", None) == symbol:
                    return float(asset.pip_size)
        except Exception:  # noqa: BLE001
            pass
        return 0.0001

    def _get_typical_spread(self, symbol: str) -> float:
        """Return typical_spread for symbol from assets config. Default 0.0002."""
        if self._cfg is None:
            return 0.0002
        try:
            for asset in self._cfg.assets:
                if getattr(asset, "symbol", None) == symbol:
                    return float(asset.typical_spread)
        except Exception:  # noqa: BLE001
            pass
        return 0.0002

    def _direction_to_order_types(self, direction: str) -> list[int]:
        """Map a trade direction to MT5 order type integers for conflict checks.

        BUY_LIMIT=2, BUY_STOP=4, BUY_STOP_LIMIT=6 for LONG;
        SELL_LIMIT=3, SELL_STOP=5, SELL_STOP_LIMIT=7 for SHORT.

        Args:
            direction: ``"LONG"`` or ``"SHORT"``.

        Returns:
            List of MT5 order type integers that represent the same direction.
        """
        if direction == "LONG":
            return [2, 4, 6]   # BUY_LIMIT, BUY_STOP, BUY_STOP_LIMIT
        return [3, 5, 7]       # SELL_LIMIT, SELL_STOP, SELL_STOP_LIMIT
