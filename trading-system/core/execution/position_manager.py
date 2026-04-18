"""PositionManager — Python-side stop management, trailing, TP1/TP2, regime invalidation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.utils.logger import get_logger
from core.risk import TradeOrder
from core.execution.mt5_connector import MT5ConnectionError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MT5 constants
# ---------------------------------------------------------------------------
TRADE_ACTION_SLTP = 6   # Modify SL/TP on an existing position
TRADE_RETCODE_DONE = 10009

# Monitoring loop cadence
_POLL_INTERVAL_SECONDS = 30

# Emergency stop ATR multiple for orphaned positions
_ORPHAN_EMERGENCY_ATR_MULTIPLE = 2.0


# ---------------------------------------------------------------------------
# PositionState
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Live state of a managed position."""

    ticket: int
    symbol: str
    direction: str           # LONG | SHORT
    module: str              # MOMENTUM | MEAN_REVERSION | BREAKOUT
    lot_size: float
    entry_price: float
    fill_price: float
    current_stop: float      # managed in Python, NOT set in MT5
    tp1: float
    tp2: float
    atr_at_entry: float
    timeframe: str
    magic_number: int
    open_time: datetime
    tp1_hit: bool = False
    trailing_active: bool = False
    regime_at_entry: str = "RANGE"
    hold_time_limit: int = 12  # candles
    max_hold_candles: int = 12


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------

class PositionManager:
    """Python-managed position monitoring. Runs in a dedicated background thread.

    Polls every 30 seconds for intracandle checks (stop/TP).
    Reacts to candle close events for trailing stops, regime checks, time expiry.

    CRITICAL: MT5 never has SL/TP set. This class is the sole authority
    on when positions close. On startup it scans for orphaned positions
    and sets emergency MT5 SL as fallback.
    """

    def __init__(
        self,
        mt5_client,
        stop_target_engine,
        circuit_breaker,
        trade_journal=None,
        regime_detector=None,
    ) -> None:
        """Initialise with all required dependencies.

        Args:
            mt5_client:         Connected MT5 bridge client.
            stop_target_engine: StopTargetEngine for trailing stop computation.
            circuit_breaker:    CircuitBreaker instance for halt detection.
            trade_journal:      Optional TradeJournal for lifecycle logging.
            regime_detector:    Optional RegimeDetector for regime invalidation.
        """
        self._positions: dict[int, PositionState] = {}  # ticket -> PositionState
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self._mt5 = mt5_client
        self._stop_target_engine = stop_target_engine
        self._circuit_breaker = circuit_breaker
        self._journal = trade_journal
        self._regime_detector = regime_detector

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread. Scan for orphaned positions first."""
        logger.info("PositionManager: starting background monitor thread")

        try:
            self._scan_orphaned_positions()
        except Exception as exc:
            logger.error(
                "PositionManager: orphan scan raised an unexpected error — %s", exc
            )

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="position-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("PositionManager: monitor thread started (pid-level daemon)")

    def stop(self) -> None:
        """Signal monitoring thread to stop. Wait for graceful exit (timeout=10s)."""
        logger.info("PositionManager: stopping monitor thread")
        self._stop_event.set()

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
            if self._monitor_thread.is_alive():
                logger.warning(
                    "PositionManager: monitor thread did not exit within 10s"
                )
            else:
                logger.info("PositionManager: monitor thread stopped cleanly")

        self._monitor_thread = None

    # ------------------------------------------------------------------
    # Position registry
    # ------------------------------------------------------------------

    def add_position(self, ticket: int, trade_order: TradeOrder, fill_price: float) -> None:
        """Register a newly filled position.

        Removes any emergency MT5 SL that was set (sl=0 via order modify)
        since Python is now managing the stop.

        Args:
            ticket:      MT5 ticket of the filled position.
            trade_order: Fully-populated TradeOrder for this fill.
            fill_price:  Actual fill price from MT5.
        """
        # Determine regime at entry
        regime_at_entry = "RANGE"
        if trade_order.regime_context is not None:
            regime_at_entry = str(
                getattr(trade_order.regime_context, "m15_regime", "RANGE")
            )

        pos = PositionState(
            ticket=ticket,
            symbol=trade_order.symbol,
            direction=trade_order.direction,
            module=trade_order.module,
            lot_size=trade_order.lot_size,
            entry_price=trade_order.entry_price,
            fill_price=fill_price,
            current_stop=trade_order.stop_loss,
            tp1=trade_order.take_profit_1,
            tp2=trade_order.take_profit_2,
            atr_at_entry=trade_order.atr,
            timeframe=trade_order.timeframe,
            magic_number=trade_order.magic_number,
            open_time=datetime.now(timezone.utc),
            regime_at_entry=regime_at_entry,
        )

        with self._lock:
            self._positions[ticket] = pos

        logger.info(
            "PositionManager: registered position ticket=%d %s %s "
            "fill=%.5f sl=%.5f tp1=%.5f tp2=%.5f",
            ticket,
            trade_order.symbol,
            trade_order.direction,
            fill_price,
            trade_order.stop_loss,
            trade_order.take_profit_1,
            trade_order.take_profit_2,
        )

        # Remove any emergency MT5 SL that was set during orphan scan
        self._modify_sl_in_mt5(ticket, sl=0)

    def remove_position(self, ticket: int) -> None:
        """Remove position from tracking after close.

        Args:
            ticket: MT5 ticket to deregister.
        """
        with self._lock:
            removed = self._positions.pop(ticket, None)

        if removed is not None:
            logger.info("PositionManager: deregistered ticket=%d", ticket)
        else:
            logger.debug(
                "PositionManager: remove_position called for unknown ticket=%d", ticket
            )

    def get_open_positions(self) -> list[PositionState]:
        """Return list of all PositionState objects (snapshot copy).

        Returns:
            List of currently tracked :class:`PositionState` objects.
        """
        with self._lock:
            return list(self._positions.values())

    # ------------------------------------------------------------------
    # Closing helpers
    # ------------------------------------------------------------------

    def close_position(self, ticket: int, reason: str) -> bool:
        """Close full position at market.

        Args:
            ticket: MT5 ticket to close.
            reason: Human-readable close reason for logging and journal.

        Returns:
            True on success, False on failure.
        """
        with self._lock:
            pos = self._positions.get(ticket)

        if pos is None:
            logger.warning(
                "PositionManager.close_position: ticket=%d not tracked — skipping",
                ticket,
            )
            return False

        logger.info(
            "PositionManager: closing ticket=%d %s %s reason=%s",
            ticket,
            pos.symbol,
            pos.direction,
            reason,
        )

        try:
            result = self._mt5.position_close(ticket)
            retcode = result.get("retcode")

            if retcode == TRADE_RETCODE_DONE:
                exit_price = float(result.get("price", 0.0))
                logger.info(
                    "PositionManager: ticket=%d closed — reason=%s exit_price=%.5f",
                    ticket,
                    reason,
                    exit_price,
                )

                if self._journal is not None:
                    try:
                        self._journal.on_close(
                            ticket=ticket,
                            exit_price=exit_price,
                            exit_reason=reason,
                            entry_price=pos.entry_price,
                            lot_size=pos.lot_size,
                            stop_loss=pos.current_stop,
                            direction=pos.direction,
                            entry_time=pos.open_time,
                        )
                    except Exception as exc:
                        logger.error(
                            "PositionManager: journal.on_close raised for "
                            "ticket=%d — %s",
                            ticket,
                            exc,
                        )

                self.remove_position(ticket)
                return True

            logger.error(
                "PositionManager: close_position failed ticket=%d "
                "retcode=%s comment=%s",
                ticket,
                retcode,
                result.get("comment", ""),
            )
            return False

        except MT5ConnectionError as exc:
            logger.error(
                "PositionManager: MT5ConnectionError closing ticket=%d — %s",
                ticket,
                exc,
            )
            return False

    def partial_close(self, ticket: int, pct: float, reason: str) -> bool:
        """Close pct% of position at market.

        Args:
            ticket: MT5 ticket to partially close.
            pct:    Fraction to close, e.g. 0.50 closes half.
            reason: Human-readable reason for logging and journal.

        Returns:
            True on success, False on failure.
        """
        with self._lock:
            pos = self._positions.get(ticket)

        if pos is None:
            logger.warning(
                "PositionManager.partial_close: ticket=%d not tracked — skipping",
                ticket,
            )
            return False

        close_volume = round(pos.lot_size * pct, 4)
        if close_volume <= 0:
            logger.warning(
                "PositionManager.partial_close: computed volume=%.4f <= 0 for "
                "ticket=%d — skipping",
                close_volume,
                ticket,
            )
            return False

        logger.info(
            "PositionManager: partial close ticket=%d %.0f%% volume=%.4f reason=%s",
            ticket,
            pct * 100,
            close_volume,
            reason,
        )

        try:
            result = self._mt5.position_partial_close(ticket, close_volume)
            retcode = result.get("retcode")

            if retcode == TRADE_RETCODE_DONE:
                close_price = float(result.get("price", 0.0))
                logger.info(
                    "PositionManager: partial close done ticket=%d "
                    "volume=%.4f price=%.5f",
                    ticket,
                    close_volume,
                    close_price,
                )

                if self._journal is not None:
                    try:
                        self._journal.on_partial_close(
                            ticket=ticket,
                            volume=close_volume,
                            price=close_price,
                            reason=reason,
                        )
                    except Exception as exc:
                        logger.error(
                            "PositionManager: journal.on_partial_close raised "
                            "for ticket=%d — %s",
                            ticket,
                            exc,
                        )

                # Update tracked lot size
                with self._lock:
                    if ticket in self._positions:
                        self._positions[ticket].lot_size = round(
                            pos.lot_size - close_volume, 4
                        )

                return True

            logger.error(
                "PositionManager: partial_close failed ticket=%d "
                "retcode=%s comment=%s",
                ticket,
                retcode,
                result.get("comment", ""),
            )
            return False

        except MT5ConnectionError as exc:
            logger.error(
                "PositionManager: MT5ConnectionError partial_close ticket=%d — %s",
                ticket,
                exc,
            )
            return False

    def emergency_close_all(self, reason: str) -> None:
        """Close all tracked positions immediately.

        Args:
            reason: Human-readable reason applied to every close call.
        """
        logger.critical(
            "PositionManager: EMERGENCY CLOSE ALL — reason=%s", reason
        )

        with self._lock:
            tickets = list(self._positions.keys())

        for ticket in tickets:
            try:
                self.close_position(ticket, reason)
            except Exception as exc:
                logger.error(
                    "PositionManager: emergency_close_all failed for "
                    "ticket=%d — %s",
                    ticket,
                    exc,
                )

    # ------------------------------------------------------------------
    # Candle close hook
    # ------------------------------------------------------------------

    def on_candle_close(
        self,
        symbol: str,
        timeframe: str,
        features=None,
        regime_state=None,
    ) -> None:
        """Called on every candle close. Runs candle-level checks for all positions.

        Filters to positions matching *symbol* and *timeframe*, then calls
        :meth:`_candle_close_checks` for each.

        Args:
            symbol:       Symbol whose candle just closed.
            timeframe:    Timeframe of the closed candle.
            features:     Optional feature DataFrame for the current bar.
            regime_state: Optional current RegimeState for the symbol.
        """
        with self._lock:
            relevant = [
                pos
                for pos in self._positions.values()
                if pos.symbol == symbol and pos.timeframe == timeframe
            ]

        for pos in relevant:
            try:
                self._candle_close_checks(pos, features=features, regime_state=regime_state)
            except Exception as exc:
                logger.error(
                    "PositionManager: _candle_close_checks raised for "
                    "ticket=%d — %s",
                    pos.ticket,
                    exc,
                )

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Main polling loop. Runs every 30 seconds.

        Priority order per position:
        1. Hard stop check: fetch live price; if crossed stop -> close("stop_hit").
        2. Circuit breaker: if halted -> emergency_close_all("circuit_breaker").
        3. TP1 check: if price crosses tp1 -> partial_close(50%), move stop to
           breakeven, set trailing_active=True, tp1_hit=True.
        4. TP2 check: if tp1_hit and price crosses tp2 -> close("tp2_hit").
        """
        logger.info("PositionManager._monitor_loop: started")

        while not self._stop_event.is_set():
            try:
                self._run_poll_cycle()
            except Exception as exc:
                logger.error(
                    "PositionManager._monitor_loop: unhandled exception — %s; "
                    "continuing",
                    exc,
                )

            # Sleep in short chunks so stop_event is responsive
            for _ in range(_POLL_INTERVAL_SECONDS):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("PositionManager._monitor_loop: exiting")

    def _run_poll_cycle(self) -> None:
        """Execute one full polling cycle across all tracked positions."""
        with self._lock:
            tickets = list(self._positions.keys())

        for ticket in tickets:
            # Re-read under lock each iteration — position may have been closed
            with self._lock:
                pos = self._positions.get(ticket)

            if pos is None:
                continue

            try:
                self._check_single_position(pos)
            except Exception as exc:
                logger.error(
                    "PositionManager: poll check raised for ticket=%d — %s",
                    ticket,
                    exc,
                )

    def _check_single_position(self, pos: PositionState) -> None:
        """Run all intracandle priority checks for one position.

        Args:
            pos: The :class:`PositionState` to evaluate.
        """
        ticket = pos.ticket

        # ---- 1. Hard stop check ----------------------------------------
        try:
            live_price = self._get_live_price(pos.symbol, pos.direction)
        except Exception as exc:
            logger.warning(
                "PositionManager: could not fetch live price for ticket=%d "
                "%s — %s; skipping stop check",
                ticket,
                pos.symbol,
                exc,
            )
            return

        stop_hit = (
            (pos.direction == "LONG" and live_price <= pos.current_stop)
            or (pos.direction == "SHORT" and live_price >= pos.current_stop)
        )

        if stop_hit:
            logger.info(
                "PositionManager: STOP HIT ticket=%d %s %s "
                "price=%.5f stop=%.5f",
                ticket,
                pos.symbol,
                pos.direction,
                live_price,
                pos.current_stop,
            )
            self.close_position(ticket, "stop_hit")
            return  # position closed; nothing more to do

        # ---- 2. Circuit breaker ----------------------------------------
        try:
            account_info = self._mt5.account_info()
            cb_level, cb_desc = self._circuit_breaker.check(
                account_state={
                    "balance": account_info.get("balance", 0.0),
                    "equity": account_info.get("equity", 0.0),
                    "daily_loss_pct": account_info.get("daily_loss_pct", 0.0),
                    "weekly_pnl_pct": account_info.get("weekly_pnl_pct", 0.0),
                    "peak_equity": account_info.get("peak_equity",
                                                    account_info.get("equity", 0.0)),
                    "consecutive_losses": account_info.get("consecutive_losses", 0),
                    "global_risk_state": account_info.get("global_risk_state", "RISK_ON"),
                    "avg_spread_ratio": account_info.get("avg_spread_ratio", 1.0),
                    "mt5_connected": True,
                },
                recent_trades=[],
                open_positions=self.get_open_positions(),
            )

            if self._circuit_breaker.is_trading_halted(cb_level):
                logger.warning(
                    "PositionManager: circuit breaker level=%d — %s; "
                    "emergency closing all positions",
                    cb_level,
                    cb_desc,
                )
                self.emergency_close_all("circuit_breaker")
                return

        except MT5ConnectionError as exc:
            logger.warning(
                "PositionManager: MT5ConnectionError fetching account info "
                "for CB check — %s; skipping CB check this cycle",
                exc,
            )
        except Exception as exc:
            logger.error(
                "PositionManager: circuit breaker check failed for "
                "ticket=%d — %s",
                ticket,
                exc,
            )

        # Re-read pos — it may have been removed by emergency_close_all
        with self._lock:
            pos = self._positions.get(ticket)

        if pos is None:
            return

        # ---- 3. TP1 check -----------------------------------------------
        if not pos.tp1_hit:
            tp1_reached = (
                (pos.direction == "LONG" and live_price >= pos.tp1)
                or (pos.direction == "SHORT" and live_price <= pos.tp1)
            )

            if tp1_reached:
                logger.info(
                    "PositionManager: TP1 HIT ticket=%d price=%.5f tp1=%.5f",
                    ticket,
                    live_price,
                    pos.tp1,
                )

                # Partial close 50 %
                self.partial_close(ticket, 0.50, "tp1_hit")

                # Move stop to breakeven and activate trailing
                with self._lock:
                    if ticket in self._positions:
                        old_stop = self._positions[ticket].current_stop
                        self._positions[ticket].current_stop = pos.entry_price
                        self._positions[ticket].trailing_active = True
                        self._positions[ticket].tp1_hit = True

                if self._journal is not None:
                    try:
                        self._journal.on_position_event(
                            ticket=ticket,
                            event_type="stop_to_breakeven",
                            details={
                                "old_stop": old_stop,
                                "new_stop": pos.entry_price,
                                "trigger": "tp1_hit",
                                "price": live_price,
                            },
                        )
                    except Exception as exc:
                        logger.error(
                            "PositionManager: journal.on_position_event raised "
                            "for ticket=%d — %s",
                            ticket,
                            exc,
                        )

                return  # TP1 handled; re-evaluate next poll

        # ---- 4. TP2 check -----------------------------------------------
        if pos.tp1_hit:
            tp2_reached = (
                (pos.direction == "LONG" and live_price >= pos.tp2)
                or (pos.direction == "SHORT" and live_price <= pos.tp2)
            )

            if tp2_reached:
                logger.info(
                    "PositionManager: TP2 HIT ticket=%d price=%.5f tp2=%.5f",
                    ticket,
                    live_price,
                    pos.tp2,
                )
                self.close_position(ticket, "tp2_hit")

    # ------------------------------------------------------------------
    # Candle-level checks
    # ------------------------------------------------------------------

    def _candle_close_checks(
        self,
        pos: PositionState,
        features=None,
        regime_state=None,
    ) -> None:
        """Per-candle checks for a single position.

        5. Trailing stop update: if trailing_active, compute and apply new stop.
        6. Regime invalidation: if regime changed from entry regime and TP1 not
           yet hit, close the position.
        7. Time expiry: if candles elapsed > max_hold_candles and price < entry
           (no profit), close the position.

        Args:
            pos:          The :class:`PositionState` to check.
            features:     Optional feature DataFrame with the current bar.
            regime_state: Optional current :class:`RegimeState`.
        """
        ticket = pos.ticket

        # ---- 5. Trailing stop -------------------------------------------
        if pos.trailing_active:
            try:
                live_price = self._get_live_price(pos.symbol, pos.direction)
                atr = pos.atr_at_entry  # fallback; ideally from features

                if features is not None and hasattr(features, "columns"):
                    try:
                        import pandas as pd  # noqa: PLC0415
                        if not features.empty and "atr" in features.columns:
                            atr = float(features["atr"].iloc[-1])
                    except Exception:
                        pass  # keep atr_at_entry as fallback

                new_stop = self._stop_target_engine.compute_trailing_stop(
                    current_price=live_price,
                    current_stop=pos.current_stop,
                    atr=atr,
                    direction=pos.direction,
                )

                if new_stop != pos.current_stop:
                    old_stop = pos.current_stop

                    with self._lock:
                        if ticket in self._positions:
                            self._positions[ticket].current_stop = new_stop

                    logger.info(
                        "PositionManager: trailing stop updated ticket=%d "
                        "%s old=%.5f new=%.5f",
                        ticket,
                        pos.direction,
                        old_stop,
                        new_stop,
                    )

                    if self._journal is not None:
                        try:
                            self._journal.on_position_event(
                                ticket=ticket,
                                event_type="trail_update",
                                details={
                                    "old_stop": old_stop,
                                    "new_stop": new_stop,
                                    "current_price": live_price,
                                    "atr": atr,
                                },
                            )
                        except Exception as exc:
                            logger.error(
                                "PositionManager: journal trail_update raised "
                                "for ticket=%d — %s",
                                ticket,
                                exc,
                            )

            except Exception as exc:
                logger.error(
                    "PositionManager: trailing stop update failed for "
                    "ticket=%d — %s",
                    ticket,
                    exc,
                )

        # ---- 6. Regime invalidation ------------------------------------
        if regime_state is not None and not pos.tp1_hit:
            try:
                current_regime = str(
                    getattr(regime_state, "m15_regime", "RANGE")
                )
                if current_regime != pos.regime_at_entry:
                    logger.info(
                        "PositionManager: REGIME SHIFT ticket=%d "
                        "entry_regime=%s current_regime=%s — closing",
                        ticket,
                        pos.regime_at_entry,
                        current_regime,
                    )
                    self.close_position(ticket, "regime_shift")
                    return
            except Exception as exc:
                logger.error(
                    "PositionManager: regime invalidation check raised for "
                    "ticket=%d — %s",
                    ticket,
                    exc,
                )

        # Re-confirm position still exists after potential regime close
        with self._lock:
            pos = self._positions.get(ticket)

        if pos is None:
            return

        # ---- 7. Time expiry --------------------------------------------
        try:
            candles_open = (
                datetime.now(timezone.utc) - pos.open_time
            ).total_seconds() / (15 * 60)

            breakeven = pos.entry_price

            try:
                live_price = self._get_live_price(pos.symbol, pos.direction)
            except Exception:
                live_price = pos.entry_price  # conservative fallback

            at_or_below_breakeven = (
                (pos.direction == "LONG" and live_price < breakeven)
                or (pos.direction == "SHORT" and live_price > breakeven)
            )

            if candles_open > pos.max_hold_candles and at_or_below_breakeven:
                logger.info(
                    "PositionManager: TIME EXPIRY ticket=%d "
                    "candles_open=%.1f max=%d price=%.5f entry=%.5f",
                    ticket,
                    candles_open,
                    pos.max_hold_candles,
                    live_price,
                    pos.entry_price,
                )
                self.close_position(ticket, "time_expiry")

        except Exception as exc:
            logger.error(
                "PositionManager: time expiry check failed for ticket=%d — %s",
                ticket,
                exc,
            )

    # ------------------------------------------------------------------
    # Orphan scanner
    # ------------------------------------------------------------------

    def _scan_orphaned_positions(self) -> None:
        """On startup: scan all MT5 open positions and set emergency stops on orphans.

        Any MT5 position not currently tracked in self._positions is considered
        an orphan (system was restarted mid-trade). A 2*ATR emergency stop is
        set in MT5 as a safety net while the system stabilises.
        """
        logger.info("PositionManager: scanning for orphaned positions")

        try:
            mt5_positions = self._mt5.positions_get()
        except MT5ConnectionError as exc:
            logger.error(
                "PositionManager: MT5ConnectionError during orphan scan — %s", exc
            )
            return
        except Exception as exc:
            logger.error(
                "PositionManager: unexpected error during orphan scan — %s", exc
            )
            return

        with self._lock:
            tracked_tickets = set(self._positions.keys())

        for raw_pos in mt5_positions:
            try:
                ticket = int(raw_pos.get("ticket", 0))
                if ticket == 0:
                    continue

                if ticket in tracked_tickets:
                    continue  # already managed; skip

                symbol = str(raw_pos.get("symbol", ""))
                open_price = float(raw_pos.get("open_price", 0.0))
                # MT5 type: 0=buy (LONG), 1=sell (SHORT)
                pos_type = int(raw_pos.get("type", 0))
                direction = "LONG" if pos_type == 0 else "SHORT"

                logger.warning(
                    "PositionManager: ORPHAN FOUND ticket=%d %s %s "
                    "open_price=%.5f — setting emergency MT5 SL",
                    ticket,
                    symbol,
                    direction,
                    open_price,
                )

                # Estimate ATR from a rough 1% of price as absolute fallback
                atr_fallback = open_price * 0.01
                emergency_stop_distance = _ORPHAN_EMERGENCY_ATR_MULTIPLE * atr_fallback

                if direction == "LONG":
                    emergency_sl = open_price - emergency_stop_distance
                else:
                    emergency_sl = open_price + emergency_stop_distance

                self._modify_sl_in_mt5(ticket, sl=emergency_sl)

                logger.warning(
                    "PositionManager: orphan ticket=%d emergency SL set at %.5f",
                    ticket,
                    emergency_sl,
                )

            except Exception as exc:
                logger.error(
                    "PositionManager: error processing orphan position — %s", exc
                )

    # ------------------------------------------------------------------
    # MT5 interaction helpers
    # ------------------------------------------------------------------

    def _get_live_price(self, symbol: str, direction: str) -> float:
        """Fetch current bid (SHORT) or ask (LONG) from MT5.

        Args:
            symbol:    Instrument symbol.
            direction: "LONG" uses ask; "SHORT" uses bid.

        Returns:
            Current market price as a float.

        Raises:
            MT5ConnectionError: If the tick cannot be fetched.
            ValueError: If the tick data is malformed.
        """
        tick = self._mt5.symbol_info_tick(symbol)

        if direction == "LONG":
            price = tick.get("ask")
        else:
            price = tick.get("bid")

        if price is None:
            raise ValueError(
                f"PositionManager: tick for {symbol} missing "
                f"{'ask' if direction == 'LONG' else 'bid'} field"
            )

        return float(price)

    def _modify_sl_in_mt5(self, ticket: int, sl: float) -> bool:
        """Set (or remove) SL on an MT5 position using TRADE_ACTION_SLTP.

        This is used ONLY for emergency orphan fallback stops (sl > 0) and
        for clearing those stops after a position is registered (sl = 0).

        Args:
            ticket: MT5 position ticket.
            sl:     New stop-loss price. Pass 0 to remove the SL.

        Returns:
            True on success, False otherwise.
        """
        request = {
            "action": TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": 0,
        }

        try:
            result = self._mt5.order_send(request)
            retcode = result.get("retcode")

            if retcode == TRADE_RETCODE_DONE:
                if sl == 0:
                    logger.debug(
                        "PositionManager: cleared MT5 SL for ticket=%d", ticket
                    )
                else:
                    logger.info(
                        "PositionManager: set emergency MT5 SL=%.5f for ticket=%d",
                        sl,
                        ticket,
                    )
                return True

            logger.error(
                "PositionManager: _modify_sl_in_mt5 failed ticket=%d "
                "retcode=%s comment=%s",
                ticket,
                retcode,
                result.get("comment", ""),
            )
            return False

        except MT5ConnectionError as exc:
            logger.error(
                "PositionManager: MT5ConnectionError in _modify_sl_in_mt5 "
                "ticket=%d — %s",
                ticket,
                exc,
            )
            return False
        except Exception as exc:
            logger.error(
                "PositionManager: unexpected error in _modify_sl_in_mt5 "
                "ticket=%d — %s",
                ticket,
                exc,
            )
            return False
