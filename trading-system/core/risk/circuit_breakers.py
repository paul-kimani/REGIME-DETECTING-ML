"""Four-level circuit breaker system for session and system-level loss limits."""

from __future__ import annotations

from core.utils.logger import get_logger

logger = get_logger(__name__)


def _load_config() -> object | None:
    """Load application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Default thresholds (mirror prop_firm.yaml circuit_breakers section)
# ---------------------------------------------------------------------------
_L1_DAILY_LOSS  = 0.015
_L1_CONSEC      = 3
_L2_DAILY_LOSS  = 0.025
_L2_CONSEC      = 5
_L3_DAILY_LOSS  = 0.040
_L3_WEEKLY_LOSS = 0.060
_L3_PEAK_DD     = 0.080


class CircuitBreaker:
    """Four-level circuit breaker system.

    Level 0: All clear.
    Level 1: daily_loss > 1.5% OR consecutive_losses >= 3
             -> reduce_size_30pct (size multiplier 0.70).
    Level 2: daily_loss > 2.5% OR consecutive_losses >= 5
             -> halt_session (size multiplier 0.0).
    Level 3: daily_loss > 4.0% OR weekly_loss > 6% OR peak_drawdown > 8%
             -> halt_all_manual_review (size multiplier 0.0).
    Level 4: global_crisis_state OR spread_multiplier_exceeded_3x
             OR mt5_connection_unstable
             -> emergency_close_all (size multiplier 0.0).
    """

    def __init__(self) -> None:
        """Initialise and load circuit-breaker thresholds from config."""
        cfg = _load_config()

        try:
            cb = cfg.circuit_breakers  # type: ignore[union-attr]
            self._l1_daily: float  = float(cb.level_1.daily_loss_trigger)
            self._l1_consec: int   = int(cb.level_1.consecutive_losses_trigger)
            self._l2_daily: float  = float(cb.level_2.daily_loss_trigger)
            self._l2_consec: int   = int(cb.level_2.consecutive_losses_trigger)
            self._l3_daily: float  = float(cb.level_3.daily_loss_trigger)
            self._l3_weekly: float = float(cb.level_3.weekly_loss_trigger)
            self._l3_dd: float     = float(cb.level_3.peak_drawdown_trigger)
        except Exception:  # noqa: BLE001
            logger.warning(
                "CircuitBreaker: failed to read circuit_breakers config — using defaults."
            )
            self._l1_daily  = _L1_DAILY_LOSS
            self._l1_consec = _L1_CONSEC
            self._l2_daily  = _L2_DAILY_LOSS
            self._l2_consec = _L2_CONSEC
            self._l3_daily  = _L3_DAILY_LOSS
            self._l3_weekly = _L3_WEEKLY_LOSS
            self._l3_dd     = _L3_PEAK_DD

        logger.debug(
            "CircuitBreaker initialised: L1(daily=%.3f consec=%d) "
            "L2(daily=%.3f consec=%d) L3(daily=%.3f weekly=%.3f dd=%.3f)",
            self._l1_daily, self._l1_consec,
            self._l2_daily, self._l2_consec,
            self._l3_daily, self._l3_weekly, self._l3_dd,
        )

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(
        self,
        account_state: dict,
        recent_trades: list,
        open_positions: list,
    ) -> tuple[int, str]:
        """Check all circuit breaker conditions in order from highest to lowest.

        account_state keys:
            balance            (float) account balance
            equity             (float) account equity
            daily_pnl_pct      (float) today's PnL as a signed fraction
            weekly_pnl_pct     (float) this week's PnL as a signed fraction
            peak_equity        (float) high-water mark equity
            daily_loss_pct     (float) today's loss as a *positive* fraction
            consecutive_losses (int)   number of consecutive losing trades
            global_risk_state  (str)   "RISK_ON" | "RISK_OFF" | "CRISIS"
            avg_spread_ratio   (float) current spread / typical spread
            mt5_connected      (bool)  MT5 connection status

        Args:
            account_state:   Live account metrics dict.
            recent_trades:   List of recent closed trade objects (unused
                             directly; consecutive_losses is in account_state).
            open_positions:  List of currently open position objects (unused
                             directly; could be extended).

        Returns:
            Tuple of (level: int, description: str).
            level 0 means all clear.
        """
        daily_loss      = float(account_state.get("daily_loss_pct", 0.0))
        weekly_pnl      = float(account_state.get("weekly_pnl_pct", 0.0))
        consec_losses   = int(account_state.get("consecutive_losses", 0))
        peak_equity     = float(account_state.get("peak_equity", 0.0))
        equity          = float(account_state.get("equity", 0.0))
        global_state    = str(account_state.get("global_risk_state", "RISK_ON")).upper()
        spread_ratio    = float(account_state.get("avg_spread_ratio", 1.0))
        mt5_connected   = bool(account_state.get("mt5_connected", True))

        # Compute peak drawdown from equity vs peak_equity
        if peak_equity > 0:
            peak_drawdown = max(0.0, (peak_equity - equity) / peak_equity)
        else:
            peak_drawdown = 0.0

        # Weekly loss as a positive fraction (negative weekly_pnl = loss)
        weekly_loss = max(0.0, -weekly_pnl)

        # ---- Level 4: emergency conditions (check first — most severe) ----
        level_4_triggers: list[str] = []
        if global_state == "CRISIS":
            level_4_triggers.append("global_crisis_state")
        if spread_ratio >= 3.0:
            level_4_triggers.append(f"spread_multiplier_exceeded_3x (ratio={spread_ratio:.2f})")
        if not mt5_connected:
            level_4_triggers.append("mt5_connection_unstable")

        if level_4_triggers:
            desc = "LEVEL 4 emergency_close_all: " + "; ".join(level_4_triggers)
            logger.warning("CircuitBreaker: %s", desc)
            return 4, desc

        # ---- Level 3: systemic loss / drawdown ----
        level_3_triggers: list[str] = []
        if daily_loss > self._l3_daily:
            level_3_triggers.append(
                f"daily_loss={daily_loss:.3f} > threshold={self._l3_daily:.3f}"
            )
        if weekly_loss > self._l3_weekly:
            level_3_triggers.append(
                f"weekly_loss={weekly_loss:.3f} > threshold={self._l3_weekly:.3f}"
            )
        if peak_drawdown > self._l3_dd:
            level_3_triggers.append(
                f"peak_drawdown={peak_drawdown:.3f} > threshold={self._l3_dd:.3f}"
            )

        if level_3_triggers:
            desc = "LEVEL 3 halt_all_manual_review: " + "; ".join(level_3_triggers)
            logger.warning("CircuitBreaker: %s", desc)
            return 3, desc

        # ---- Level 2: session halt ----
        level_2_triggers: list[str] = []
        if daily_loss > self._l2_daily:
            level_2_triggers.append(
                f"daily_loss={daily_loss:.3f} > threshold={self._l2_daily:.3f}"
            )
        if consec_losses >= self._l2_consec:
            level_2_triggers.append(
                f"consecutive_losses={consec_losses} >= threshold={self._l2_consec}"
            )

        if level_2_triggers:
            desc = "LEVEL 2 halt_session: " + "; ".join(level_2_triggers)
            logger.warning("CircuitBreaker: %s", desc)
            return 2, desc

        # ---- Level 1: reduce size ----
        level_1_triggers: list[str] = []
        if daily_loss > self._l1_daily:
            level_1_triggers.append(
                f"daily_loss={daily_loss:.3f} > threshold={self._l1_daily:.3f}"
            )
        if consec_losses >= self._l1_consec:
            level_1_triggers.append(
                f"consecutive_losses={consec_losses} >= threshold={self._l1_consec}"
            )

        if level_1_triggers:
            desc = "LEVEL 1 reduce_size_30pct: " + "; ".join(level_1_triggers)
            logger.info("CircuitBreaker: %s", desc)
            return 1, desc

        logger.debug(
            "CircuitBreaker: LEVEL 0 all clear "
            "(daily_loss=%.3f weekly_loss=%.3f consec=%d peak_dd=%.3f)",
            daily_loss, weekly_loss, consec_losses, peak_drawdown,
        )
        return 0, "all clear"

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------

    def get_size_multiplier(self, level: int) -> float:
        """Return the position-sizing multiplier for the given circuit-breaker level.

        Args:
            level: Circuit-breaker level (0–4).

        Returns:
            0.70 for level 1 (reduce 30%), 0.0 for level 2 and above, 1.0 for level 0.
        """
        if level == 0:
            return 1.0
        if level == 1:
            return 0.70
        # Levels 2, 3, 4 all halt trading
        return 0.0

    def is_trading_halted(self, level: int) -> bool:
        """Return True when trading must be halted (level >= 2).

        Args:
            level: Circuit-breaker level (0–4).

        Returns:
            True when level >= 2.
        """
        return level >= 2

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def get_recovery_multiplier(self, days_since_reset: int) -> float:
        """Return a graduated sizing multiplier for recovery after a circuit-breaker reset.

        Args:
            days_since_reset: Number of complete calendar days since the
                              circuit breaker was last reset (0-indexed, so
                              day_1 means days_since_reset == 1).

        Returns:
            Sizing multiplier: 0.50 on day 1, 0.75 on day 2, 1.00 on day 3+.
        """
        cfg = _load_config()
        try:
            rec = cfg.recovery_sizing  # type: ignore[union-attr]
            d1  = float(rec.day_1_multiplier)
            d2  = float(rec.day_2_multiplier)
            d3  = float(rec.day_3_plus_multiplier)
        except Exception:  # noqa: BLE001
            d1, d2, d3 = 0.50, 0.75, 1.00

        if days_since_reset <= 1:
            return d1
        if days_since_reset == 2:
            return d2
        return d3
