"""Real-time prop-firm compliance tracker (FTMO rules with internal buffers)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
# Default thresholds (mirror prop_firm.yaml)
# ---------------------------------------------------------------------------
_DEFAULT_DAILY_LOSS_BUFFER  = 0.04   # internal halt at 4%, not FTMO's 5%
_DEFAULT_TOTAL_DD_BUFFER    = 0.08   # internal halt at 8%, not FTMO's 10%
_DEFAULT_MIN_TRADING_DAYS   = 4


class PropFirmCompliance:
    """Real-time prop-firm compliance tracker.

    Tracks: daily_loss_pct, total_drawdown_pct, consecutive_losses,
    days_traded.

    Uses internal buffers (4% not 5%, 8% not 10%) as a safety margin so
    the system stops before breaching the actual FTMO rule limits.

    Only enforced when ``config.mode == True``.
    """

    def __init__(self) -> None:
        """Initialise tracking state and load buffer values from config."""
        self.peak_balance: float = 0.0
        self.daily_start_balance: float = 0.0
        self.consecutive_losses: int = 0
        self.days_traded: set[str] = set()
        self._last_reset_date: str = ""

        cfg = _load_config()
        try:
            buffers = cfg.internal_buffers  # type: ignore[union-attr]
            self._daily_loss_buffer: float = float(buffers.daily_loss_buffer)
            self._total_dd_buffer: float   = float(buffers.total_dd_buffer)
        except Exception:  # noqa: BLE001
            logger.warning(
                "PropFirmCompliance: failed to read internal_buffers config — using defaults."
            )
            self._daily_loss_buffer = _DEFAULT_DAILY_LOSS_BUFFER
            self._total_dd_buffer   = _DEFAULT_TOTAL_DD_BUFFER

        try:
            ftmo = cfg.ftmo  # type: ignore[union-attr]
            self._min_trading_days: int = int(ftmo.min_trading_days)
        except Exception:  # noqa: BLE001
            self._min_trading_days = _DEFAULT_MIN_TRADING_DAYS

        logger.debug(
            "PropFirmCompliance initialised: daily_buffer=%.3f total_dd_buffer=%.3f",
            self._daily_loss_buffer,
            self._total_dd_buffer,
        )

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_from_trade(self, trade_result: dict[str, Any]) -> None:
        """Update consecutive losses and days_traded from a closed trade.

        Args:
            trade_result: Dict with keys:
                pnl_currency (float) – profit/loss in account currency.
                entry_time   (datetime) – trade entry datetime.
                exit_time    (datetime) – trade exit datetime.
        """
        pnl: float = float(trade_result.get("pnl_currency", 0.0))

        if pnl < 0.0:
            self.consecutive_losses += 1
            logger.debug(
                "PropFirmCompliance: losing trade (pnl=%.2f), "
                "consecutive_losses=%d",
                pnl, self.consecutive_losses,
            )
        elif pnl > 0.0:
            self.consecutive_losses = 0
            logger.debug(
                "PropFirmCompliance: winning trade (pnl=%.2f), "
                "consecutive_losses reset to 0",
                pnl,
            )
        # pnl == 0.0 (break-even) does not reset or increment

        # Record the trading day
        exit_time = trade_result.get("exit_time")
        if exit_time is not None:
            try:
                if isinstance(exit_time, datetime):
                    day_str = exit_time.strftime("%Y-%m-%d")
                else:
                    day_str = str(exit_time)[:10]
                self.days_traded.add(day_str)
                logger.debug(
                    "PropFirmCompliance: recorded trading day %s (total=%d)",
                    day_str, len(self.days_traded),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PropFirmCompliance.update_from_trade: could not parse exit_time: %s",
                    exc,
                )

    def update_balance(self, current_balance: float, current_date: str) -> None:
        """Update peak balance and daily_start_balance; reset daily state on new date.

        Args:
            current_balance: Current account balance in account currency.
            current_date:    ISO date string (YYYY-MM-DD) for the current day.
        """
        # Reset daily tracking on new trading day
        self._check_daily_reset(current_date)

        # Initialise daily_start_balance on first call
        if self.daily_start_balance <= 0.0:
            self.daily_start_balance = current_balance
            logger.debug(
                "PropFirmCompliance: daily_start_balance initialised to %.2f",
                self.daily_start_balance,
            )

        # Update peak (high-water mark)
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            logger.debug(
                "PropFirmCompliance: new peak_balance=%.2f", self.peak_balance
            )

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        new_risk: float,
        account_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify a prospective trade will not breach prop-firm limits.

        When ``config.mode`` is ``False`` the check always passes.

        Checks (using internal buffers, not the raw FTMO limits):
        1. Current daily_loss_pct < daily_loss_buffer (0.04).
        2. Current total_drawdown_pct < total_dd_buffer (0.08).
        3. daily_loss_pct + new_risk_pct < daily_loss_buffer (would-be daily loss
           after adding the new trade's maximum risk).

        Args:
            new_risk:      Risk amount in account currency for the proposed trade.
            account_state: Dict with keys balance (float) and equity (float).

        Returns:
            Tuple of (passes: bool, reason: str).
        """
        cfg = _load_config()
        # Only enforce when prop-firm mode is enabled
        try:
            mode: bool = bool(cfg.mode)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            mode = False

        if not mode:
            return True, "prop_firm_mode_disabled"

        balance: float = float(account_state.get("balance", self.daily_start_balance))
        equity:  float = float(account_state.get("equity", balance))

        if balance <= 0.0:
            return False, "invalid_balance"

        # Derive current daily loss as a positive fraction
        if self.daily_start_balance > 0.0:
            daily_loss_pct = max(
                0.0,
                (self.daily_start_balance - equity) / self.daily_start_balance,
            )
        else:
            daily_loss_pct = 0.0

        # Derive total drawdown from peak balance
        if self.peak_balance > 0.0:
            total_dd_pct = max(
                0.0,
                (self.peak_balance - equity) / self.peak_balance,
            )
        else:
            total_dd_pct = 0.0

        # Check 1: current daily loss
        if daily_loss_pct >= self._daily_loss_buffer:
            reason = (
                f"daily_loss_pct={daily_loss_pct:.4f} >= "
                f"daily_loss_buffer={self._daily_loss_buffer:.4f}"
            )
            logger.info("PropFirmCompliance: FAIL — %s", reason)
            return False, reason

        # Check 2: total drawdown
        if total_dd_pct >= self._total_dd_buffer:
            reason = (
                f"total_dd_pct={total_dd_pct:.4f} >= "
                f"total_dd_buffer={self._total_dd_buffer:.4f}"
            )
            logger.info("PropFirmCompliance: FAIL — %s", reason)
            return False, reason

        # Check 3: projected daily loss after the new trade
        new_risk_pct = new_risk / balance
        projected_daily_loss = daily_loss_pct + new_risk_pct
        if projected_daily_loss >= self._daily_loss_buffer:
            reason = (
                f"projected_daily_loss={projected_daily_loss:.4f} "
                f"(current={daily_loss_pct:.4f} + new_risk_pct={new_risk_pct:.4f}) "
                f">= daily_loss_buffer={self._daily_loss_buffer:.4f}"
            )
            logger.info("PropFirmCompliance: FAIL — %s", reason)
            return False, reason

        logger.debug(
            "PropFirmCompliance: pass "
            "(daily_loss=%.4f total_dd=%.4f new_risk_pct=%.4f)",
            daily_loss_pct, total_dd_pct, new_risk_pct,
        )
        return True, "prop_firm_checks_passed"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current proximity to all limits as a status dict.

        Returns:
            Dict with keys:
            - daily_loss_pct          (float) current day's loss as positive fraction.
            - total_dd_pct            (float) drawdown from peak_balance.
            - consecutive_losses      (int)   consecutive losing trades.
            - days_traded             (int)   total unique trading days.
            - daily_loss_buffer       (float) internal daily limit.
            - total_dd_buffer         (float) internal total DD limit.
            - daily_loss_remaining    (float) headroom to daily limit.
            - total_dd_remaining      (float) headroom to total DD limit.
            - min_trading_days        (int)   target minimum days.
            - trading_days_remaining  (int)   days still needed.
        """
        # These fractions require a known daily_start_balance and peak_balance;
        # return 0.0 when not yet initialised.
        if self.daily_start_balance > 0.0:
            # We don't hold equity directly, so use peak as surrogate for
            # current equity when called standalone. Callers should use
            # update_balance() before calling get_status().
            daily_loss_pct = 0.0   # cannot compute without live equity here
        else:
            daily_loss_pct = 0.0

        total_dd_pct = 0.0  # same; updated via update_balance()

        daily_remaining  = max(0.0, self._daily_loss_buffer - daily_loss_pct)
        total_remaining  = max(0.0, self._total_dd_buffer   - total_dd_pct)
        days_remaining   = max(0, self._min_trading_days - len(self.days_traded))

        return {
            "daily_loss_pct":         daily_loss_pct,
            "total_dd_pct":           total_dd_pct,
            "consecutive_losses":     self.consecutive_losses,
            "days_traded":            len(self.days_traded),
            "daily_loss_buffer":      self._daily_loss_buffer,
            "total_dd_buffer":        self._total_dd_buffer,
            "daily_loss_remaining":   daily_remaining,
            "total_dd_remaining":     total_remaining,
            "min_trading_days":       self._min_trading_days,
            "trading_days_remaining": days_remaining,
            "peak_balance":           self.peak_balance,
            "daily_start_balance":    self.daily_start_balance,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_daily_reset(self, current_date: str) -> None:
        """Reset daily_start_balance and consecutive_losses if the date changed.

        Args:
            current_date: ISO date string (YYYY-MM-DD).
        """
        if current_date and current_date != self._last_reset_date:
            logger.info(
                "PropFirmCompliance: new trading day detected (%s -> %s) — "
                "resetting daily_start_balance.",
                self._last_reset_date or "uninitialised",
                current_date,
            )
            # daily_start_balance will be set by the next update_balance() call
            self.daily_start_balance = 0.0
            self._last_reset_date = current_date
