"""Pre-trade checklist: R:R, session, spread, regime, news, and prop-firm gates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.utils.helpers import encode_session
from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level RR minimums (mirrors stop_target_engine.py defaults)
# ---------------------------------------------------------------------------
_RR_MINIMUMS: dict[str, float] = {
    "momentum":       2.0,
    "mean_reversion": 1.5,
    "breakout":       2.5,
}

# Session name -> encode_session() integer value
_SESSION_CODES: dict[str, int] = {
    "asian":    0,
    "london":   1,
    "newyork":  2,
    "overlap":  3,
}

# Maximum stop size as an ATR multiple (matches stop_target_engine default)
_MAX_STOP_ATR_MULTIPLE: float = 2.0

# Spread tolerance (current must not exceed typical * this factor)
_MAX_SPREAD_RATIO: float = 1.5

# Staleness: maximum age of a signal in minutes (2 M15 bars)
_MAX_SIGNAL_AGE_MINUTES: float = 30.0

# Circuit-breaker level 1 daily-loss trigger
_CB_L1_DAILY_LOSS: float = 0.015


def _load_config() -> object | None:
    """Load application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class PreTradeChecker:
    """Pre-trade checklist — fail-fast ordered validation.

    Each individual check returns ``(passed: bool, reason: str)``.
    ``run_all`` executes them in strict order and stops at the first
    critical failure.
    """

    def __init__(self) -> None:
        """Initialise and lazily wire up the PropFirmCompliance component."""
        from core.risk.prop_firm_compliance import PropFirmCompliance
        self._prop_firm = PropFirmCompliance()

        # Load asset config for session filter and spread lookup
        cfg = _load_config()
        try:
            self._assets: list = list(cfg.assets)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            self._assets = []

        logger.debug("PreTradeChecker initialised.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        signal: Any,
        account_state: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Run all pre-trade checks in order.

        Fails fast on the first check that does not pass: subsequent checks
        are skipped and the failing reason is returned immediately.

        Args:
            signal:        SignalOutput from the signal router.
            account_state: Live account metrics dict.

        Returns:
            Tuple of (all_passed: bool, failed_reasons: list[str]).
            When all checks pass, failed_reasons is an empty list.
        """
        checks = [
            lambda: self.rr_check(signal),
            lambda: self.stop_size_check(signal),
            lambda: self.session_check(signal, account_state),
            lambda: self.spread_check(signal, account_state),
            lambda: self.regime_check(signal),
            lambda: self.staleness_check(signal),
            lambda: self.existing_position_check(signal, account_state),
            lambda: self.daily_loss_check(account_state),
            lambda: self.news_filter_check(signal),
            lambda: self.prop_firm_check(signal, account_state),
        ]

        for check_fn in checks:
            passed, reason = check_fn()
            logger.debug("PreTradeChecker [%s]: %s", check_fn.__name__ if hasattr(check_fn, "__name__") else "check", reason)
            if not passed:
                logger.info(
                    "PreTradeChecker: FAILED — %s | signal=%s %s",
                    reason,
                    getattr(signal, "asset", "?"),
                    getattr(signal, "signal", "?"),
                )
                return False, [reason]

        logger.debug(
            "PreTradeChecker: all checks passed for %s %s",
            getattr(signal, "asset", "?"),
            getattr(signal, "signal", "?"),
        )
        return True, []

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def rr_check(self, signal: Any) -> tuple[bool, str]:
        """Check that the signal's R:R ratio meets the module's minimum.

        Minimums: momentum=2.0, mean_reversion=1.5, breakout=2.5.

        Args:
            signal: SignalOutput with attributes ``rr_ratio`` and ``module``.

        Returns:
            (passed, reason)
        """
        rr_ratio: float = float(getattr(signal, "rr_ratio", 0.0))
        module: str     = str(getattr(signal, "module", "")).lower()

        min_rr = _RR_MINIMUMS.get(module)
        if min_rr is None:
            # Load from config as fallback
            cfg = _load_config()
            try:
                rr_cfg = cfg.rr_minimums  # type: ignore[union-attr]
                min_rr = float(getattr(rr_cfg, module, 2.0))
            except Exception:  # noqa: BLE001
                min_rr = 2.0

        if rr_ratio >= min_rr:
            return True, f"rr_check passed (rr={rr_ratio:.2f} >= min={min_rr:.2f})"

        return False, (
            f"rr_check failed: rr={rr_ratio:.2f} < min={min_rr:.2f} "
            f"for module={module}"
        )

    def stop_size_check(self, signal: Any) -> tuple[bool, str]:
        """Check that the stop distance does not exceed max_stop_atr_multiple * ATR.

        Args:
            signal: SignalOutput with attributes ``entry_price``, ``stop_loss``,
                    ``atr``, and ``module``.

        Returns:
            (passed, reason)
        """
        entry: float  = float(getattr(signal, "entry_price", 0.0))
        stop: float   = float(getattr(signal, "stop_loss", 0.0))
        atr: float    = float(getattr(signal, "atr", 0.0))
        module: str   = str(getattr(signal, "module", "")).upper()

        if atr <= 0.0:
            return False, f"stop_size_check failed: invalid atr={atr}"

        stop_distance = abs(entry - stop)
        # BREAKOUT uses a stricter cap of 1.5 * ATR
        max_multiple = 1.5 if module == "BREAKOUT" else _MAX_STOP_ATR_MULTIPLE
        max_stop = max_multiple * atr

        if stop_distance <= max_stop:
            return True, (
                f"stop_size_check passed "
                f"(stop_dist={stop_distance:.5f} <= max={max_stop:.5f})"
            )

        return False, (
            f"stop_size_check failed: stop_dist={stop_distance:.5f} > "
            f"max={max_stop:.5f} ({max_multiple}*ATR={atr:.5f})"
        )

    def session_check(
        self,
        signal: Any,
        account_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check that the current UTC session is in the asset's allowed sessions.

        Args:
            signal:        SignalOutput with attribute ``asset``.
            account_state: Dict optionally containing ``timestamp`` (datetime
                           or pd.Timestamp).  Falls back to datetime.utcnow()
                           when absent.

        Returns:
            (passed, reason)
        """
        asset: str = str(getattr(signal, "asset", "")).upper()

        # Determine current UTC hour
        ts_raw = account_state.get("timestamp")
        if ts_raw is not None:
            try:
                if isinstance(ts_raw, pd.Timestamp):
                    current_ts = ts_raw
                elif isinstance(ts_raw, datetime):
                    current_ts = pd.Timestamp(ts_raw)
                else:
                    current_ts = pd.Timestamp(ts_raw)
            except Exception:  # noqa: BLE001
                current_ts = pd.Timestamp(datetime.now(timezone.utc))
        else:
            current_ts = pd.Timestamp(datetime.now(timezone.utc))

        current_session_code: int = encode_session(current_ts)

        # Look up the asset's session_filter from config
        allowed_sessions: list[str] | None = None
        for asset_cfg in self._assets:
            try:
                if str(asset_cfg.symbol).upper() == asset:
                    raw_filter = asset_cfg.session_filter
                    allowed_sessions = [str(s).lower() for s in raw_filter]
                    break
            except AttributeError:
                continue

        if allowed_sessions is None:
            # Asset not found in config — allow all sessions
            logger.debug(
                "session_check: asset %s not found in config — allowing all sessions",
                asset,
            )
            return True, f"session_check passed (asset {asset} not in config, permitting)"

        allowed_codes: set[int] = {
            _SESSION_CODES[s] for s in allowed_sessions if s in _SESSION_CODES
        }

        if current_session_code in allowed_codes:
            return True, (
                f"session_check passed "
                f"(session_code={current_session_code} in {allowed_sessions})"
            )

        return False, (
            f"session_check failed: current_session_code={current_session_code} "
            f"not in allowed={allowed_sessions} for {asset}"
        )

    def spread_check(
        self,
        signal: Any,
        account_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check that the current spread does not exceed typical_spread * 1.5.

        Args:
            signal:        SignalOutput with attribute ``asset``.
            account_state: Dict with key ``current_spread`` (float, in price units).

        Returns:
            (passed, reason)
        """
        asset: str          = str(getattr(signal, "asset", "")).upper()
        current_spread: float = float(account_state.get("current_spread", 0.0))

        # Look up typical spread from config
        typical_spread: float | None = None
        for asset_cfg in self._assets:
            try:
                if str(asset_cfg.symbol).upper() == asset:
                    typical_spread = float(asset_cfg.typical_spread)
                    break
            except AttributeError:
                continue

        if typical_spread is None or typical_spread <= 0.0:
            # Cannot validate — skip (permissive)
            logger.debug(
                "spread_check: no typical_spread for %s — skipping", asset
            )
            return True, f"spread_check skipped (no typical_spread for {asset})"

        if current_spread <= 0.0:
            logger.debug(
                "spread_check: no current_spread in account_state — skipping"
            )
            return True, "spread_check skipped (no current_spread in account_state)"

        max_spread = typical_spread * _MAX_SPREAD_RATIO

        if current_spread <= max_spread:
            return True, (
                f"spread_check passed "
                f"(current={current_spread:.5f} <= max={max_spread:.5f})"
            )

        return False, (
            f"spread_check failed: current_spread={current_spread:.5f} > "
            f"max_spread={max_spread:.5f} ({_MAX_SPREAD_RATIO}x typical={typical_spread:.5f})"
        )

    def regime_check(self, signal: Any) -> tuple[bool, str]:
        """Check that the signal's module matches the current regime's active_strategy.

        Args:
            signal: SignalOutput with attributes ``module`` and ``regime_context``
                    (RegimeState or None).

        Returns:
            (passed, reason)
        """
        module: str = str(getattr(signal, "module", "")).lower()
        regime_ctx  = getattr(signal, "regime_context", None)

        if regime_ctx is None:
            return True, "regime_check skipped (no regime_context on signal)"

        active_strategy: str = str(getattr(regime_ctx, "active_strategy", "")).lower()

        if not active_strategy or active_strategy == "no_trade":
            return False, (
                f"regime_check failed: active_strategy='{active_strategy}' "
                f"means no trading"
            )

        if module == active_strategy:
            return True, (
                f"regime_check passed (module={module} matches active_strategy)"
            )

        return False, (
            f"regime_check failed: signal module='{module}' != "
            f"regime active_strategy='{active_strategy}'"
        )

    def staleness_check(self, signal: Any) -> tuple[bool, str]:
        """Check that the signal is not older than 2 M15 bars (30 minutes).

        Args:
            signal: SignalOutput with attribute ``timestamp`` (datetime).

        Returns:
            (passed, reason)
        """
        sig_ts = getattr(signal, "timestamp", None)
        if sig_ts is None:
            return True, "staleness_check skipped (no timestamp on signal)"

        now_utc = datetime.now(timezone.utc)

        try:
            if isinstance(sig_ts, datetime):
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.replace(tzinfo=timezone.utc)
                age_minutes = (now_utc - sig_ts).total_seconds() / 60.0
            else:
                # pd.Timestamp or string
                sig_ts_dt = pd.Timestamp(sig_ts).to_pydatetime()
                if sig_ts_dt.tzinfo is None:
                    sig_ts_dt = sig_ts_dt.replace(tzinfo=timezone.utc)
                age_minutes = (now_utc - sig_ts_dt).total_seconds() / 60.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("staleness_check: could not parse timestamp: %s", exc)
            return True, "staleness_check skipped (timestamp parse error)"

        if age_minutes <= _MAX_SIGNAL_AGE_MINUTES:
            return True, (
                f"staleness_check passed (age={age_minutes:.1f}min <= "
                f"max={_MAX_SIGNAL_AGE_MINUTES:.0f}min)"
            )

        return False, (
            f"staleness_check failed: signal age={age_minutes:.1f}min > "
            f"max={_MAX_SIGNAL_AGE_MINUTES:.0f}min (2 M15 bars)"
        )

    def existing_position_check(
        self,
        signal: Any,
        account_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check there is no existing open position in the same asset and direction.

        Args:
            signal:        SignalOutput with attributes ``asset`` and ``signal``
                           (direction string).
            account_state: Dict with optional key ``open_positions`` (list of
                           position objects with ``.asset`` / ``.symbol`` and
                           ``.direction`` / ``.signal`` attributes).

        Returns:
            (passed, reason)
        """
        asset: str     = str(getattr(signal, "asset", "")).upper()
        direction: str = str(getattr(signal, "signal", "")).upper()
        open_positions = account_state.get("open_positions", [])

        for pos in open_positions:
            pos_asset = str(
                getattr(pos, "asset", getattr(pos, "symbol", ""))
            ).upper()
            pos_dir = str(
                getattr(pos, "direction", getattr(pos, "signal", ""))
            ).upper()

            if pos_asset == asset and pos_dir == direction:
                return False, (
                    f"existing_position_check failed: open {direction} on "
                    f"{asset} already exists"
                )

        return True, (
            f"existing_position_check passed (no open {direction} on {asset})"
        )

    def daily_loss_check(self, account_state: dict[str, Any]) -> tuple[bool, str]:
        """Check that today's losses have not yet hit the circuit-breaker level 1 trigger.

        Args:
            account_state: Dict with key ``daily_loss_pct`` (float, positive = loss).

        Returns:
            (passed, reason)
        """
        daily_loss: float = float(account_state.get("daily_loss_pct", 0.0))

        # Load trigger from config if available
        cfg = _load_config()
        try:
            trigger = float(cfg.circuit_breakers.level_1.daily_loss_trigger)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            trigger = _CB_L1_DAILY_LOSS

        if daily_loss < trigger:
            return True, (
                f"daily_loss_check passed "
                f"(daily_loss={daily_loss:.4f} < trigger={trigger:.4f})"
            )

        return False, (
            f"daily_loss_check failed: daily_loss={daily_loss:.4f} >= "
            f"circuit_breaker_l1_trigger={trigger:.4f}"
        )

    def news_filter_check(self, signal: Any) -> tuple[bool, str]:
        """Check there is no high-impact news within the block window.

        When ``news_filter.enabled`` is False in config, always passes.
        In this version the live schedule lookup is stubbed: it always passes
        when enabled, pending integration with a news calendar feed.

        Args:
            signal: SignalOutput (unused in stub — all checks are time-based).

        Returns:
            (passed, reason)
        """
        cfg = _load_config()
        try:
            news_cfg = cfg.news_filter  # type: ignore[union-attr]
            enabled: bool = bool(news_cfg.enabled)
        except Exception:  # noqa: BLE001
            enabled = False

        if not enabled:
            return True, "news_filter_check passed (news_filter disabled)"

        # Stub: live news calendar integration pending.
        # When the calendar is wired in, evaluate against hardcoded weekly
        # schedule using block_before_minutes / block_after_minutes from config.
        return True, "news_filter_check passed (stub — no news events detected)"

    def prop_firm_check(
        self,
        signal: Any,
        account_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check all prop-firm limits still have buffer remaining.

        Delegates to :class:`PropFirmCompliance.pre_trade_check`.

        Args:
            signal:        SignalOutput (used to derive risk amount when
                           ``risk_amount`` attribute is present).
            account_state: Live account metrics dict.

        Returns:
            (passed, reason)
        """
        # Determine the risk amount for this trade
        risk_amount: float = float(getattr(signal, "risk_amount", 0.0))
        if risk_amount <= 0.0:
            # Estimate from sizing_inputs if pre-computed
            sizing = getattr(signal, "sizing_inputs", {})
            risk_amount = float(sizing.get("risk_amount_currency", 0.0))

        passed, reason = self._prop_firm.pre_trade_check(
            new_risk=risk_amount,
            account_state=account_state,
        )

        if passed:
            return True, f"prop_firm_check passed: {reason}"

        return False, f"prop_firm_check failed: {reason}"
