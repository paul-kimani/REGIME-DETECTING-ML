"""Risk engine package — orchestrates all sizing, compliance, and circuit-breaker components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from core.risk.circuit_breakers import CircuitBreaker
from core.risk.kelly_sizer import KellySizer
from core.risk.portfolio_risk import PortfolioRiskManager
from core.risk.pre_trade_checks import PreTradeChecker
from core.risk.prop_firm_compliance import PropFirmCompliance
from core.risk.stop_target_engine import StopTargetEngine
from core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default risk bounds — used when config is unavailable
# ---------------------------------------------------------------------------
_DEFAULT_MIN_RISK: float = 0.002
_DEFAULT_MAX_RISK: float = 0.015


def _load_config() -> object | None:
    """Load application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# TradeOrder dataclass
# ---------------------------------------------------------------------------


@dataclass
class TradeOrder:
    """Complete trade specification ready for execution.

    All prices are in the instrument's native units.  Lot size has been
    validated and rounded to the broker's lot step.
    """

    signal_id: int
    magic_number: int
    symbol: str
    timeframe: str
    direction: str        # "LONG" | "SHORT"
    module: str

    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    atr: float
    stop_distance_pips: float
    rr_ratio: float

    # Sizing breakdown (for logging / audit trail)
    account_balance: float
    base_risk_pct: float
    kelly_multiplier: float
    volatility_scalar: float
    regime_age_multiplier: float
    alignment_multiplier: float
    correlation_multiplier: float
    global_multiplier: float
    final_risk_pct: float
    risk_amount_currency: float

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime_context: Optional[object] = None


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Orchestrates all risk components.

    This is the main interface for the execution layer.  ``process()``
    accepts a ``SignalOutput`` and account state, runs the full validation
    and sizing pipeline, and either returns a ready-to-execute
    :class:`TradeOrder` or ``None`` if any step rejects the trade.
    """

    def __init__(self) -> None:
        """Instantiate all sub-components."""
        self._pre_trade   = PreTradeChecker()
        self._cb          = CircuitBreaker()
        self._prop_firm   = PropFirmCompliance()
        self._kelly       = KellySizer()
        self._portfolio   = PortfolioRiskManager()
        self._stops       = StopTargetEngine()

        # Load min/max risk bounds from config for final clipping
        cfg = _load_config()
        try:
            sizing = cfg.sizing  # type: ignore[union-attr]
            self._min_risk: float = float(sizing.min_risk_per_trade)
            self._max_risk: float = float(sizing.max_risk_per_trade)
        except Exception:  # noqa: BLE001
            self._min_risk = _DEFAULT_MIN_RISK
            self._max_risk = _DEFAULT_MAX_RISK

        logger.debug(
            "RiskEngine initialised: min_risk=%.3f max_risk=%.3f",
            self._min_risk, self._max_risk,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        signal: object,
        account_state: dict,
        features: pd.DataFrame,
        open_positions: list,
    ) -> Optional[TradeOrder]:
        """Run the full risk pipeline for a candidate signal.

        Steps (any failure returns None):
        1.  PreTradeChecker.run_all()      — fundamental validity gates.
        2.  CircuitBreaker.check()         — reject when level >= 2.
        3.  PropFirmCompliance.pre_trade_check() — reject on prop-firm breach.
        4.  KellySizer.compute_base_risk() — base risk amount in account currency.
        5.  Volatility scalar              — derived from ``atr_ratio`` feature.
        6.  PortfolioRiskManager.compute_sizing_multipliers() — heat & correlation.
        7.  Final lot size                 — effective risk clipped, then sized.
        8.  StopTargetEngine.compute_stops()      — validate stop distance.
        9.  StopTargetEngine.validate_rr_after_stops() — post-stop R:R check.
        10. Build and return TradeOrder.

        Args:
            signal:         :class:`~core.signals.signal_router.SignalOutput`.
            account_state:  Live account metrics dict (see PreTradeChecker docs).
            features:       Feature DataFrame with the most recent bars as rows.
                            Must contain an ``atr_ratio`` column when available.
            open_positions: Currently open position objects.

        Returns:
            A fully populated :class:`TradeOrder`, or ``None`` on rejection.
        """
        asset:     str = str(getattr(signal, "asset", "UNKNOWN"))
        direction: str = str(getattr(signal, "signal", "UNKNOWN"))
        module:    str = str(getattr(signal, "module", "UNKNOWN"))
        timeframe: str = str(getattr(signal, "timeframe", "M15"))

        # ------------------------------------------------------------------
        # Step 1: Pre-trade checklist
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 1 — pre-trade checks", asset, direction)
        all_passed, reasons = self._pre_trade.run_all(signal, account_state)
        if not all_passed:
            logger.info(
                "RiskEngine [%s %s]: rejected at pre-trade checks — %s",
                asset, direction, reasons,
            )
            return None

        # ------------------------------------------------------------------
        # Step 2: Circuit breaker
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 2 — circuit breaker", asset, direction)
        recent_trades: list = account_state.get("recent_trades", [])
        cb_level, cb_desc = self._cb.check(account_state, recent_trades, open_positions)

        if self._cb.is_trading_halted(cb_level):
            logger.info(
                "RiskEngine [%s %s]: rejected — circuit breaker level %d: %s",
                asset, direction, cb_level, cb_desc,
            )
            return None

        cb_size_mult: float = self._cb.get_size_multiplier(cb_level)
        logger.debug(
            "RiskEngine [%s %s]: CB level=%d size_mult=%.2f",
            asset, direction, cb_level, cb_size_mult,
        )

        # ------------------------------------------------------------------
        # Step 3: Prop-firm compliance
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 3 — prop-firm compliance", asset, direction)
        balance: float = float(account_state.get("balance", 0.0))

        # Derive a preliminary risk estimate for the prop-firm check.
        # Use balance * base_risk as a conservative upper bound before Kelly.
        preliminary_risk = balance * self._max_risk
        pf_passed, pf_reason = self._prop_firm.pre_trade_check(
            new_risk=preliminary_risk,
            account_state=account_state,
        )
        if not pf_passed:
            logger.info(
                "RiskEngine [%s %s]: rejected — prop-firm compliance: %s",
                asset, direction, pf_reason,
            )
            return None

        # ------------------------------------------------------------------
        # Step 4: Kelly base risk
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 4 — Kelly base risk", asset, direction)
        confidence: float = float(getattr(signal, "confidence", 0.55))
        rr_ratio:   float = float(getattr(signal, "rr_ratio", 2.0))

        base_risk_amount: float = self._kelly.compute_base_risk(
            account_balance=balance,
            p_win=confidence,
            rr_ratio=rr_ratio,
        )
        base_risk_pct: float = base_risk_amount / balance if balance > 0 else 0.0

        logger.debug(
            "RiskEngine [%s %s]: base_risk=%.4f (%.3f%%)",
            asset, direction, base_risk_amount, base_risk_pct * 100,
        )

        # ------------------------------------------------------------------
        # Step 5: Volatility scalar
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 5 — volatility scalar", asset, direction)
        atr_ratio: float = 1.0
        if features is not None and not features.empty and "atr_ratio" in features.columns:
            try:
                atr_ratio = float(features["atr_ratio"].iloc[-1])
            except Exception:  # noqa: BLE001
                atr_ratio = 1.0

        if atr_ratio > 1.5:
            volatility_scalar: float = 0.70
        elif atr_ratio > 1.2:
            volatility_scalar = 0.85
        elif atr_ratio < 0.70:
            volatility_scalar = 0.90
        else:
            volatility_scalar = 1.0

        logger.debug(
            "RiskEngine [%s %s]: atr_ratio=%.3f -> volatility_scalar=%.2f",
            asset, direction, atr_ratio, volatility_scalar,
        )

        # ------------------------------------------------------------------
        # Step 6: Portfolio sizing multipliers (heat + correlation)
        # ------------------------------------------------------------------
        logger.debug(
            "RiskEngine [%s %s]: Step 6 — portfolio sizing multipliers",
            asset, direction,
        )
        portfolio_result: dict = self._portfolio.compute_sizing_multipliers(
            new_signal=signal,
            open_positions=open_positions,
            account_balance=balance,
            returns_data=None,   # returns data is not passed through account_state
        )

        heat_ok:            bool  = bool(portfolio_result.get("heat_ok", True))
        correlation_mult:   float = float(portfolio_result.get("correlation_multiplier", 1.0))
        heat_reason:        str   = str(portfolio_result.get("heat_reason", ""))

        if not heat_ok:
            logger.info(
                "RiskEngine [%s %s]: rejected — portfolio heat: %s",
                asset, direction, heat_reason,
            )
            return None

        # ------------------------------------------------------------------
        # Extract regime multipliers from signal.sizing_inputs / regime_context
        # ------------------------------------------------------------------
        regime_ctx    = getattr(signal, "regime_context", None)
        sizing_inputs = getattr(signal, "sizing_inputs", {})

        global_mult: float = float(
            getattr(regime_ctx, "global_multiplier", None)
            if regime_ctx is not None else sizing_inputs.get("global_multiplier", 1.0)
        )
        alignment_mult: float = float(
            getattr(regime_ctx, "alignment_sizing_multiplier", None)
            if regime_ctx is not None else sizing_inputs.get("alignment_multiplier", 1.0)
        )
        age_mult: float = float(
            getattr(regime_ctx, "regime_age_multiplier", None)
            if regime_ctx is not None else sizing_inputs.get("age_multiplier", 1.0)
        )
        # Kelly fraction (already baked into base_risk via half_kelly); for
        # audit purposes record it as 1.0 here since KellySizer handles it.
        kelly_multiplier: float = 1.0

        # ------------------------------------------------------------------
        # Step 7: Final effective risk and lot size
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 7 — final lot size", asset, direction)

        effective_risk_pct: float = (
            base_risk_pct
            * volatility_scalar
            * correlation_mult
            * global_mult
            * age_mult
            * alignment_mult
            * cb_size_mult
        )

        # Clip to [min_risk, max_risk]
        effective_risk_pct = max(self._min_risk, min(self._max_risk, effective_risk_pct))
        effective_risk_amount: float = effective_risk_pct * balance

        # Stop in pips for lot sizing
        stop_distance_pips: float = float(getattr(signal, "stop_distance_pips", 0.0))
        if stop_distance_pips <= 0.0:
            logger.info(
                "RiskEngine [%s %s]: rejected — stop_distance_pips=%.5f invalid",
                asset, direction, stop_distance_pips,
            )
            return None

        lot_size: float = self._kelly.compute_lot_size(
            risk_amount=effective_risk_amount,
            stop_distance_pips=stop_distance_pips,
            symbol=asset,
        )

        if lot_size <= 0.0:
            logger.info(
                "RiskEngine [%s %s]: rejected — lot_size=0 "
                "(risk=%.2f stop_pips=%.2f)",
                asset, direction, effective_risk_amount, stop_distance_pips,
            )
            return None

        logger.debug(
            "RiskEngine [%s %s]: "
            "effective_risk_pct=%.4f risk_amount=%.2f lot_size=%.4f",
            asset, direction, effective_risk_pct, effective_risk_amount, lot_size,
        )

        # ------------------------------------------------------------------
        # Step 8: Stop/target validation
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 8 — stop/target validation", asset, direction)
        atr_value: float = float(getattr(signal, "atr", 0.0))
        stops: dict = self._stops.compute_stops(
            signal=signal,
            features=features,
            atr=atr_value,
        )

        if not stops.get("valid", False):
            logger.info(
                "RiskEngine [%s %s]: rejected — stop validation: %s",
                asset, direction, stops.get("reason", ""),
            )
            return None

        # ------------------------------------------------------------------
        # Step 9: R:R check after final stop placement
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 9 — post-stop R:R check", asset, direction)
        rr_valid, final_rr = self._stops.validate_rr_after_stops(
            signal=signal,
            stops=stops,
            module=module,
        )

        if not rr_valid:
            logger.info(
                "RiskEngine [%s %s]: rejected — post-stop R:R=%.2f below minimum",
                asset, direction, final_rr,
            )
            return None

        logger.debug(
            "RiskEngine [%s %s]: post-stop R:R=%.2f — valid",
            asset, direction, final_rr,
        )

        # ------------------------------------------------------------------
        # Step 10: Build TradeOrder
        # ------------------------------------------------------------------
        logger.debug("RiskEngine [%s %s]: Step 10 — building TradeOrder", asset, direction)

        sig_id:       int = getattr(signal, "signal_id", id(signal))
        magic_number: int = int(getattr(signal, "magic_number", 0))
        entry_price:  float = float(getattr(signal, "entry_price", 0.0))
        signal_ts     = getattr(signal, "timestamp", None)

        if signal_ts is None or not isinstance(signal_ts, datetime):
            order_ts = datetime.now(timezone.utc)
        else:
            order_ts = signal_ts if signal_ts.tzinfo else signal_ts.replace(tzinfo=timezone.utc)

        order = TradeOrder(
            signal_id=sig_id,
            magic_number=magic_number,
            symbol=asset,
            timeframe=timeframe,
            direction=direction,
            module=module,
            lot_size=lot_size,
            entry_price=entry_price,
            stop_loss=float(stops["stop_loss"]),
            take_profit_1=float(stops["tp1"]),
            take_profit_2=float(stops["tp2"]),
            atr=atr_value,
            stop_distance_pips=stop_distance_pips,
            rr_ratio=final_rr,
            account_balance=balance,
            base_risk_pct=base_risk_pct,
            kelly_multiplier=kelly_multiplier,
            volatility_scalar=volatility_scalar,
            regime_age_multiplier=age_mult,
            alignment_multiplier=alignment_mult,
            correlation_multiplier=correlation_mult,
            global_multiplier=global_mult,
            final_risk_pct=effective_risk_pct,
            risk_amount_currency=effective_risk_amount,
            timestamp=order_ts,
            regime_context=regime_ctx,
        )

        logger.info(
            "RiskEngine [%s %s]: TradeOrder created — "
            "lots=%.4f entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f rr=%.2f "
            "risk_pct=%.3f%% risk_usd=%.2f",
            asset, direction,
            order.lot_size,
            order.entry_price,
            order.stop_loss,
            order.take_profit_1,
            order.take_profit_2,
            order.rr_ratio,
            order.final_risk_pct * 100,
            order.risk_amount_currency,
        )
        return order


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TradeOrder",
    "RiskEngine",
    "CircuitBreaker",
    "KellySizer",
    "PortfolioRiskManager",
    "PreTradeChecker",
    "PropFirmCompliance",
    "StopTargetEngine",
]
