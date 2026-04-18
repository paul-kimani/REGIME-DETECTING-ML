"""SignalRouter — routes regime state to the correct signal module."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd

from core.utils.logger import get_logger

if TYPE_CHECKING:
    from core.regime.regime_detector import RegimeState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SignalOutput dataclass — shared across all signal modules and risk engine
# ---------------------------------------------------------------------------


@dataclass
class SignalOutput:
    """Complete signal specification passed downstream to the risk engine.

    Produced by any of the three strategy modules (Momentum, MeanReversion,
    Breakout) and validated/enriched by SignalRouter before being handed to
    the RiskEngine.
    """

    asset: str
    timeframe: str
    timestamp: datetime
    signal: str                # "LONG" | "SHORT" | "NO_TRADE"
    module: str                # "MOMENTUM" | "MEAN_REVERSION" | "BREAKOUT"
    confidence: float

    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    atr: float
    stop_distance_pips: float
    rr_ratio: float

    regime_context: Optional[object] = None   # RegimeState (avoid circular import)
    sizing_inputs: dict = field(default_factory=dict)
    magic_number: int = 0


# ---------------------------------------------------------------------------
# SignalRouter
# ---------------------------------------------------------------------------


class SignalRouter:
    """Orchestrates all three signal modules and routes to the right one.

    Reads the active_strategy field of the RegimeState to select which
    module to call, then validates and enriches the returned SignalOutput
    before passing it to the risk engine.
    """

    def __init__(self, momentum_module, mean_reversion_module, breakout_module) -> None:
        """Store references to the three signal modules.

        Args:
            momentum_module:       Initialised MomentumModule instance.
            mean_reversion_module: Initialised MeanReversionModule instance.
            breakout_module:       Initialised BreakoutModule instance.
        """
        self._momentum = momentum_module
        self._mean_reversion = mean_reversion_module
        self._breakout = breakout_module
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        regime_state: "RegimeState",
        latest_bar: dict,
    ) -> Optional[SignalOutput]:
        """Route to the correct module and return a validated SignalOutput.

        Steps:
        1. Abort if models are not trained (final_sizing_multiplier == 0).
        2. Select module based on regime_state.active_strategy.
        3. Call module.predict(features, regime_state, latest_bar).
        4. Validate the returned signal.
        5. Enrich with regime context and magic number.

        Args:
            symbol:        Instrument identifier (e.g. "XAUUSD").
            timeframe:     Active timeframe string (e.g. "M15").
            features:      Feature DataFrame with the latest bars as rows.
            regime_state:  Current RegimeState from RegimeDetector.
            latest_bar:    Dict with current bid, ask, atr, high, low, close.

        Returns:
            Enriched :class:`SignalOutput` or ``None`` if no tradeable signal.
        """
        # Gate: models must be trained
        if regime_state.final_sizing_multiplier <= 0.0:
            self._log.debug(
                "%s — skipping signal: models not trained (multiplier=0)", symbol
            )
            return None

        strategy = regime_state.active_strategy
        self._log.debug("%s %s — active strategy: %s", symbol, timeframe, strategy)

        signal: Optional[SignalOutput] = None

        if strategy == "momentum":
            signal = self._momentum.predict(features, regime_state, latest_bar)
        elif strategy == "mean_reversion":
            signal = self._mean_reversion.predict(features, regime_state, latest_bar)
        elif strategy == "breakout":
            signal = self._breakout.predict(features, regime_state, latest_bar)
        else:
            self._log.debug("%s — strategy '%s' → no_trade", symbol, strategy)
            return None

        if signal is None:
            self._log.debug("%s %s — module returned no signal", symbol, strategy)
            return None

        if not self._validate_signal(signal):
            self._log.warning(
                "%s — signal failed basic validation: %s", symbol, signal
            )
            return None

        # Enrich
        signal.regime_context = regime_state
        signal.magic_number = self._generate_magic_number(symbol, signal.timestamp)
        signal.sizing_inputs = {
            "final_sizing_multiplier": regime_state.final_sizing_multiplier,
            "global_multiplier": regime_state.global_multiplier,
            "alignment_multiplier": regime_state.alignment_sizing_multiplier,
            "age_multiplier": regime_state.regime_age_multiplier,
            "confidence": signal.confidence,
            "regime": regime_state.m15_regime,
            "global_risk_state": regime_state.global_risk_state,
        }

        self._log.info(
            "%s %s — %s signal via %s | conf=%.2f rr=%.2f",
            symbol,
            timeframe,
            signal.signal,
            signal.module,
            signal.confidence,
            signal.rr_ratio,
        )
        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_magic_number(self, symbol: str, timestamp: datetime) -> int:
        """Generate a unique 8-digit magic number for MT5 order tagging.

        Combines a unix timestamp with a symbol hash so that concurrent
        signals for different assets or timeframes never collide.

        Args:
            symbol:    Instrument symbol string.
            timestamp: Signal timestamp.

        Returns:
            Integer magic number in range [10_000_000, 99_999_999].
        """
        ts_int = int(timestamp.timestamp()) if hasattr(timestamp, "timestamp") else 0
        sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:4], 16)
        raw = (ts_int * 10000 + sym_hash) % 90_000_000 + 10_000_000
        return raw

    def _validate_signal(self, signal: SignalOutput) -> bool:
        """Perform basic sanity checks on a returned signal.

        Checks:
        - signal direction is LONG or SHORT (not NO_TRADE).
        - entry_price, stop_loss, take_profit_1, take_profit_2 are all positive.
        - rr_ratio is positive.
        - stop_loss is strictly below entry for LONG and above for SHORT.

        Args:
            signal: The :class:`SignalOutput` to validate.

        Returns:
            ``True`` if all checks pass, ``False`` otherwise.
        """
        if signal.signal not in ("LONG", "SHORT"):
            return False
        if signal.entry_price <= 0 or signal.stop_loss <= 0:
            return False
        if signal.take_profit_1 <= 0 or signal.take_profit_2 <= 0:
            return False
        if signal.rr_ratio <= 0:
            return False
        if signal.signal == "LONG" and signal.stop_loss >= signal.entry_price:
            return False
        if signal.signal == "SHORT" and signal.stop_loss <= signal.entry_price:
            return False
        return True
