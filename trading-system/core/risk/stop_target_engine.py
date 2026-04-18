"""Stop-loss and take-profit placement engine with ATR-based trailing stops."""

from __future__ import annotations

import pandas as pd

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------
_DEFAULT_MAX_STOP_ATR_MULTIPLE = 2.0
_DEFAULT_BREAKOUT_MAX_STOP_ATR = 1.5
_DEFAULT_TRAIL_ATR_MULTIPLE = 1.0

_DEFAULT_RR_MINIMUMS: dict[str, float] = {
    "momentum": 2.0,
    "mean_reversion": 1.5,
    "breakout": 2.5,
}


def _load_config() -> object | None:
    """Load application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class StopTargetEngine:
    """Stop-loss and take-profit placement with ATR-based trailing stops.

    Validates that stops produced by the signal modules are within acceptable
    ATR multiples, computes trailing stop updates, and re-checks R:R ratios
    after final stop placement.
    """

    def __init__(self) -> None:
        """Initialise and load stop/RR parameters from config with fallbacks."""
        cfg = _load_config()

        try:
            stop_limits = cfg.stop_limits  # type: ignore[union-attr]
            self._max_stop_atr_multiple: float = float(stop_limits.max_stop_atr_multiple)
        except Exception:  # noqa: BLE001
            logger.warning(
                "StopTargetEngine: failed to read stop_limits config — using defaults."
            )
            self._max_stop_atr_multiple = _DEFAULT_MAX_STOP_ATR_MULTIPLE

        try:
            rr_cfg = cfg.rr_minimums  # type: ignore[union-attr]
            self._rr_minimums: dict[str, float] = {
                "momentum": float(rr_cfg.momentum),
                "mean_reversion": float(rr_cfg.mean_reversion),
                "breakout": float(rr_cfg.breakout),
            }
        except Exception:  # noqa: BLE001
            logger.warning(
                "StopTargetEngine: failed to read rr_minimums config — using defaults."
            )
            self._rr_minimums = dict(_DEFAULT_RR_MINIMUMS)

        # BREAKOUT uses a stricter stop cap (1.5 * ATR vs the general 2.0 * ATR)
        self._breakout_max_stop_atr: float = _DEFAULT_BREAKOUT_MAX_STOP_ATR
        self._trail_atr_multiple: float = _DEFAULT_TRAIL_ATR_MULTIPLE

        logger.debug(
            "StopTargetEngine initialised: max_stop_atr=%.1f breakout_cap=%.1f "
            "trail_mult=%.1f rr_mins=%s",
            self._max_stop_atr_multiple,
            self._breakout_max_stop_atr,
            self._trail_atr_multiple,
            self._rr_minimums,
        )

    # ------------------------------------------------------------------
    # Stop and target computation / validation
    # ------------------------------------------------------------------

    def compute_stops(
        self,
        signal,
        features: pd.DataFrame,
        atr: float,
    ) -> dict:
        """Compute or validate stop-loss and take-profit levels for *signal*.

        For MOMENTUM and MEAN_REVERSION modules the stops and targets are
        already embedded in the signal by the producing module.  This method
        validates that the stop is not wider than *max_stop_atr_multiple* ATR.

        For BREAKOUT the same validation applies, with a stricter cap of
        1.5 * ATR.

        Args:
            signal:   SignalOutput from a signal module.  Expected attributes:
                      ``entry_price``, ``stop_loss``, ``take_profit_1``,
                      ``take_profit_2``, ``module``.
            features: Feature DataFrame for the current bar (may be used by
                      future extensions; unused here).
            atr:      Current ATR value for the instrument (14-bar Wilder ATR).

        Returns:
            Dict with keys:
            - ``"stop_loss"``   (float) Validated stop-loss price.
            - ``"tp1"``         (float) First take-profit price.
            - ``"tp2"``         (float) Second take-profit price.
            - ``"valid"``       (bool)  False when the stop is too wide.
            - ``"reason"``      (str)   Human-readable validation result.
        """
        entry: float = float(getattr(signal, "entry_price", 0.0))
        stop: float = float(getattr(signal, "stop_loss", 0.0))
        tp1: float = float(getattr(signal, "take_profit_1", 0.0))
        tp2: float = float(getattr(signal, "take_profit_2", 0.0))
        module: str = str(getattr(signal, "module", "")).upper()

        if atr <= 0:
            reason = f"invalid ATR={atr:.6f}"
            logger.warning("compute_stops: %s", reason)
            return {
                "stop_loss": stop, "tp1": tp1, "tp2": tp2,
                "valid": False, "reason": reason,
            }

        stop_distance = abs(entry - stop)

        # Select the applicable ATR cap
        if module == "BREAKOUT":
            max_stop = self._breakout_max_stop_atr * atr
        else:
            max_stop = self._max_stop_atr_multiple * atr

        if stop_distance > max_stop:
            reason = (
                f"{module}: stop_distance={stop_distance:.5f} > "
                f"max_allowed={max_stop:.5f} ({atr:.5f} * "
                f"{'1.5' if module == 'BREAKOUT' else str(self._max_stop_atr_multiple)} ATR)"
            )
            logger.info("compute_stops: INVALID — %s", reason)
            return {
                "stop_loss": stop, "tp1": tp1, "tp2": tp2,
                "valid": False, "reason": reason,
            }

        reason = (
            f"{module}: stop_distance={stop_distance:.5f} within "
            f"max={max_stop:.5f} — valid"
        )
        logger.debug("compute_stops: %s", reason)
        return {
            "stop_loss": stop, "tp1": tp1, "tp2": tp2,
            "valid": True, "reason": reason,
        }

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def compute_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        atr: float,
        direction: str,
    ) -> float:
        """Compute a new ATR-based trailing stop level.

        Trail distance is ``trail_atr_multiple * atr`` (default 1.0 * ATR).
        The stop is only ever moved in the favourable direction — it is never
        widened.

        - LONG:  new_stop = current_price - 1.0 * atr  (move UP only)
        - SHORT: new_stop = current_price + 1.0 * atr  (move DOWN only)

        Args:
            current_price: Latest market price.
            current_stop:  Existing stop-loss level.
            atr:           Current ATR value (14-bar Wilder ATR).
            direction:     "LONG" or "SHORT".

        Returns:
            Updated stop-loss price.  Equals *current_stop* if the new level
            would be worse (i.e. further from the current price).
        """
        trail_dist = self._trail_atr_multiple * atr
        dir_upper = direction.upper()

        if dir_upper == "LONG":
            proposed = current_price - trail_dist
            # Never move stop DOWN for a long
            new_stop = max(proposed, current_stop)
        elif dir_upper == "SHORT":
            proposed = current_price + trail_dist
            # Never move stop UP for a short
            new_stop = min(proposed, current_stop)
        else:
            logger.warning(
                "compute_trailing_stop: unknown direction '%s' — returning current_stop",
                direction,
            )
            return current_stop

        logger.debug(
            "compute_trailing_stop: %s price=%.5f atr=%.5f "
            "proposed=%.5f current_stop=%.5f -> new_stop=%.5f",
            dir_upper, current_price, atr, proposed, current_stop, new_stop,
        )
        return new_stop

    # ------------------------------------------------------------------
    # R:R validation
    # ------------------------------------------------------------------

    def validate_rr_after_stops(
        self,
        signal,
        stops: dict,
        module: str,
    ) -> tuple[bool, float]:
        """Verify that the R:R ratio remains acceptable after stop placement.

        Uses ``tp2`` (the full target) for the R:R calculation::

            rr = |tp2 - entry| / |entry - stop_loss|

        Module-specific minimum R:R values:
            - MOMENTUM:      2.0
            - MEAN_REVERSION: 1.5
            - BREAKOUT:      2.5

        Args:
            signal: SignalOutput with ``entry_price`` attribute.
            stops:  Dict returned by :meth:`compute_stops` containing
                    ``"stop_loss"`` and ``"tp2"`` keys.
            module: Strategy module name — one of "MOMENTUM", "MEAN_REVERSION",
                    or "BREAKOUT" (case-insensitive).

        Returns:
            Tuple of (valid: bool, rr_ratio: float).
            valid=False when rr_ratio is below the module's minimum threshold.
        """
        entry: float = float(getattr(signal, "entry_price", 0.0))
        stop: float = float(stops.get("stop_loss", 0.0))
        tp2: float = float(stops.get("tp2", 0.0))
        module_key = module.lower()

        stop_dist = abs(entry - stop)
        if stop_dist == 0.0:
            logger.warning(
                "validate_rr_after_stops: zero stop distance for entry=%.5f "
                "stop=%.5f", entry, stop,
            )
            return False, 0.0

        rr_ratio = abs(tp2 - entry) / stop_dist
        min_rr = self._rr_minimums.get(module_key, _DEFAULT_RR_MINIMUMS.get(module_key, 2.0))
        valid = rr_ratio >= min_rr

        logger.debug(
            "validate_rr_after_stops: module=%s rr=%.4f min_rr=%.2f valid=%s",
            module_key, rr_ratio, min_rr, valid,
        )
        return valid, rr_ratio
