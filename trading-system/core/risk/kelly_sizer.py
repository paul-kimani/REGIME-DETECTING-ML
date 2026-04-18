"""Kelly criterion position sizer with half-Kelly and lot-step rounding."""

from __future__ import annotations

from core.utils.helpers import pip_value, round_to_lot_step
from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config fallback defaults (used when Config cannot be loaded)
# ---------------------------------------------------------------------------
_DEFAULT_BASE_RISK = 0.010
_DEFAULT_MAX_RISK = 0.015
_DEFAULT_MIN_RISK = 0.002
_DEFAULT_KELLY_MIN_CONFIDENCE = 0.55
_DEFAULT_MIN_LOT = 0.01
_DEFAULT_MAX_LOT = 100.0
_DEFAULT_LOT_STEP = 0.01


def _load_config() -> object | None:
    """Load the application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class KellySizer:
    """Kelly criterion position sizer with half-Kelly and lot-step rounding.

    Implements the standard Kelly formula scaled to half-Kelly for capital
    preservation.  Lot sizes are floored to the broker's minimum lot step and
    clipped to the configured [min_lot, max_lot] range.
    """

    def __init__(self) -> None:
        """Initialise and load sizing parameters from config with fallbacks."""
        cfg = _load_config()

        try:
            sizing = cfg.sizing  # type: ignore[union-attr]
            self._base_risk: float = float(sizing.base_risk_per_trade)
            self._max_risk: float = float(sizing.max_risk_per_trade)
            self._min_risk: float = float(sizing.min_risk_per_trade)
            self._kelly_min_confidence: float = float(sizing.kelly_min_confidence)
        except Exception:  # noqa: BLE001
            logger.warning(
                "KellySizer: failed to read sizing config — using defaults."
            )
            self._base_risk = _DEFAULT_BASE_RISK
            self._max_risk = _DEFAULT_MAX_RISK
            self._min_risk = _DEFAULT_MIN_RISK
            self._kelly_min_confidence = _DEFAULT_KELLY_MIN_CONFIDENCE

        try:
            assets_list = cfg.assets  # type: ignore[union-attr]
            self._assets: list = list(assets_list)
        except Exception:  # noqa: BLE001
            self._assets = []

        logger.debug(
            "KellySizer initialised: base_risk=%.3f max_risk=%.3f "
            "min_risk=%.3f kelly_min_conf=%.2f",
            self._base_risk,
            self._max_risk,
            self._min_risk,
            self._kelly_min_confidence,
        )

    # ------------------------------------------------------------------
    # Kelly formula
    # ------------------------------------------------------------------

    def kelly_fraction(self, p_win: float, rr_ratio: float) -> float:
        """Compute the full Kelly fraction.

        Uses the standard formula:  f* = (p * b - q) / b
        where b = rr_ratio (reward-to-risk) and q = 1 - p_win.

        Args:
            p_win:    Estimated probability of a winning trade (0 < p_win < 1).
            rr_ratio: Reward-to-risk ratio (e.g. 2.0 means 2:1 RR).

        Returns:
            Kelly fraction in [0, 1], clamped to 0.0 when the edge is negative.
        """
        if rr_ratio <= 0:
            logger.warning("kelly_fraction: rr_ratio must be positive, got %.4f", rr_ratio)
            return 0.0

        q_lose = 1.0 - p_win
        fraction = (p_win * rr_ratio - q_lose) / rr_ratio
        result = max(0.0, fraction)

        logger.debug(
            "kelly_fraction: p_win=%.4f rr=%.4f -> f*=%.4f",
            p_win, rr_ratio, result,
        )
        return result

    def half_kelly(self, p_win: float, rr_ratio: float) -> float:
        """Compute the half-Kelly fraction (f* / 2).

        Half-Kelly significantly reduces drawdown variance at the cost of
        slightly lower long-run growth compared to full Kelly.

        Args:
            p_win:    Estimated probability of a winning trade.
            rr_ratio: Reward-to-risk ratio.

        Returns:
            Half-Kelly fraction in [0, 0.5].
        """
        return self.kelly_fraction(p_win, rr_ratio) * 0.5

    # ------------------------------------------------------------------
    # Risk amount computation
    # ------------------------------------------------------------------

    def compute_base_risk(
        self,
        account_balance: float,
        p_win: float,
        rr_ratio: float,
    ) -> float:
        """Compute the risk amount in account currency for one trade.

        Steps:
        1. If p_win < kelly_min_confidence: use base_risk_per_trade directly
           (no Kelly scaling — insufficient confidence to size by edge).
        2. Else: compute half-Kelly fraction and multiply by account_balance.
        3. Clip the resulting fraction to [min_risk_per_trade, max_risk_per_trade].
        4. Return the absolute risk amount in account currency.

        Args:
            account_balance: Current account equity in account currency.
            p_win:           Model-estimated win probability for this trade.
            rr_ratio:        Reward-to-risk ratio of the trade.

        Returns:
            Risk amount in account currency (e.g. USD).
        """
        if account_balance <= 0:
            logger.warning("compute_base_risk: account_balance must be positive, got %.2f", account_balance)
            return 0.0

        if p_win < self._kelly_min_confidence:
            # Insufficient confidence — fall back to flat base risk
            fraction = self._base_risk
            logger.debug(
                "compute_base_risk: p_win=%.4f < kelly_min_conf=%.2f "
                "-> using flat base_risk=%.3f",
                p_win, self._kelly_min_confidence, fraction,
            )
        else:
            fraction = self.half_kelly(p_win, rr_ratio)
            logger.debug(
                "compute_base_risk: p_win=%.4f rr=%.4f -> half_kelly=%.4f",
                p_win, rr_ratio, fraction,
            )

        # Clip to [min_risk, max_risk]
        clipped = max(self._min_risk, min(self._max_risk, fraction))

        risk_amount = clipped * account_balance

        logger.debug(
            "compute_base_risk: fraction=%.4f (clipped=%.4f) "
            "balance=%.2f -> risk_amount=%.2f",
            fraction, clipped, account_balance, risk_amount,
        )
        return risk_amount

    # ------------------------------------------------------------------
    # Lot size computation
    # ------------------------------------------------------------------

    def compute_lot_size(
        self,
        risk_amount: float,
        stop_distance_pips: float,
        symbol: str,
        account_currency: str = "USD",
    ) -> float:
        """Convert a risk amount into a lot size for the given instrument.

        Formula::

            lot_size = risk_amount / (stop_distance_pips * pip_value_per_lot)

        The raw lot size is floored to the broker's lot step, then clipped to
        [min_lot, max_lot].  Returns 0.0 if the result would fall below min_lot.

        Args:
            risk_amount:        Maximum loss in account currency for this trade.
            stop_distance_pips: Distance from entry to stop-loss in pips.
            symbol:             Instrument identifier, e.g. "EURUSD".
            account_currency:   Three-letter ISO account currency (default "USD").

        Returns:
            Lot size rounded down to the nearest lot step, or 0.0 if too small.
        """
        if stop_distance_pips <= 0:
            logger.warning(
                "compute_lot_size: stop_distance_pips must be positive, got %.4f",
                stop_distance_pips,
            )
            return 0.0

        if risk_amount <= 0:
            logger.warning(
                "compute_lot_size: risk_amount must be positive, got %.4f",
                risk_amount,
            )
            return 0.0

        # Retrieve lot step, min_lot, max_lot from assets config
        lot_step, min_lot, max_lot = self._get_lot_params(symbol)

        # pip_value() returns pip value for the given lot_size — use 1 lot as
        # the per-lot rate then scale
        pip_value_per_lot = pip_value(symbol, 1.0, account_currency)

        if pip_value_per_lot <= 0:
            logger.warning(
                "compute_lot_size: pip_value_per_lot is %.6f for %s — returning 0.0",
                pip_value_per_lot, symbol,
            )
            return 0.0

        raw_lots = risk_amount / (stop_distance_pips * pip_value_per_lot)

        # Floor to lot step
        stepped_lots = round_to_lot_step(raw_lots, lot_step)

        # Clip to [min_lot, max_lot]
        if stepped_lots < min_lot:
            logger.debug(
                "compute_lot_size: stepped_lots=%.4f < min_lot=%.4f for %s "
                "-> returning 0.0",
                stepped_lots, min_lot, symbol,
            )
            return 0.0

        final_lots = min(stepped_lots, max_lot)

        logger.debug(
            "compute_lot_size: %s risk=%.2f stop_pips=%.2f pip_val=%.4f "
            "raw=%.4f stepped=%.4f final=%.4f",
            symbol, risk_amount, stop_distance_pips,
            pip_value_per_lot, raw_lots, stepped_lots, final_lots,
        )
        return final_lots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lot_params(self, symbol: str) -> tuple[float, float, float]:
        """Return (lot_step, min_lot, max_lot) for *symbol* from assets config.

        Falls back to default values if the symbol is not found in config.

        Args:
            symbol: Instrument identifier, e.g. "EURUSD".

        Returns:
            Tuple of (lot_step, min_lot, max_lot).
        """
        symbol_upper = symbol.upper().strip()
        for asset in self._assets:
            try:
                if str(asset.symbol).upper() == symbol_upper:
                    return (
                        float(asset.lot_step),
                        float(asset.min_lot),
                        float(asset.max_lot),
                    )
            except AttributeError:
                continue

        logger.debug(
            "_get_lot_params: %s not found in assets config — using defaults "
            "(step=%.2f min=%.2f max=%.2f)",
            symbol, _DEFAULT_LOT_STEP, _DEFAULT_MIN_LOT, _DEFAULT_MAX_LOT,
        )
        return _DEFAULT_LOT_STEP, _DEFAULT_MIN_LOT, _DEFAULT_MAX_LOT
