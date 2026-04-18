"""Multi-timeframe alignment scoring and active strategy selection."""

from __future__ import annotations

from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Regime label constants (mirrors hmm_model.py)
# ---------------------------------------------------------------------------
_TREND_UP   = "TREND_UP"
_TREND_DOWN = "TREND_DOWN"
_RANGE      = "RANGE"
_VOLATILE   = "VOLATILE"

_TREND_REGIMES: frozenset[str] = frozenset({_TREND_UP, _TREND_DOWN})

# ---------------------------------------------------------------------------
# Default weights (overridden by config when available)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {"H4": 0.50, "H1": 0.30, "M15": 0.20}


class MTFAlignment:
    """Multi-timeframe alignment scoring and active strategy selection.

    Computes a weighted alignment score across three timeframes (H4, H1,
    M15) and maps the result to a position-sizing multiplier and an active
    trading strategy.
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

        # Load weights from config if available; fall back to defaults.
        try:
            from core.utils.config import get_config  # local import for safety
            cfg = get_config()
            w = cfg.mtf.weights
            self._weights: dict[str, float] = {
                "H4":  float(w.H4),
                "H1":  float(w.H1),
                "M15": float(w.M15),
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "MTFAlignment: could not load weights from config (%s). "
                "Using defaults %s.",
                exc,
                _DEFAULT_WEIGHTS,
            )
            self._weights = dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_alignment(
        self,
        regimes: dict[str, str],
        confidences: dict[str, float],
    ) -> dict:
        """Compute MTF alignment score and determine active strategy.

        Parameters
        ----------
        regimes:
            Mapping of timeframe to regime string, e.g.
            ``{"H4": "TREND_UP", "H1": "RANGE", "M15": "TREND_UP"}``.
        confidences:
            Mapping of timeframe to confidence score in ``[0, 1]``, e.g.
            ``{"H4": 0.80, "H1": 0.65, "M15": 0.72}``.

        Alignment scoring rules
        -----------------------
        * All three same trend direction            -> raw score 1.0
        * H4 + H1 agree on trend, M15 different    -> raw score 0.75
        * H4 agrees with at least one (but not all) -> raw score 0.50
        * All different                             -> raw score 0.25
        * Any timeframe is VOLATILE                 -> subtract 0.20 (min 0.0)
        * Weight-adjusted score: apply weights {H4: 0.50, H1: 0.30, M15: 0.20}

        Returns
        -------
        dict with keys:
            ``alignment_score`` (float),
            ``h4_regime`` (str),
            ``h1_regime`` (str),
            ``m15_regime`` (str),
            ``active_strategy`` (str),
            ``sizing_multiplier`` (float),
            ``is_aligned`` (bool).
        """
        h4  = regimes.get("H4",  _RANGE)
        h1  = regimes.get("H1",  _RANGE)
        m15 = regimes.get("M15", _RANGE)

        # ---- Raw structural score ----------------------------------------
        if h4 == h1 == m15 and h4 in _TREND_REGIMES:
            raw_score = 1.0
        elif (h4 == h1) and (h4 in _TREND_REGIMES) and (m15 != h4):
            raw_score = 0.75
        elif h4 in {h1, m15}:
            # H4 agrees with at least one, not all three the same
            raw_score = 0.50
        else:
            raw_score = 0.25

        # ---- VOLATILE penalty -----------------------------------------------
        if _VOLATILE in (h4, h1, m15):
            raw_score = max(0.0, raw_score - 0.20)

        # ---- Confidence-weighted adjustment ---------------------------------
        # Weight the per-timeframe confidences by their structural weights and
        # use them to modulate the raw structural score.
        w_h4  = self._weights["H4"]
        w_h1  = self._weights["H1"]
        w_m15 = self._weights["M15"]

        c_h4  = float(confidences.get("H4",  0.5))
        c_h1  = float(confidences.get("H1",  0.5))
        c_m15 = float(confidences.get("M15", 0.5))

        weighted_confidence = (
            w_h4  * c_h4
            + w_h1  * c_h1
            + w_m15 * c_m15
        )
        # Blend: structural score adjusted proportionally by average confidence
        alignment_score = float(round(raw_score * weighted_confidence / 0.5, 4))
        alignment_score = max(0.0, min(1.0, alignment_score))

        active_strategy   = self.determine_active_strategy(h4, h1, m15)
        sizing_multiplier = self.get_sizing_multiplier(alignment_score)
        is_aligned        = alignment_score > 0.60

        self._logger.debug(
            "MTFAlignment — H4=%s H1=%s M15=%s | raw=%.2f conf_w=%.3f "
            "score=%.4f strategy=%s multiplier=%.2f",
            h4, h1, m15,
            raw_score, weighted_confidence,
            alignment_score, active_strategy, sizing_multiplier,
        )

        return {
            "alignment_score":    alignment_score,
            "h4_regime":          h4,
            "h1_regime":          h1,
            "m15_regime":         m15,
            "active_strategy":    active_strategy,
            "sizing_multiplier":  sizing_multiplier,
            "is_aligned":         is_aligned,
        }

    def determine_active_strategy(self, h4: str, h1: str, m15: str) -> str:
        """Determine which strategy module to route to.

        Parameters
        ----------
        h4:
            Regime label for the H4 timeframe.
        h1:
            Regime label for the H1 timeframe.
        m15:
            Regime label for the M15 timeframe.

        Logic (evaluated top-to-bottom, first match wins)
        -------------------------------------------------
        1. Any VOLATILE              -> ``"breakout"``
        2. H4 trend AND H1 same trend direction -> ``"momentum"``
        3. H4 == RANGE AND H1 == RANGE          -> ``"mean_reversion"``
        4. H4 == RANGE AND H1 is trend          -> ``"mean_reversion"``
        5. H4 is trend AND H1 == RANGE          -> ``"mean_reversion"``
        6. Otherwise                             -> ``"no_trade"``

        Returns
        -------
        str
            One of ``"momentum"``, ``"mean_reversion"``, ``"breakout"``,
            or ``"no_trade"``.
        """
        if _VOLATILE in (h4, h1, m15):
            return "breakout"

        if h4 in _TREND_REGIMES and h1 in _TREND_REGIMES and h4 == h1:
            return "momentum"

        if h4 == _RANGE and h1 == _RANGE:
            return "mean_reversion"

        if h4 == _RANGE and h1 in _TREND_REGIMES:
            return "mean_reversion"

        if h4 in _TREND_REGIMES and h1 == _RANGE:
            return "mean_reversion"

        return "no_trade"

    def get_sizing_multiplier(self, alignment_score: float) -> float:
        """Convert an alignment score to a position sizing multiplier.

        Parameters
        ----------
        alignment_score:
            Normalised alignment score in ``[0.0, 1.0]``.

        Returns
        -------
        float
            * ``1.00`` when score >= 0.85
            * ``0.75`` when score >= 0.60
            * ``0.50`` when score >= 0.40
            * ``0.25`` when score < 0.40
        """
        if alignment_score >= 0.85:
            return 1.00
        if alignment_score >= 0.60:
            return 0.75
        if alignment_score >= 0.40:
            return 0.50
        return 0.25
