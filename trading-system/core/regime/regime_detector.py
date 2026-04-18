"""RegimeDetector — orchestrates HMM, XGBoost, Universal, and MTF components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.regime.hmm_model import HMMRegimeModel, REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_VOLATILE
from core.regime.xgb_classifier import XGBRegimeClassifier
from core.regime.universal_model import UniversalModel
from core.regime.mtf_alignment import MTFAlignment
from core.utils.logger import get_logger
from core.utils.config import get_config

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegimeState:
    """Complete regime state snapshot for a single symbol at a point in time."""

    symbol: str
    timestamp: datetime

    # Universal
    global_risk_state: str = "NEUTRAL"
    global_multiplier: float = 0.7

    # Per timeframe
    h4_regime: str = "RANGE"
    h4_confidence: float = 0.5
    h4_hmm_probs: dict = field(default_factory=dict)
    h1_regime: str = "RANGE"
    h1_confidence: float = 0.5
    h1_hmm_probs: dict = field(default_factory=dict)
    m15_regime: str = "RANGE"
    m15_confidence: float = 0.5
    m15_hmm_probs: dict = field(default_factory=dict)

    # Alignment
    alignment_score: float = 0.5
    alignment_sizing_multiplier: float = 0.75
    active_strategy: str = "no_trade"

    # Regime state
    regime_confirmed: bool = False
    regime_age_candles: int = 0
    regime_maturity_flag: str = "young"
    regime_age_multiplier: float = 1.0

    # Final
    final_sizing_multiplier: float = 0.0
    strategy_module: str = "no_trade"


# ---------------------------------------------------------------------------
# Persistence config defaults
# ---------------------------------------------------------------------------

_DEFAULT_PERSISTENCE_CONFIG: dict = {
    "probability_threshold_base": 0.65,
    "thresholds_by_current_regime": {
        "trend_up":   {"to_range": 0.70, "to_volatile": 0.60, "to_trend_down": 0.65},
        "trend_down": {"to_range": 0.70, "to_volatile": 0.60, "to_trend_up":   0.65},
        "range":      {"to_trend_up": 0.60, "to_trend_down": 0.60, "to_volatile": 0.55},
        "volatile":   {"to_any": 0.75},
    },
    "candle_confirmation": {
        "volatile_to_any":  1,
        "range_to_trend":   3,
        "trend_to_range":   4,
        "trend_to_trend":   2,
    },
    "dynamic_n_adjustment": {
        "high_atr_ratio_threshold": 1.4,
        "high_atr_n_reduction":     1,
        "low_atr_ratio_threshold":  0.8,
        "low_atr_n_increase":       1,
    },
}

_DEFAULT_REGIME_AGE_CONFIG: dict = {
    "young":         {"max_candles": 20,   "multiplier": 1.00},
    "mature":        {"max_candles": 50,   "multiplier": 0.85},
    "extended":      {"max_candles": 100,  "multiplier": 0.70},
    "very_extended": {"max_candles": 9999, "multiplier": 0.50},
}

# Timeframes to process in detection order
_TIMEFRAMES: list[str] = ["H4", "H1", "M15"]


def _load_persistence_config() -> dict:
    """Return persistence config from YAML; fall back to defaults on error."""
    try:
        cfg = get_config()
        raw = cfg.persistence.as_dict()
        return raw
    except Exception:  # noqa: BLE001
        return dict(_DEFAULT_PERSISTENCE_CONFIG)


def _load_regime_age_config() -> dict:
    """Return regime-age multiplier config from YAML; fall back to defaults."""
    try:
        cfg = get_config()
        raw = cfg.regime_age_multipliers.as_dict()
        return raw
    except Exception:  # noqa: BLE001
        return dict(_DEFAULT_REGIME_AGE_CONFIG)


def _transition_key(current: str, candidate: str) -> str:
    """Build the config key for a regime transition.

    Returns a string such as ``"to_range"``, ``"to_trend_up"``, etc.
    """
    return "to_" + candidate.lower().replace(" ", "_")


def _required_candles(current: str, candidate: str, persistence_config: dict) -> int:
    """Look up base candle-confirmation count for a given transition."""
    cc: dict = persistence_config.get("candle_confirmation", {})

    current_lower   = current.lower()
    candidate_lower = candidate.lower()

    if current_lower == "volatile":
        return int(cc.get("volatile_to_any", 1))
    if current_lower == "range" and "trend" in candidate_lower:
        return int(cc.get("range_to_trend", 3))
    if "trend" in current_lower and candidate_lower == "range":
        return int(cc.get("trend_to_range", 4))
    if "trend" in current_lower and "trend" in candidate_lower:
        return int(cc.get("trend_to_trend", 2))
    # generic fallback
    return int(cc.get("range_to_trend", 3))


def _probability_threshold(current: str, candidate: str, persistence_config: dict) -> float:
    """Return the required probability threshold for switching regimes."""
    base: float = float(persistence_config.get("probability_threshold_base", 0.65))
    by_regime: dict = persistence_config.get("thresholds_by_current_regime", {})
    current_thresholds: dict = by_regime.get(current.lower(), {})

    key = _transition_key(current, candidate)
    if key in current_thresholds:
        return float(current_thresholds[key])
    # volatile special case — "to_any" key
    if "to_any" in current_thresholds:
        return float(current_thresholds["to_any"])
    return base


# ---------------------------------------------------------------------------
# RegimeStateMachine
# ---------------------------------------------------------------------------


class RegimeStateMachine:
    """Dual-gate regime persistence. Prevents premature regime switches.

    Gate 1: probability threshold (from config, different per transition).
    Gate 2: candle confirmation count (dynamic based on ATR ratio).
    """

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self.current_regime: str = REGIME_RANGE
        self.candidate_regime: Optional[str] = None
        self.candidate_count: int = 0
        self.regime_age: int = 0
        self.regime_start_time: Optional[datetime] = None

    def update(
        self,
        new_regime: str,
        probability: float,
        atr_ratio: float,
        persistence_config: dict,
    ) -> bool:
        """Update state machine with a new regime prediction.

        Parameters
        ----------
        new_regime:
            The regime label predicted by the model for this candle.
        probability:
            Model confidence for *new_regime* in ``[0, 1]``.
        atr_ratio:
            Current ATR ratio (ATR-14 / ATR-50-mean).
        persistence_config:
            Persistence thresholds and candle-count config dict.

        Returns
        -------
        bool
            ``True`` if the current regime actually changed.

        Logic
        -----
        1. If ``new_regime == current_regime``: reset candidate, increment
           age, return ``False``.
        2. If ``new_regime != candidate_regime``: reset candidate counter,
           set candidate to *new_regime*, count = 1.
        3. Gate 1 — probability >= threshold for this transition.
        4. Gate 2 — candidate_count >= required candles (ATR-adjusted).
        5. Both gates pass: switch regime, reset age, return ``True``.
        6. Else: increment candidate_count, return ``False``.
        """
        # --- Step 1: same as current ---
        if new_regime == self.current_regime:
            self.candidate_regime = None
            self.candidate_count  = 0
            self.regime_age      += 1
            return False

        # --- Step 2: new or changed candidate ---
        if new_regime != self.candidate_regime:
            self.candidate_regime = new_regime
            self.candidate_count  = 1
        else:
            self.candidate_count += 1

        # --- Gate 1: probability threshold ---
        threshold = _probability_threshold(
            self.current_regime, new_regime, persistence_config
        )
        if probability < threshold:
            return False

        # --- Gate 2: candle confirmation (ATR-adjusted) ---
        base_candles = _required_candles(
            self.current_regime, new_regime, persistence_config
        )
        dyn: dict = persistence_config.get("dynamic_n_adjustment", {})
        high_thr:  float = float(dyn.get("high_atr_ratio_threshold", 1.4))
        high_red:  int   = int(dyn.get("high_atr_n_reduction",       1))
        low_thr:   float = float(dyn.get("low_atr_ratio_threshold",  0.8))
        low_inc:   int   = int(dyn.get("low_atr_n_increase",         1))

        required = base_candles
        if atr_ratio > high_thr:
            required = max(1, required - high_red)
        elif atr_ratio < low_thr:
            required = required + low_inc

        if self.candidate_count < required:
            return False

        # --- Both gates passed: switch ---
        self.current_regime   = new_regime
        self.candidate_regime = None
        self.candidate_count  = 0
        self.regime_age       = 0
        self.regime_start_time = datetime.now(tz=timezone.utc)
        return True

    def get_maturity_flag(self) -> str:
        """Return regime maturity category based on ``regime_age``.

        Returns
        -------
        str
            One of ``"young"``, ``"mature"``, ``"extended"``, or
            ``"very_extended"``.
        """
        if self.regime_age <= 20:
            return "young"
        if self.regime_age <= 50:
            return "mature"
        if self.regime_age <= 100:
            return "extended"
        return "very_extended"

    def get_age_multiplier(self, regime_age_config: dict) -> float:
        """Look up sizing multiplier from *regime_age_config* for current age.

        Parameters
        ----------
        regime_age_config:
            Dict mapping maturity label to ``{max_candles, multiplier}``,
            e.g. ``{"young": {"max_candles": 20, "multiplier": 1.0}, ...}``.

        Returns
        -------
        float
            Sizing multiplier appropriate for the current regime age.
        """
        for _label, params in regime_age_config.items():
            try:
                if isinstance(params, dict):
                    max_c = int(params.get("max_candles", 9999))
                    mult  = float(params.get("multiplier", 1.0))
                else:
                    # _ConfigNode
                    max_c = int(params.max_candles)
                    mult  = float(params.multiplier)
                if self.regime_age <= max_c:
                    return mult
            except (AttributeError, TypeError, ValueError):
                continue
        return 0.50  # fallback — very extended


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Orchestrates HMM, XGBoost, Universal, and MTF components.

    One instance is typically shared across the entire trading session.
    Models are keyed by ``symbol`` (HMM / XGBoost) and by
    ``symbol + "_" + timeframe`` (state machines).
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

        self.hmm_models: dict[str, HMMRegimeModel]       = {}
        self.xgb_classifiers: dict[str, XGBRegimeClassifier] = {}
        self.universal_model: UniversalModel             = UniversalModel()
        self.mtf_alignment: MTFAlignment                 = MTFAlignment()
        self.state_machines: dict[str, RegimeStateMachine] = {}

        self._persistence_config: dict = _load_persistence_config()
        self._age_config: dict         = _load_regime_age_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_hmm(self, symbol: str, timeframe: str) -> HMMRegimeModel:
        """Return existing or freshly created HMM model for *symbol+timeframe*."""
        key = f"{symbol}_{timeframe}"
        if key not in self.hmm_models:
            self.hmm_models[key] = HMMRegimeModel(symbol=f"{symbol}_{timeframe}")
        return self.hmm_models[key]

    def _get_or_create_xgb(self, symbol: str, timeframe: str) -> XGBRegimeClassifier:
        """Return existing or freshly created XGBoost classifier for *symbol+timeframe*."""
        key = f"{symbol}_{timeframe}"
        if key not in self.xgb_classifiers:
            self.xgb_classifiers[key] = XGBRegimeClassifier(symbol=f"{symbol}_{timeframe}")
        return self.xgb_classifiers[key]

    def _get_or_create_state_machine(self, symbol: str, timeframe: str) -> RegimeStateMachine:
        """Return existing or freshly created state machine for *symbol+timeframe*."""
        key = f"{symbol}_{timeframe}"
        if key not in self.state_machines:
            self.state_machines[key] = RegimeStateMachine(symbol=f"{symbol}_{timeframe}")
        return self.state_machines[key]

    def _extract_atr_ratio(self, features: pd.DataFrame) -> float:
        """Extract the most recent ``atr_ratio`` value from *features*.

        Falls back to ``1.0`` (neutral) if the column is absent or empty.
        """
        if "atr_ratio" in features.columns and not features.empty:
            val = features["atr_ratio"].iloc[-1]
            if pd.notna(val):
                return float(val)
        return 1.0

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        symbol: str,
        data_dict: dict,
        features: pd.DataFrame,
    ) -> RegimeState:
        """Run full regime detection for *symbol* on the current M15 close.

        Parameters
        ----------
        symbol:
            Instrument identifier, e.g. ``"XAUUSD"``.
        data_dict:
            Mapping of timeframe to OHLCV DataFrame, e.g.
            ``{"H4": df_h4, "H1": df_h1, "M15": df_m15}``.
        features:
            Pre-computed feature DataFrame aligned to the M15 timeframe.
            Must include at least the columns expected by HMM and XGBoost.

        Returns
        -------
        RegimeState
            Fully populated snapshot.  If models are not yet trained, a
            default ``RegimeState`` with all ``RANGE`` regimes and
            ``final_sizing_multiplier = 0.0`` is returned.

        Steps
        -----
        1. For each timeframe (H4, H1, M15):
           a. Get / create HMM model for symbol+timeframe.
           b. Get / create XGBoost classifier for symbol+timeframe.
           c. If both trained: run XGB prediction on *features*.
           d. Run state machine update.
           e. Extract regime, confidence, hmm_probs.
        2. Get global risk state from universal model.
        3. Compute MTF alignment.
        4. Compute regime age multiplier (using M15 state machine as primary).
        5. Compute ``final_sizing_multiplier``.
        6. Return populated ``RegimeState``.
        """
        now = datetime.now(tz=timezone.utc)
        atr_ratio = self._extract_atr_ratio(features)

        # Per-timeframe results accumulator
        tf_results: dict[str, dict] = {}

        for tf in _TIMEFRAMES:
            hmm_model = self._get_or_create_hmm(symbol, tf)
            xgb_clf   = self._get_or_create_xgb(symbol, tf)
            sm        = self._get_or_create_state_machine(symbol, tf)

            # Default values in case models are untrained
            raw_regime     = REGIME_RANGE
            confidence_val = 0.5
            hmm_probs: dict = {}

            if hmm_model._is_trained and xgb_clf._is_trained:
                try:
                    # Get HMM probabilities first
                    hmm_result = hmm_model.predict_latest(
                        features.to_numpy(dtype=float),
                        feature_names=list(features.columns),
                    )
                    hmm_probs_arr: np.ndarray = (
                        hmm_model._hmm.predict_proba(
                            hmm_model._scaler.transform(
                                hmm_model._select_hmm_features(
                                    features.to_numpy(dtype=float),
                                    list(features.columns),
                                )
                            )
                        )
                    )
                    hmm_probs = hmm_result.get("probabilities", {})

                    # XGBoost prediction (augmented with HMM posterior)
                    xgb_result = xgb_clf.predict(
                        features,
                        hmm_probs=hmm_probs_arr if xgb_clf._hmm_cols_used else None,
                    )
                    raw_regime     = xgb_result["regime"]
                    confidence_val = xgb_result["confidence"]
                    hmm_probs      = xgb_result.get("probabilities", hmm_probs)

                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "[%s/%s] Prediction failed (%s). Using default RANGE.",
                        symbol, tf, exc,
                    )
                    raw_regime     = REGIME_RANGE
                    confidence_val = 0.5
                    hmm_probs      = {}

                # State machine update
                sm.update(raw_regime, confidence_val, atr_ratio, self._persistence_config)
            else:
                # Models not trained: do not update state machine; leave at default
                self._logger.debug(
                    "[%s/%s] Models not trained — using default RANGE regime.", symbol, tf
                )

            tf_results[tf] = {
                "regime":     sm.current_regime,
                "confidence": confidence_val,
                "hmm_probs":  hmm_probs,
                "sm":         sm,
            }

        # ---- Check all models were trained ----------------------------------
        all_trained = all(
            self._get_or_create_hmm(symbol, tf)._is_trained
            and self._get_or_create_xgb(symbol, tf)._is_trained
            for tf in _TIMEFRAMES
        )
        if not all_trained:
            return RegimeState(
                symbol=symbol,
                timestamp=now,
                final_sizing_multiplier=0.0,
                strategy_module="no_trade",
            )

        # ---- Global risk state -----------------------------------------------
        global_result: dict = {"global_risk_state": "NEUTRAL", "global_multiplier": 0.7}
        try:
            universal_features = self.universal_model.compute_features(data_dict)
            global_result = self.universal_model.predict(universal_features)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "[%s] Universal model prediction failed (%s). Using NEUTRAL.", symbol, exc
            )

        global_risk_state = global_result.get("global_risk_state", "NEUTRAL")
        global_multiplier = float(global_result.get("global_multiplier", 0.7))

        # ---- MTF alignment ---------------------------------------------------
        regimes     = {tf: tf_results[tf]["regime"]     for tf in _TIMEFRAMES}
        confidences = {tf: tf_results[tf]["confidence"] for tf in _TIMEFRAMES}

        alignment = self.mtf_alignment.compute_alignment(regimes, confidences)
        alignment_score      = float(alignment["alignment_score"])
        alignment_multiplier = float(alignment["sizing_multiplier"])
        active_strategy      = str(alignment["active_strategy"])

        # ---- Regime age (use M15 state machine as primary) ------------------
        sm_m15           = tf_results["M15"]["sm"]
        regime_age       = sm_m15.regime_age
        maturity_flag    = sm_m15.get_maturity_flag()
        age_multiplier   = sm_m15.get_age_multiplier(self._age_config)
        regime_confirmed = regime_age >= 3  # confirmed after 3+ candles in regime

        # ---- Final sizing multiplier ----------------------------------------
        final_multiplier = global_multiplier * alignment_multiplier * age_multiplier
        final_multiplier = round(max(0.0, min(1.0, final_multiplier)), 4)

        return RegimeState(
            symbol=symbol,
            timestamp=now,
            # Universal
            global_risk_state=global_risk_state,
            global_multiplier=global_multiplier,
            # H4
            h4_regime=tf_results["H4"]["regime"],
            h4_confidence=tf_results["H4"]["confidence"],
            h4_hmm_probs=tf_results["H4"]["hmm_probs"],
            # H1
            h1_regime=tf_results["H1"]["regime"],
            h1_confidence=tf_results["H1"]["confidence"],
            h1_hmm_probs=tf_results["H1"]["hmm_probs"],
            # M15
            m15_regime=tf_results["M15"]["regime"],
            m15_confidence=tf_results["M15"]["confidence"],
            m15_hmm_probs=tf_results["M15"]["hmm_probs"],
            # Alignment
            alignment_score=alignment_score,
            alignment_sizing_multiplier=alignment_multiplier,
            active_strategy=active_strategy,
            # Regime state
            regime_confirmed=regime_confirmed,
            regime_age_candles=regime_age,
            regime_maturity_flag=maturity_flag,
            regime_age_multiplier=age_multiplier,
            # Final
            final_sizing_multiplier=final_multiplier,
            strategy_module=active_strategy,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all(self, historical_data: dict, symbols: list[str]) -> None:
        """Train HMM and XGBoost models for every symbol and timeframe.

        Parameters
        ----------
        historical_data:
            Nested mapping ``{symbol: {timeframe: features_dataframe}}``,
            where each value is a :class:`pandas.DataFrame` of pre-computed
            features aligned to that timeframe.
        symbols:
            List of instrument identifiers to train.

        For each symbol and timeframe:
        1. Extract features from *historical_data*.
        2. Train HMM on the features to discover regimes.
        3. Generate HMM labels for supervised XGBoost training.
        4. Train XGBoost using HMM labels + features.
        5. Save both models to disk.

        All errors are logged but do not abort training for other symbols.
        """
        for symbol in symbols:
            self._logger.info("[%s] Starting train_all...", symbol)

            symbol_data: dict = historical_data.get(symbol, {})
            if not symbol_data:
                self._logger.warning(
                    "[%s] No historical data provided — skipping.", symbol
                )
                continue

            for tf in _TIMEFRAMES:
                tf_features: Optional[pd.DataFrame] = symbol_data.get(tf)
                if tf_features is None or tf_features.empty:
                    self._logger.warning(
                        "[%s/%s] No feature data — skipping.", symbol, tf
                    )
                    continue

                key = f"{symbol}_{tf}"
                try:
                    # ---- HMM training ----------------------------------------
                    hmm_model = HMMRegimeModel(symbol=key)
                    feat_arr  = tf_features.to_numpy(dtype=float)
                    feat_names = list(tf_features.columns)
                    hmm_model.train(feat_arr, feature_names=feat_names)

                    # ---- XGBoost training ------------------------------------
                    mapped_labels, hmm_probs_arr = hmm_model.predict(
                        feat_arr, feature_names=feat_names
                    )
                    label_series = pd.Series(mapped_labels, index=tf_features.index)

                    xgb_clf = XGBRegimeClassifier(symbol=key)
                    xgb_clf.train(
                        features=tf_features,
                        labels=label_series,
                        hmm_probs=hmm_probs_arr,
                        use_optuna=False,
                    )

                    # ---- Persist to disk ------------------------------------
                    self.hmm_models[key]      = hmm_model
                    self.xgb_classifiers[key] = xgb_clf

                    hmm_path = os.path.join("models", "hmm",     f"{key}.pkl")
                    xgb_path = os.path.join("models", "xgboost", f"{key}.pkl")
                    os.makedirs(os.path.dirname(hmm_path), exist_ok=True)
                    os.makedirs(os.path.dirname(xgb_path), exist_ok=True)
                    hmm_model.save_model(hmm_path)
                    xgb_clf.save_model(xgb_path)

                    self._logger.info("[%s/%s] Training complete.", symbol, tf)

                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "[%s/%s] Training failed: %s", symbol, tf, exc
                    )

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_models(self, symbols: list[str]) -> None:
        """Load saved HMM and XGBoost models from disk for all symbols.

        Parameters
        ----------
        symbols:
            List of instrument identifiers whose models should be loaded.

        Missing files are logged at WARNING level and skipped; they do not
        raise an exception so that a partially-trained system can still run.
        """
        for symbol in symbols:
            for tf in _TIMEFRAMES:
                key      = f"{symbol}_{tf}"
                hmm_path = os.path.join("models", "hmm",     f"{key}.pkl")
                xgb_path = os.path.join("models", "xgboost", f"{key}.pkl")

                # HMM
                try:
                    hmm_model = HMMRegimeModel(symbol=key)
                    hmm_model.load_model(hmm_path)
                    self.hmm_models[key] = hmm_model
                except FileNotFoundError:
                    self._logger.warning(
                        "[%s/%s] HMM model file not found: %s", symbol, tf, hmm_path
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "[%s/%s] Failed to load HMM model: %s", symbol, tf, exc
                    )

                # XGBoost
                try:
                    xgb_clf = XGBRegimeClassifier(symbol=key)
                    xgb_clf.load_model(xgb_path)
                    self.xgb_classifiers[key] = xgb_clf
                except FileNotFoundError:
                    self._logger.warning(
                        "[%s/%s] XGBoost model file not found: %s", symbol, tf, xgb_path
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "[%s/%s] Failed to load XGBoost model: %s", symbol, tf, exc
                    )

    def save_models(self, symbols: list[str]) -> None:
        """Save all trained HMM and XGBoost models to disk.

        Parameters
        ----------
        symbols:
            List of instrument identifiers whose models should be saved.

        Untrained models and write errors are logged and skipped.
        """
        for symbol in symbols:
            for tf in _TIMEFRAMES:
                key      = f"{symbol}_{tf}"
                hmm_path = os.path.join("models", "hmm",     f"{key}.pkl")
                xgb_path = os.path.join("models", "xgboost", f"{key}.pkl")

                # HMM
                hmm_model: Optional[HMMRegimeModel] = self.hmm_models.get(key)
                if hmm_model is not None and hmm_model._is_trained:
                    try:
                        os.makedirs(os.path.dirname(hmm_path), exist_ok=True)
                        hmm_model.save_model(hmm_path)
                    except OSError as exc:
                        self._logger.error(
                            "[%s/%s] Failed to save HMM model: %s", symbol, tf, exc
                        )
                else:
                    self._logger.debug(
                        "[%s/%s] HMM model untrained or absent — skipping save.", symbol, tf
                    )

                # XGBoost
                xgb_clf: Optional[XGBRegimeClassifier] = self.xgb_classifiers.get(key)
                if xgb_clf is not None and xgb_clf._is_trained:
                    try:
                        os.makedirs(os.path.dirname(xgb_path), exist_ok=True)
                        xgb_clf.save_model(xgb_path)
                    except OSError as exc:
                        self._logger.error(
                            "[%s/%s] Failed to save XGBoost model: %s", symbol, tf, exc
                        )
                else:
                    self._logger.debug(
                        "[%s/%s] XGBoost model untrained or absent — skipping save.", symbol, tf
                    )
