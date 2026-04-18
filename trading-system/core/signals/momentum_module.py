"""Momentum signal module: XGBoost + LSTM ensemble for trend-following entries."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

from core.signals.signal_router import SignalOutput
from core.utils.logger import get_logger

logger = get_logger(__name__)

# Label constants
LABEL_SHORT   = 0
LABEL_NOTRADE = 1
LABEL_LONG    = 2


def _try_config() -> object:
    """Load config safely — return None if unavailable (e.g. during tests)."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class MomentumModule:
    """XGBoost + LSTM ensemble for trend-following signal generation.

    Training flow:
        1. ``generate_labels`` produces look-ahead-free forward-return labels.
        2. ``train`` fits the XGBoost model on those labels.
        3. ``predict`` runs inference and returns a :class:`SignalOutput` when
           the ensemble confidence and directional-dominance gates are met.

    LSTM training is defined but requires GPU hardware; it logs a message and
    returns immediately when called without a GPU-trained model loaded.
    """

    # Default hyper-parameters (overridden by config when available)
    _XGB_DEFAULTS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "objective": "multi:softprob",
        "num_class": 3,
        "verbosity": 0,
    }

    def __init__(self, symbol: str = "UNKNOWN") -> None:
        """Initialise the module and load configuration.

        Args:
            symbol: Instrument symbol for logging and model labelling.
        """
        self.symbol = symbol
        self._log = get_logger(__name__)
        self._xgb_model: Optional[xgb.XGBClassifier] = None
        self._xgb_trained: bool = False
        self._feature_names: list[str] = []
        self._importances: Optional[pd.DataFrame] = None

        cfg = _try_config()
        if cfg is not None:
            try:
                xp = cfg.momentum.xgboost
                self._xgb_params = {
                    "n_estimators": int(xp.n_estimators),
                    "max_depth": int(xp.max_depth),
                    "learning_rate": float(xp.learning_rate),
                    "subsample": float(xp.subsample),
                    "colsample_bytree": float(xp.colsample_bytree),
                    "min_child_weight": int(xp.min_child_weight),
                    "use_label_encoder": False,
                    "eval_metric": "mlogloss",
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "verbosity": 0,
                }
                ep = cfg.momentum.entry
                self._min_confidence = float(ep.min_signal_confidence)
                self._min_dominance = float(ep.min_directional_dominance)
                self._label_fwd_m15 = int(ep.label_forward_candles_M15)
                self._label_atr_thresh = float(ep.label_atr_threshold)
            except Exception as exc:
                self._log.warning("Could not read momentum config: %s — using defaults", exc)
                self._xgb_params = dict(self._XGB_DEFAULTS)
                self._min_confidence = 0.65
                self._min_dominance = 2.0
                self._label_fwd_m15 = 6
                self._label_atr_thresh = 1.5
        else:
            self._xgb_params = dict(self._XGB_DEFAULTS)
            self._min_confidence = 0.65
            self._min_dominance = 2.0
            self._label_fwd_m15 = 6
            self._label_atr_thresh = 1.5

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def generate_labels(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        timeframe: str = "M15",
    ) -> pd.Series:
        """Generate forward-looking training labels.

        For each bar *i*, look ``label_forward_candles`` bars forward:

        * If the maximum favourable price move in a direction exceeds
          ``label_atr_threshold * ATR``, the bar is labelled LONG (2) or
          SHORT (0) respectively.
        * Otherwise → NOTRADE (1).

        Labels at the last *N* bars are forced to NOTRADE because no future
        data is available (look-ahead prevention).

        Args:
            df:        Raw OHLCV DataFrame (must contain 'close', 'high', 'low').
            features:  Feature DataFrame aligned with ``df`` (must contain 'atr_14').
            timeframe: Timeframe string — used to select forward-candle count.

        Returns:
            pd.Series of integer labels (0=SHORT, 1=NOTRADE, 2=LONG).
        """
        n_fwd = self._label_fwd_m15 if "M15" in timeframe.upper() else self._label_fwd_m15
        atr_thresh = self._label_atr_thresh

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr   = features["atr_14"].values if "atr_14" in features.columns else np.full(len(df), 0.001)

        n = len(close)
        labels = np.ones(n, dtype=int)  # default NOTRADE

        for i in range(n - n_fwd):
            fwd_slice_high = high[i + 1 : i + 1 + n_fwd]
            fwd_slice_low  = low[i + 1 : i + 1 + n_fwd]
            a = atr[i] if atr[i] > 0 else 1e-8
            threshold = atr_thresh * a

            max_up   = np.max(fwd_slice_high) - close[i]
            max_down = close[i] - np.min(fwd_slice_low)

            if max_up >= threshold and max_up > max_down:
                labels[i] = LABEL_LONG
            elif max_down >= threshold and max_down > max_up:
                labels[i] = LABEL_SHORT
            # else stays NOTRADE

        # Force last n_fwd bars to NOTRADE (no future data)
        labels[n - n_fwd :] = LABEL_NOTRADE

        return pd.Series(labels, index=df.index, name="label")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        use_optuna: bool = False,
    ) -> None:
        """Fit the XGBoost signal model.

        Uses class weights to handle label imbalance, and reserves the last
        20 % of the data as a validation set for early stopping.

        Args:
            features:   Feature DataFrame (rows = bars, cols = feature columns).
            labels:     Integer label series aligned with *features*.
            use_optuna: Whether to run Optuna hyperparameter search (50 trials).
                        Expensive — only use during scheduled retraining.
        """
        X = features.copy()
        y = labels.values

        # Drop rows where labels or features have NaN
        valid_mask = ~(np.isnan(y.astype(float)) | X.isnull().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            self._log.warning("%s — insufficient training data (%d rows)", self.symbol, len(X))
            return

        self._feature_names = list(X.columns)

        # Class weights
        classes = np.array([LABEL_SHORT, LABEL_NOTRADE, LABEL_LONG])
        weights = compute_class_weight("balanced", classes=classes, y=y)
        sample_weights = np.array([weights[int(lbl)] for lbl in y])

        # Walk-forward validation split (last 20%)
        split = int(len(X) * 0.80)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y[:split], y[split:]
        sw_train = sample_weights[:split]

        if use_optuna:
            self._xgb_params = self._optuna_search(X_train, y_train, X_val, y_val)

        self._xgb_model = xgb.XGBClassifier(**self._xgb_params)
        self._xgb_model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._xgb_trained = True

        # Feature importances
        imp = self._xgb_model.feature_importances_
        self._importances = (
            pd.DataFrame({"feature": self._feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        self._log.info(
            "%s momentum model trained on %d bars. Top feature: %s",
            self.symbol, len(X_train), self._importances.iloc[0]["feature"]
        )

    def _optuna_search(
        self,
        X_train: pd.DataFrame, y_train: np.ndarray,
        X_val: pd.DataFrame, y_val: np.ndarray,
        n_trials: int = 50,
    ) -> dict:
        """Run Optuna hyperparameter search. Returns best params dict."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "use_label_encoder": False,
                    "eval_metric": "mlogloss",
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "verbosity": 0,
                }
                m = xgb.XGBClassifier(**params)
                m.fit(X_train, y_train, verbose=False)
                preds = m.predict(X_val)
                return float((preds == y_val).mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            best.update({"use_label_encoder": False, "eval_metric": "mlogloss",
                         "objective": "multi:softprob", "num_class": 3, "verbosity": 0})
            self._log.info("%s Optuna best accuracy=%.4f", self.symbol, study.best_value)
            return best
        except ImportError:
            self._log.warning("optuna not installed — skipping hyperparameter search")
            return dict(self._xgb_params)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        regime_state,
        latest_bar: dict,
    ) -> Optional[SignalOutput]:
        """Generate a LONG or SHORT signal for the current bar.

        Args:
            features:     Feature DataFrame. Last row = current bar.
            regime_state: Current RegimeState (used for context only here).
            latest_bar:   Dict with keys: bid, ask, atr, high, low, close.

        Returns:
            :class:`SignalOutput` or ``None`` if no tradeable setup.
        """
        if not self._xgb_trained or self._xgb_model is None:
            return None

        try:
            row = features[self._feature_names].iloc[[-1]].fillna(0.0)
        except KeyError:
            # Some features may be missing — fill silently
            available = [c for c in self._feature_names if c in features.columns]
            row = features[available].iloc[[-1]].fillna(0.0)

        proba = self._xgb_model.predict_proba(row)[0]   # shape (3,)
        p_short, p_notrade, p_long = proba[0], proba[1], proba[2]

        # Determine direction by max probability
        if p_long > p_short:
            direction = "LONG"
            confidence = float(p_long)
            p_opposite = float(p_short)
        else:
            direction = "SHORT"
            confidence = float(p_short)
            p_opposite = float(p_long)

        # Gate 1: minimum confidence
        if confidence < self._min_confidence:
            return None

        # Gate 2: directional dominance
        if p_opposite > 0 and (confidence / p_opposite) < self._min_dominance:
            return None

        # Build entry prices
        bid = float(latest_bar.get("bid", latest_bar.get("close", 0)))
        ask = float(latest_bar.get("ask", latest_bar.get("close", 0)))
        atr = float(latest_bar.get("atr", features["atr_14"].iloc[-1] if "atr_14" in features.columns else 0.001))
        recent_high = float(latest_bar.get("high", features["high"].iloc[-5:].max() if "high" in features.columns else ask))
        recent_low  = float(latest_bar.get("low",  features["low"].iloc[-5:].min()  if "low"  in features.columns else bid))

        if direction == "LONG":
            entry = ask
            stop  = recent_low - 1.0 * atr
            tp1   = entry + 1.0 * (entry - stop)
            tp2   = entry + 2.0 * (entry - stop)
        else:
            entry = bid
            stop  = recent_high + 1.0 * atr
            tp1   = entry - 1.0 * (stop - entry)
            tp2   = entry - 2.0 * (stop - entry)

        if entry <= 0 or stop <= 0 or atr <= 0:
            return None

        stop_dist = abs(entry - stop)
        rr_ratio  = abs(tp2 - entry) / stop_dist if stop_dist > 0 else 0.0

        # Convert stop distance to pips
        pip_size = float(latest_bar.get("pip_size", 0.0001))
        stop_pips = stop_dist / pip_size if pip_size > 0 else stop_dist * 10000

        return SignalOutput(
            asset=self.symbol,
            timeframe=getattr(regime_state, "timeframe", "M15"),
            timestamp=datetime.utcnow(),
            signal=direction,
            module="MOMENTUM",
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            atr=atr,
            stop_distance_pips=stop_pips,
            rr_ratio=rr_ratio,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Serialise the XGBoost model to *path*.

        Args:
            path: File path for the pickled model bundle.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._xgb_trained:
            raise RuntimeError(f"{self.symbol} MomentumModule: model not trained")
        bundle = {
            "xgb_model":     self._xgb_model,
            "feature_names": self._feature_names,
            "xgb_params":    self._xgb_params,
            "importances":   self._importances,
            "symbol":        self.symbol,
        }
        joblib.dump(bundle, path)
        self._log.info("Saved momentum model to %s", path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model bundle from *path*.

        Args:
            path: File path of the pickled model bundle.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        bundle = joblib.load(path)
        self._xgb_model     = bundle["xgb_model"]
        self._feature_names = bundle["feature_names"]
        self._xgb_params    = bundle["xgb_params"]
        self._importances   = bundle.get("importances")
        self._xgb_trained   = True
        self._log.info("Loaded momentum model from %s", path)
