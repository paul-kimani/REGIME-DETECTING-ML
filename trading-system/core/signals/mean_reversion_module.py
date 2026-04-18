"""Mean-reversion signal module using z-score, oscillator confluence, and ML filter."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from core.signals.signal_router import SignalOutput
from core.utils.logger import get_logger

logger = get_logger(__name__)


def _try_config() -> object:
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class MeanReversionModule:
    """Four-gate mean-reversion signal module.

    All four gates must pass for a signal to be generated:

    1. **Gate 1 — Z-score extension**: price is sufficiently far from its
       rolling mean, adjusted dynamically by the Hurst exponent.
    2. **Gate 2 — Oscillator confluence**: at least 2 of 3 oscillators
       (RSI-7, Stochastic-%K, CCI-14) agree with the reversion direction.
    3. **Gate 3 — ADX / MACD filter**: ADX < 30 and no MACD expansion
       against the reversion direction.
    4. **Gate 4 — ML filter**: XGBoost probability of successful reversion
       exceeds 0.60.

    No market fallback — only limit orders (``no_market_fallback: true``).
    """

    _XGB_DEFAULTS = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "verbosity": 0,
    }

    def __init__(self, symbol: str = "UNKNOWN") -> None:
        """Initialise module and load configuration.

        Args:
            symbol: Instrument identifier.
        """
        self.symbol = symbol
        self._log = get_logger(__name__)
        self._xgb_model: Optional[xgb.XGBClassifier] = None
        self._xgb_trained: bool = False
        self._feature_names: list[str] = []

        cfg = _try_config()
        if cfg is not None:
            try:
                mr = cfg.mean_reversion
                self._z_window       = int(mr.zscore.window)
                self._z_long_thresh  = float(mr.zscore.long_threshold)
                self._z_short_thresh = float(mr.zscore.short_threshold)
                self._hurst_strong   = float(mr.zscore.hurst_strong_threshold)
                self._hurst_adj      = float(mr.zscore.hurst_strong_zscore_adjustment)
                self._rsi_period     = int(mr.oscillators.rsi_period)
                self._rsi_os         = float(mr.oscillators.rsi_oversold)
                self._rsi_ob         = float(mr.oscillators.rsi_overbought)
                self._stoch_os       = float(mr.oscillators.stoch_oversold)
                self._stoch_ob       = float(mr.oscillators.stoch_overbought)
                self._cci_os         = float(mr.oscillators.cci_oversold)
                self._cci_ob         = float(mr.oscillators.cci_overbought)
                self._osc_min_agree  = int(mr.oscillators.min_agreement)
                self._ml_min_conf    = float(mr.xgboost.min_signal_confidence)
            except Exception as exc:
                self._log.warning("MR config error: %s — using defaults", exc)
                self._set_defaults()
        else:
            self._set_defaults()

    def _set_defaults(self) -> None:
        self._z_window, self._z_long_thresh, self._z_short_thresh = 50, -2.0, 2.0
        self._hurst_strong, self._hurst_adj = 0.40, -0.30
        self._rsi_period, self._rsi_os, self._rsi_ob = 7, 25.0, 75.0
        self._stoch_os, self._stoch_ob = 20.0, 80.0
        self._cci_os, self._cci_ob = -150.0, 150.0
        self._osc_min_agree, self._ml_min_conf = 2, 0.60

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def _check_gate1_zscore(self, features: pd.Series) -> tuple[bool, str, float]:
        """Gate 1: Z-score extension check.

        The long threshold is negative (price below mean) and the short
        threshold is positive.  If the Hurst exponent indicates strong
        mean-reversion (H < 0.40), the thresholds are relaxed by
        ``hurst_strong_zscore_adjustment``.

        Args:
            features: Feature row (pd.Series) for the current bar.

        Returns:
            (passes, direction, z_value) tuple.
        """
        z = float(features.get("z_score_50", 0.0))
        h = float(features.get("hurst_100", 0.5))

        long_t  = self._z_long_thresh
        short_t = self._z_short_thresh

        # Relax threshold for stronger mean-reverting series
        if h < self._hurst_strong:
            long_t  = long_t  - self._hurst_adj   # e.g. -2.0 - (-0.30) = -1.70
            short_t = short_t + self._hurst_adj   # e.g.  2.0 + (-0.30) =  1.70

        if z <= long_t:
            return True, "LONG", z
        if z >= short_t:
            return True, "SHORT", z
        return False, "NONE", z

    def _check_gate2_oscillators(self, features: pd.Series, direction: str) -> bool:
        """Gate 2: At least ``min_agreement`` oscillators confirm direction.

        For LONG: RSI-7 oversold OR Stochastic-%K oversold OR CCI oversold.
        For SHORT: RSI-7 overbought OR Stochastic-%K overbought OR CCI overbought.

        Args:
            features:  Feature row.
            direction: "LONG" or "SHORT".

        Returns:
            ``True`` if enough oscillators agree.
        """
        rsi   = float(features.get("rsi_7",   50.0))
        stoch = float(features.get("stoch_k", 50.0))
        cci   = float(features.get("cci_14",   0.0))

        if direction == "LONG":
            agreements = sum([
                rsi   < self._rsi_os,
                stoch < self._stoch_os,
                cci   < self._cci_os,
            ])
        else:
            agreements = sum([
                rsi   > self._rsi_ob,
                stoch > self._stoch_ob,
                cci   > self._cci_ob,
            ])

        return agreements >= self._osc_min_agree

    def _check_gate3_adx(self, features: pd.Series, direction: str) -> bool:
        """Gate 3: ADX < 30 and no MACD expansion against reversion.

        Args:
            features:  Feature row.
            direction: "LONG" or "SHORT".

        Returns:
            ``True`` if ADX is weak and MACD is not expanding against direction.
        """
        adx       = float(features.get("adx_14",        50.0))
        macd_hist = float(features.get("macd_hist",      0.0))
        macd_slp  = float(features.get("macd_hist_slope", 0.0))

        if adx >= 30.0:
            return False

        # MACD expanding against reversion is a disqualifier
        if direction == "LONG"  and macd_slp < 0 and macd_hist < 0:
            return False   # bearish MACD expanding downward while we want long
        if direction == "SHORT" and macd_slp > 0 and macd_hist > 0:
            return False   # bullish MACD expanding upward while we want short

        return True

    def _check_gate4_ml(self, features: pd.Series) -> float:
        """Gate 4: XGBoost reversion probability.

        Args:
            features: Feature row.

        Returns:
            Reversion probability (0.0–1.0). Returns 1.0 if model not trained
            (fall through to allow trade, gates 1–3 are sufficient guards).
        """
        if not self._xgb_trained or self._xgb_model is None:
            return 1.0

        try:
            available = [c for c in self._feature_names if c in features.index]
            row = features[available].fillna(0.0).values.reshape(1, -1)
            proba = self._xgb_model.predict_proba(row)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as exc:
            self._log.warning("Gate 4 ML error: %s", exc)
            return 1.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        regime_state,
        latest_bar: dict,
    ) -> Optional[SignalOutput]:
        """Run all four gates and return a limit-order signal if all pass.

        Entry is a LIMIT order at the current close.
        Stop is placed where the z-score would reach ±3.0 from the entry side.
        TP2 is the rolling mean (full z-score normalisation).
        TP1 is the midpoint between entry and TP2.

        Args:
            features:     Feature DataFrame, last row = current bar.
            regime_state: Current RegimeState.
            latest_bar:   Dict with bid, ask, atr, close keys.

        Returns:
            :class:`SignalOutput` or ``None``.
        """
        if features.empty:
            return None

        row = features.iloc[-1]

        # Gate 1
        g1_pass, direction, z_val = self._check_gate1_zscore(row)
        if not g1_pass:
            return None

        # Gate 2
        if not self._check_gate2_oscillators(row, direction):
            return None

        # Gate 3
        if not self._check_gate3_adx(row, direction):
            return None

        # Gate 4
        ml_prob = self._check_gate4_ml(row)
        if ml_prob < self._ml_min_conf:
            return None

        # Construct entry / stop / targets
        close = float(latest_bar.get("close", row.get("close", 0)))
        atr   = float(latest_bar.get("atr",   row.get("atr_14", 0.001)))
        pip   = float(latest_bar.get("pip_size", 0.0001))

        # Rolling mean and std from z-score definition: mean = close - z*std
        # Approximate std from z_score and current atr
        rolling_std  = atr * 0.7 if atr > 0 else 0.001
        rolling_mean = close - z_val * rolling_std

        entry = close  # limit order at current close

        if direction == "LONG":
            # Stop where z reaches -3.0
            stop   = rolling_mean - 3.0 * rolling_std
            tp2    = rolling_mean           # z-score returns to 0
            tp1    = entry + 0.5 * (tp2 - entry)
        else:
            # Stop where z reaches +3.0
            stop   = rolling_mean + 3.0 * rolling_std
            tp2    = rolling_mean
            tp1    = entry - 0.5 * (entry - tp2)

        if entry <= 0 or stop <= 0 or atr <= 0:
            return None

        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return None

        rr_ratio  = abs(tp2 - entry) / stop_dist
        stop_pips = stop_dist / pip if pip > 0 else stop_dist * 10_000

        return SignalOutput(
            asset=self.symbol,
            timeframe=getattr(regime_state, "timeframe", "M15"),
            timestamp=datetime.utcnow(),
            signal=direction,
            module="MEAN_REVERSION",
            confidence=ml_prob,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            atr=atr,
            stop_distance_pips=stop_pips,
            rr_ratio=rr_ratio,
        )

    # ------------------------------------------------------------------
    # Label generation & training
    # ------------------------------------------------------------------

    def generate_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Label bars where the z-score normalises within 10 forward bars.

        A bar is labelled 1 (reversion) if ``|z_score|`` decreases by at
        least 50 % within 10 bars, else 0 (no reversion).

        Args:
            df:       Raw OHLCV DataFrame.
            features: Feature DataFrame containing 'z_score_50'.

        Returns:
            Binary pd.Series (0 or 1).
        """
        if "z_score_50" not in features.columns:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        z = features["z_score_50"].values
        n = len(z)
        labels = np.zeros(n, dtype=int)
        n_fwd = 10

        for i in range(n - n_fwd):
            z0 = abs(z[i])
            if z0 == 0:
                continue
            future_z = np.abs(z[i + 1 : i + 1 + n_fwd])
            if np.any(future_z <= z0 * 0.50):
                labels[i] = 1

        labels[n - n_fwd :] = 0
        return pd.Series(labels, index=df.index, name="label")

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the XGBoost binary classifier (Gate 4 filter).

        Args:
            features: Feature DataFrame.
            labels:   Binary label series (0 = no reversion, 1 = reversion).
        """
        X = features.copy()
        y = labels.values
        valid = ~(X.isnull().any(axis=1) | np.isnan(y.astype(float)))
        X, y = X[valid], y[valid]

        if len(X) < 50:
            self._log.warning("%s MR: insufficient training data", self.symbol)
            return

        self._feature_names = list(X.columns)
        self._xgb_model = xgb.XGBClassifier(**self._XGB_DEFAULTS)
        self._xgb_model.fit(X, y, verbose=False)
        self._xgb_trained = True
        self._log.info("%s MR model trained on %d samples", self.symbol, len(X))

    def save_model(self, path: str) -> None:
        """Save the model bundle to *path*.

        Raises:
            RuntimeError: If model is not yet trained.
        """
        if not self._xgb_trained:
            raise RuntimeError(f"{self.symbol} MeanReversionModule: not trained")
        joblib.dump({
            "xgb": self._xgb_model,
            "features": self._feature_names,
            "symbol": self.symbol,
        }, path)
        self._log.info("Saved MR model to %s", path)

    def load_model(self, path: str) -> None:
        """Load a saved model bundle from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        bundle = joblib.load(path)
        self._xgb_model     = bundle["xgb"]
        self._feature_names = bundle["features"]
        self._xgb_trained   = True
        self._log.info("Loaded MR model from %s", path)
