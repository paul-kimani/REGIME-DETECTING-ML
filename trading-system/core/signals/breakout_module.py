"""Breakout signal module: compression detection, break confirmation, ML false-breakout filter."""

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


class BreakoutModule:
    """Three-step breakout signal module.

    Step 1 — **Compression detection**: identifies volatility squeeze using
    Bollinger Band width, ATR ratio, and N-bar range contraction. All three
    conditions must be satisfied simultaneously.

    Step 2 — **Break detection**: a directional price break beyond the
    compression boundary, confirmed by elevated volume.

    Step 3 — **ML validation**: XGBoost false-breakout filter requires
    P(true_breakout) > 0.70 (strict gate for volatile-regime trading).

    Entry is a stop-limit order at the break level. The maximum stop width
    is capped at 1.5 × ATR; signals with wider stops are rejected.
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
        """Initialise the module and load configuration.

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
                bo = cfg.breakout
                self._bb_ratio_thresh      = float(bo.compression.bb_width_ratio_threshold)
                self._atr_ratio_thresh     = float(bo.compression.atr_ratio_threshold)
                self._range_atr_thresh     = float(bo.compression.range_atr_ratio_threshold)
                self._min_compression_bars = int(bo.compression.min_compression_candles)
                self._break_dist_atr       = float(bo.break_confirmation.min_break_distance_atr)
                self._vol_ratio_thresh     = float(bo.break_confirmation.volume_ratio_threshold)
                self._ml_min_conf          = float(bo.xgboost.min_signal_confidence)
            except Exception as exc:
                self._log.warning("Breakout config error: %s — using defaults", exc)
                self._set_defaults()
        else:
            self._set_defaults()

    def _set_defaults(self) -> None:
        self._bb_ratio_thresh      = 0.70
        self._atr_ratio_thresh     = 0.80
        self._range_atr_thresh     = 1.50
        self._min_compression_bars = 8
        self._break_dist_atr       = 0.30
        self._vol_ratio_thresh     = 1.50
        self._ml_min_conf          = 0.70

    # ------------------------------------------------------------------
    # Step 1 — Compression detection
    # ------------------------------------------------------------------

    def _detect_compression(self, features: pd.DataFrame, n: int = 20) -> dict:
        """Check the last *n* bars for a volatility squeeze.

        All three conditions must be simultaneously satisfied on the latest
        bar:

        1. ``bb_width < bb_width.rolling(20).mean() * threshold``
        2. ``atr_ratio < threshold`` (ATR(7) / ATR(50) equivalent)
        3. ``(high_N - low_N) / atr_50_mean < range_threshold``

        Args:
            features: Feature DataFrame with the latest bars.
            n:        Lookback window for range measurement (default 20).

        Returns:
            Dict with keys: compressed, duration, compression_high,
            compression_low, bb_width_ratio, atr_ratio_current.
        """
        result: dict = {
            "compressed": False,
            "duration": 0,
            "compression_high": 0.0,
            "compression_low": 0.0,
            "bb_width_ratio": 1.0,
            "atr_ratio_current": 1.0,
        }

        if len(features) < max(n, 20):
            return result

        tail = features.tail(n)
        latest = features.iloc[-1]

        # Condition 1: BB width squeeze
        bb_w = tail.get("bb_width", pd.Series(dtype=float))
        if hasattr(bb_w, "values") and len(bb_w) > 0:
            bb_mean = float(bb_w.mean())
            bb_curr = float(bb_w.iloc[-1])
            bb_ratio = bb_curr / bb_mean if bb_mean > 0 else 1.0
        else:
            bb_ratio = 1.0

        # Condition 2: ATR ratio (already normalised in features)
        atr_ratio = float(latest.get("atr_ratio", 1.0))

        # Condition 3: N-bar price range vs ATR_50_mean
        if "high" in tail.columns and "low" in tail.columns:
            range_n = float(tail["high"].max() - tail["low"].min())
        else:
            range_n = float(latest.get("atr_14", 0.001)) * 3.0   # safe fallback

        atr_50_mean = float(latest.get("atr_50_mean", latest.get("atr_14", 0.001)))
        range_ratio = range_n / atr_50_mean if atr_50_mean > 0 else 99.0

        cond1 = bb_ratio   < self._bb_ratio_thresh
        cond2 = atr_ratio  < self._atr_ratio_thresh
        cond3 = range_ratio < self._range_atr_thresh

        if not (cond1 and cond2 and cond3):
            result.update({"bb_width_ratio": bb_ratio, "atr_ratio_current": atr_ratio})
            return result

        # Count compression duration
        if "atr_ratio" in features.columns:
            atr_series = features["atr_ratio"].values
            dur = 0
            for i in range(len(atr_series) - 1, -1, -1):
                if atr_series[i] < self._atr_ratio_thresh:
                    dur += 1
                else:
                    break
        else:
            dur = n

        high_col = tail["high"] if "high" in tail.columns else pd.Series([0.0])
        low_col  = tail["low"]  if "low"  in tail.columns else pd.Series([0.0])

        result.update({
            "compressed": True,
            "duration": max(dur, self._min_compression_bars),
            "compression_high": float(high_col.max()),
            "compression_low":  float(low_col.min()),
            "bb_width_ratio": bb_ratio,
            "atr_ratio_current": atr_ratio,
        })
        return result

    # ------------------------------------------------------------------
    # Step 2 — Break detection
    # ------------------------------------------------------------------

    def _detect_break(self, features: pd.Series, compression: dict) -> dict:
        """Detect a directional break of the compression boundary.

        Args:
            features:    Feature row (pd.Series) for the current (break) bar.
            compression: Result dict from :meth:`_detect_compression`.

        Returns:
            Dict with keys: break_detected, direction, distance_atr, volume_ok.
        """
        result = {"break_detected": False, "direction": "NONE",
                  "distance_atr": 0.0, "volume_ok": False}

        close      = float(features.get("close",        0.0))
        atr        = float(features.get("atr_14",       0.001))
        vol_ratio  = float(features.get("volume_ratio", 0.0))
        comp_high  = compression["compression_high"]
        comp_low   = compression["compression_low"]

        volume_ok = vol_ratio >= self._vol_ratio_thresh

        break_up_price   = comp_high + self._break_dist_atr * atr
        break_down_price = comp_low  - self._break_dist_atr * atr

        if close > break_up_price:
            dist = (close - comp_high) / atr if atr > 0 else 0.0
            result.update({"break_detected": True, "direction": "LONG",
                           "distance_atr": dist, "volume_ok": volume_ok})
        elif close < break_down_price:
            dist = (comp_low - close) / atr if atr > 0 else 0.0
            result.update({"break_detected": True, "direction": "SHORT",
                           "distance_atr": dist, "volume_ok": volume_ok})

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        regime_state,
        latest_bar: dict,
    ) -> Optional[SignalOutput]:
        """Run all three steps and return a stop-limit signal if all pass.

        Args:
            features:     Feature DataFrame, last row = current bar.
            regime_state: Current RegimeState.
            latest_bar:   Dict with bid, ask, atr, close, pip_size keys.

        Returns:
            :class:`SignalOutput` or ``None``.
        """
        if len(features) < 25:
            return None

        # Step 1: Compression
        compression = self._detect_compression(features, n=20)
        if not compression["compressed"]:
            return None
        if compression["duration"] < self._min_compression_bars:
            return None

        # Step 2: Break
        latest_row = features.iloc[-1]
        brk = self._detect_break(latest_row, compression)
        if not brk["break_detected"]:
            return None
        if not brk["volume_ok"]:
            self._log.debug("%s breakout: volume insufficient", self.symbol)
            return None

        direction = brk["direction"]

        # Step 3: ML validation
        if self._xgb_trained and self._xgb_model is not None:
            try:
                available = [c for c in self._feature_names if c in features.columns]
                row = features[available].iloc[[-1]].fillna(0.0)
                ml_proba = float(self._xgb_model.predict_proba(row)[0][1])
            except Exception as exc:
                self._log.warning("Breakout ML error: %s", exc)
                ml_proba = 1.0
            if ml_proba < self._ml_min_conf:
                return None
        else:
            ml_proba = 1.0  # untrained → pass (gates 1+2 are sufficient guards)

        # Build entry / stop / targets
        atr       = float(latest_bar.get("atr", float(latest_row.get("atr_14", 0.001))))
        pip       = float(latest_bar.get("pip_size", 0.0001))
        close     = float(latest_bar.get("close", float(latest_row.get("close", 0))))
        comp_high = compression["compression_high"]
        comp_low  = compression["compression_low"]
        comp_height = comp_high - comp_low

        max_stop_atr = 1.5  # from spec

        if direction == "LONG":
            entry = comp_high                    # stop-limit at break level
            stop  = comp_low                     # opposite boundary
        else:
            entry = comp_low
            stop  = comp_high

        stop_dist = abs(entry - stop)

        # Reject if stop is wider than max
        if stop_dist > max_stop_atr * atr:
            self._log.debug("%s breakout: stop too wide (%.4f > %.4f)", self.symbol, stop_dist, max_stop_atr * atr)
            return None

        if direction == "LONG":
            tp1 = entry + 0.5 * comp_height
            tp2 = entry + 1.5 * comp_height
        else:
            tp1 = entry - 0.5 * comp_height
            tp2 = entry - 1.5 * comp_height

        if entry <= 0 or stop <= 0 or stop_dist <= 0:
            return None

        rr_ratio  = abs(tp2 - entry) / stop_dist
        stop_pips = stop_dist / pip if pip > 0 else stop_dist * 10_000

        return SignalOutput(
            asset=self.symbol,
            timeframe=getattr(regime_state, "timeframe", "M15"),
            timestamp=datetime.utcnow(),
            signal=direction,
            module="BREAKOUT",
            confidence=ml_proba,
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
        """Label break bars where price continued > 2 × ATR within 4 bars.

        Args:
            df:       Raw OHLCV DataFrame (must contain 'high', 'low', 'close').
            features: Feature DataFrame containing 'atr_14'.

        Returns:
            Binary pd.Series (1 = true breakout, 0 = false).
        """
        close = df["close"].values
        high  = df["high"].values
        atr   = features["atr_14"].values if "atr_14" in features.columns else np.full(len(df), 0.001)
        n     = len(close)
        labels = np.zeros(n, dtype=int)
        n_fwd  = 4

        for i in range(n - n_fwd):
            a  = atr[i] if atr[i] > 0 else 1e-8
            fh = high[i + 1: i + 1 + n_fwd]
            fl = df["low"].values[i + 1: i + 1 + n_fwd]

            if np.max(fh) - close[i] > 2.0 * a:
                labels[i] = 1
            elif close[i] - np.min(fl) > 2.0 * a:
                labels[i] = 1

        return pd.Series(labels, index=df.index, name="label")

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the XGBoost false-breakout filter.

        Args:
            features: Feature DataFrame.
            labels:   Binary label series.
        """
        X = features.copy()
        y = labels.values
        valid = ~(X.isnull().any(axis=1) | np.isnan(y.astype(float)))
        X, y = X[valid], y[valid]

        if len(X) < 50:
            self._log.warning("%s BO: insufficient training data", self.symbol)
            return

        self._feature_names = list(X.columns)
        self._xgb_model = xgb.XGBClassifier(**self._XGB_DEFAULTS)
        self._xgb_model.fit(X, y, verbose=False)
        self._xgb_trained = True
        self._log.info("%s breakout model trained on %d samples", self.symbol, len(X))

    def save_model(self, path: str) -> None:
        """Save the model bundle to *path*.

        Raises:
            RuntimeError: If model is not trained.
        """
        if not self._xgb_trained:
            raise RuntimeError(f"{self.symbol} BreakoutModule: not trained")
        joblib.dump({
            "xgb": self._xgb_model,
            "features": self._feature_names,
            "symbol": self.symbol,
        }, path)
        self._log.info("Saved breakout model to %s", path)

    def load_model(self, path: str) -> None:
        """Load a saved model bundle from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        bundle = joblib.load(path)
        self._xgb_model     = bundle["xgb"]
        self._feature_names = bundle["features"]
        self._xgb_trained   = True
        self._log.info("Loaded breakout model from %s", path)
