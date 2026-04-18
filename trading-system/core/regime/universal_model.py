"""Universal macro risk-state model using cross-asset features."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from core.utils.helpers import atr, rolling_corr
from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Risk-state constants
# ---------------------------------------------------------------------------
RISK_ON  = "RISK_ON"
NEUTRAL  = "NEUTRAL"
RISK_OFF = "RISK_OFF"
CRISIS   = "CRISIS"

_RISK_STATES: list[str] = [RISK_ON, NEUTRAL, RISK_OFF, CRISIS]

# ---------------------------------------------------------------------------
# Hardcoded defaults (used when config is unavailable)
# ---------------------------------------------------------------------------
_DEFAULT_MULTIPLIERS: dict[str, float] = {
    RISK_ON:  1.0,
    NEUTRAL:  0.7,
    RISK_OFF: 0.3,
    CRISIS:   0.0,
}

_DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def _load_multipliers() -> dict[str, float]:
    """Return risk-state position multipliers from config if available.

    Falls back to :data:`_DEFAULT_MULTIPLIERS` if the config system is
    unavailable or the relevant key is missing.

    Returns
    -------
    dict[str, float]
        Keys: ``RISK_ON``, ``NEUTRAL``, ``RISK_OFF``, ``CRISIS``.
    """
    try:
        from core.utils.config import get_config  # local import to avoid import errors

        cfg = get_config()
        um = cfg.universal_model
        m = um.multipliers
        return {
            RISK_ON:  float(m.risk_on),
            NEUTRAL:  float(m.neutral),
            RISK_OFF: float(m.risk_off),
            CRISIS:   float(m.crisis),
        }
    except Exception:  # noqa: BLE001 – broad catch intentional for config fallback
        return dict(_DEFAULT_MULTIPLIERS)


class UniversalModel:
    """Global macro risk-state model using cross-asset features.

    Detects whether markets are in a risk-on, neutral, risk-off, or crisis
    state by computing a small set of cross-asset indicators (DXY momentum,
    gold/silver correlation, volatility proxy) and either applying a trained
    XGBoost classifier or falling back to a simple rule-based heuristic.

    The four possible states are:

    * ``RISK_ON``  — equities bid, gold/USD negative correlation, low vol.
    * ``NEUTRAL``  — no strong directional signal.
    * ``RISK_OFF`` — elevated vol, correlated selling across assets.
    * ``CRISIS``   — extreme vol, all-asset liquidation.

    Parameters
    ----------
    symbol:
        Optional instrument context used only in log messages.
    """

    def __init__(self, symbol: str = "UNIVERSAL") -> None:
        self.symbol = symbol
        self._logger = get_logger(__name__)
        self._model: Optional[xgb.XGBClassifier] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names: Optional[list[str]] = None
        self._is_trained: bool = False
        self._multipliers: dict[str, float] = _load_multipliers()
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_returns(df: pd.DataFrame) -> pd.Series:
        """Compute percentage returns from the ``close`` column of *df*.

        Parameters
        ----------
        df:
            DataFrame that must contain a ``close`` column.

        Returns
        -------
        pd.Series
            Percentage-change returns (first value is NaN).
        """
        return df["close"].pct_change()

    @staticmethod
    def _atr_ratio(df: pd.DataFrame) -> pd.Series:
        """Compute the ATR-14 / ATR-50-mean ratio for *df*.

        Uses ``high``, ``low``, and ``close`` columns if all three are present;
        otherwise approximates ATR from ``close`` only.

        Parameters
        ----------
        df:
            OHLC DataFrame.

        Returns
        -------
        pd.Series
            ATR ratio (current ATR normalised by a longer-term mean ATR).
        """
        if {"high", "low", "close"}.issubset(df.columns):
            atr14 = atr(df["high"], df["low"], df["close"], period=14)
            atr50_mean = atr14.rolling(50).mean()
        else:
            # Approximate TR from close-to-close
            close = df["close"]
            tr = close.diff().abs()
            atr14 = tr.ewm(com=13, min_periods=14, adjust=False).mean()
            atr50_mean = atr14.rolling(50).mean()

        ratio = atr14 / atr50_mean.replace(0, np.nan)
        return ratio.fillna(1.0)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_features(self, asset_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute universal risk-model features from cross-asset price data.

        Parameters
        ----------
        asset_data:
            Mapping of instrument ticker (e.g. ``"XAUUSD"``, ``"DXY"``) to a
            DataFrame that contains at least a ``close`` column and optionally
            ``high`` and ``low`` columns.  All DataFrames should share the
            same time index for the rolling correlations to be meaningful.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame whose columns are:

            * ``dxy_momentum``         — DXY 10-bar percentage change.
            * ``gold_dxy_corr``        — 20-bar rolling correlation of gold
                                         and DXY returns.
            * ``gold_silver_corr``     — 20-bar rolling correlation of gold
                                         and silver returns.
            * ``gold_silver_ratio``    — XAUUSD / XAGUSD close ratio.
            * ``gold_silver_ratio_mom``— 10-bar momentum of the G/S ratio.
            * ``vix_proxy``            — Mean ATR ratio across all assets.
            * ``cross_asset_avg_corr`` — Mean pairwise 20-bar return
                                         correlation across all assets.
            * ``avg_atr_ratio``        — Mean ATR ratio (same as vix_proxy,
                                         kept for explicitness).

            The index matches the union index of the provided series after
            NaN-padding for the warm-up period.  The **last row** represents
            the current state.
        """
        self._logger.debug(
            "[%s] Computing universal features for assets: %s",
            self.symbol,
            list(asset_data.keys()),
        )

        # ------------------------------------------------------------------
        # Determine a reference index from the largest DataFrame
        # ------------------------------------------------------------------
        if not asset_data:
            raise ValueError("asset_data must not be empty.")

        ref_df = max(asset_data.values(), key=len)
        ref_index = ref_df.index

        # ------------------------------------------------------------------
        # Per-asset returns and ATR ratios
        # ------------------------------------------------------------------
        all_returns: dict[str, pd.Series] = {}
        all_atr_ratios: dict[str, pd.Series] = {}

        for ticker, df in asset_data.items():
            df = df.reindex(ref_index)
            all_returns[ticker] = self._safe_returns(df)
            all_atr_ratios[ticker] = self._atr_ratio(df)

        gold_returns   = all_returns.get("XAUUSD")
        silver_returns = all_returns.get("XAGUSD")
        dxy_returns    = all_returns.get("DXY")

        # ------------------------------------------------------------------
        # Feature: dxy_momentum
        # ------------------------------------------------------------------
        if "DXY" in asset_data:
            dxy_close = asset_data["DXY"]["close"].reindex(ref_index)
            dxy_momentum = dxy_close.pct_change(10).fillna(0.0)
        else:
            dxy_momentum = pd.Series(0.0, index=ref_index)

        # ------------------------------------------------------------------
        # Feature: gold_dxy_corr
        # ------------------------------------------------------------------
        if gold_returns is not None and dxy_returns is not None:
            gold_dxy_corr = rolling_corr(gold_returns, dxy_returns, 20).fillna(0.0)
        else:
            gold_dxy_corr = pd.Series(0.0, index=ref_index)

        # ------------------------------------------------------------------
        # Feature: gold_silver_corr
        # ------------------------------------------------------------------
        if gold_returns is not None and silver_returns is not None:
            gold_silver_corr = rolling_corr(
                gold_returns, silver_returns, 20
            ).fillna(0.0)
        else:
            gold_silver_corr = pd.Series(0.0, index=ref_index)

        # ------------------------------------------------------------------
        # Feature: gold_silver_ratio & momentum
        # ------------------------------------------------------------------
        if "XAUUSD" in asset_data and "XAGUSD" in asset_data:
            gold_close   = asset_data["XAUUSD"]["close"].reindex(ref_index)
            silver_close = asset_data["XAGUSD"]["close"].reindex(ref_index)
            gold_silver_ratio     = (gold_close / silver_close.replace(0, np.nan)).fillna(method="ffill")
            gold_silver_ratio_mom = gold_silver_ratio.pct_change(10).fillna(0.0)
        else:
            gold_silver_ratio     = pd.Series(0.0, index=ref_index)
            gold_silver_ratio_mom = pd.Series(0.0, index=ref_index)

        # ------------------------------------------------------------------
        # Feature: vix_proxy — mean ATR ratio across all assets
        # ------------------------------------------------------------------
        if all_atr_ratios:
            atr_ratio_df = pd.DataFrame(all_atr_ratios, index=ref_index)
            vix_proxy    = atr_ratio_df.mean(axis=1).fillna(1.0)
            avg_atr_ratio = vix_proxy.copy()
        else:
            vix_proxy     = pd.Series(1.0, index=ref_index)
            avg_atr_ratio = pd.Series(1.0, index=ref_index)

        # ------------------------------------------------------------------
        # Feature: cross_asset_avg_corr — mean pairwise 20-bar correlation
        # ------------------------------------------------------------------
        tickers = list(all_returns.keys())
        n_assets = len(tickers)

        if n_assets >= 2:
            returns_df   = pd.DataFrame(all_returns, index=ref_index)
            rolling_corrs: list[pd.Series] = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    pair_corr = rolling_corr(
                        returns_df.iloc[:, i],
                        returns_df.iloc[:, j],
                        20,
                    )
                    rolling_corrs.append(pair_corr)
            cross_asset_avg_corr = (
                pd.concat(rolling_corrs, axis=1).mean(axis=1).fillna(0.0)
            )
        else:
            cross_asset_avg_corr = pd.Series(0.0, index=ref_index)

        # ------------------------------------------------------------------
        # Assemble output DataFrame
        # ------------------------------------------------------------------
        features = pd.DataFrame(
            {
                "dxy_momentum":          dxy_momentum,
                "gold_dxy_corr":         gold_dxy_corr,
                "gold_silver_corr":      gold_silver_corr,
                "gold_silver_ratio":     gold_silver_ratio,
                "gold_silver_ratio_mom": gold_silver_ratio_mom,
                "vix_proxy":             vix_proxy,
                "cross_asset_avg_corr":  cross_asset_avg_corr,
                "avg_atr_ratio":         avg_atr_ratio,
            },
            index=ref_index,
        )

        return features

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def generate_labels(self, features: pd.DataFrame) -> pd.Series:
        """Auto-generate regime labels from features for supervised training.

        Applies a simple rule hierarchy to the computed feature columns.

        Rules (evaluated in order):

        1. ``vix_proxy > 2.0`` → ``CRISIS``
        2. ``vix_proxy > 1.4`` AND ``cross_asset_avg_corr > 0.6`` → ``RISK_OFF``
        3. ``vix_proxy < 0.85`` AND ``gold_dxy_corr < -0.50`` → ``RISK_ON``
        4. Otherwise → ``NEUTRAL``

        Parameters
        ----------
        features:
            DataFrame produced by :meth:`compute_features`.  Must contain
            columns ``vix_proxy``, ``cross_asset_avg_corr``, and
            ``gold_dxy_corr``.

        Returns
        -------
        pd.Series of str
            Series of risk-state labels aligned to *features*.

        Raises
        ------
        KeyError
            If required columns are absent from *features*.
        """
        required = {"vix_proxy", "cross_asset_avg_corr", "gold_dxy_corr"}
        missing = required - set(features.columns)
        if missing:
            raise KeyError(
                f"[{self.symbol}] generate_labels: missing columns: {missing}"
            )

        vix    = features["vix_proxy"]
        corr   = features["cross_asset_avg_corr"]
        gd_cor = features["gold_dxy_corr"]

        labels = pd.Series(NEUTRAL, index=features.index, dtype=str)
        labels = labels.where(~(vix < 0.85) | ~(gd_cor < -0.50), RISK_ON)
        labels = labels.where(~(vix > 1.4) | ~(corr > 0.6), RISK_OFF)
        labels = labels.where(~(vix > 2.0), CRISIS)

        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train an XGBoost multi-class classifier on labelled cross-asset data.

        The model maps the feature columns produced by :meth:`compute_features`
        to one of the four risk states (``RISK_ON``, ``NEUTRAL``,
        ``RISK_OFF``, ``CRISIS``).

        Parameters
        ----------
        features:
            DataFrame of shape ``(n_samples, n_features)`` produced by
            :meth:`compute_features`.
        labels:
            Series of risk-state strings aligned to *features*.  Typically
            generated by :meth:`generate_labels`.

        Raises
        ------
        ValueError
            If *features* and *labels* have different lengths, or if any
            label is not a recognised risk state.
        RuntimeError
            If XGBoost raises an unrecoverable error during fitting.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"[{self.symbol}] features ({len(features)}) and labels "
                f"({len(labels)}) must have the same length."
            )

        unknown_labels = set(labels.unique()) - set(_RISK_STATES)
        if unknown_labels:
            raise ValueError(
                f"[{self.symbol}] Unknown risk-state labels: {unknown_labels}"
            )

        self._logger.info(
            "[%s] Training UniversalModel — n_samples=%d",
            self.symbol,
            len(features),
        )

        # Drop rows where features are all-NaN (warm-up period)
        valid_mask = features.notna().all(axis=1)
        X = features.loc[valid_mask].to_numpy(dtype=float)
        y_raw = labels.loc[valid_mask]

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y_raw)
        n_classes = len(self._label_encoder.classes_)

        params = {
            **_DEFAULT_XGB_PARAMS,
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        }

        model = xgb.XGBClassifier(**params)
        try:
            model.fit(X, y)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.symbol}] XGBoost fit failed: {exc}"
            ) from exc

        self._model = model
        self._feature_names = list(features.columns)
        self._is_trained = True

        self._metadata = {
            "training_date": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": int(len(features)),
            "n_classes": n_classes,
            "classes": list(self._label_encoder.classes_),
            "symbol": self.symbol,
        }

        self._logger.info(
            "[%s] UniversalModel training complete. Classes: %s",
            self.symbol,
            list(self._label_encoder.classes_),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> dict:
        """Predict the global macro risk state.

        Uses the trained XGBoost model if available; otherwise applies a
        deterministic rule-based fallback derived from the feature values in
        the **last row** of *features*.

        Rule-based fallback (evaluated in order):

        1. ``vix_proxy > 2.0`` → ``CRISIS``
        2. ``vix_proxy > 1.5`` → ``RISK_OFF``
        3. ``vix_proxy < 0.8`` AND ``gold_dxy_corr < -0.5`` → ``RISK_ON``
        4. Otherwise → ``NEUTRAL``

        Parameters
        ----------
        features:
            DataFrame produced by :meth:`compute_features`.

        Returns
        -------
        dict with keys:

        - ``"global_risk_state"`` (str): One of the four risk-state constants.
        - ``"global_multiplier"`` (float): Position-size multiplier from
          config (or hardcoded default).
        - ``"confidence"`` (float): Model probability of the predicted class,
          or ``1.0`` for the rule-based path.

        Raises
        ------
        KeyError
            If required feature columns are absent when using the fallback.
        """
        latest = features.iloc[-1]

        if self._is_trained and self._model is not None and self._label_encoder is not None:
            # ---- Model-based prediction ----------------------------------
            X_arr = latest.to_numpy(dtype=float).reshape(1, -1)
            proba: np.ndarray = self._model.predict_proba(X_arr)[0]
            class_idx = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            state = str(self._label_encoder.classes_[class_idx])
        else:
            # ---- Rule-based fallback -------------------------------------
            vix_proxy    = float(latest.get("vix_proxy",   1.0))
            gold_dxy_cor = float(latest.get("gold_dxy_corr", 0.0))

            if vix_proxy > 2.0:
                state = CRISIS
            elif vix_proxy > 1.5:
                state = RISK_OFF
            elif vix_proxy < 0.8 and gold_dxy_cor < -0.5:
                state = RISK_ON
            else:
                state = NEUTRAL

            confidence = 1.0

        multiplier = self._multipliers.get(state, 0.7)

        return {
            "global_risk_state": state,
            "global_multiplier": multiplier,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Persist the trained model to *path* via joblib.

        Parameters
        ----------
        path:
            Filesystem path for the output file (e.g. ``"models/universal.pkl"``).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        OSError
            If the file cannot be written.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                f"[{self.symbol}] Model is not trained. Cannot save."
            )

        payload = {
            "model": self._model,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "multipliers": self._multipliers,
            "symbol": self.symbol,
            "metadata": self._metadata,
        }
        try:
            joblib.dump(payload, path)
        except OSError as exc:
            self._logger.error(
                "[%s] Failed to save model to %s: %s", self.symbol, path, exc
            )
            raise

        self._logger.info("[%s] UniversalModel saved to %s", self.symbol, path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model from *path*.

        Parameters
        ----------
        path:
            Path to the ``.pkl`` file produced by :meth:`save_model`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If the payload is missing expected keys.
        """
        try:
            payload: dict = joblib.load(path)
        except FileNotFoundError:
            self._logger.error(
                "[%s] Model file not found: %s", self.symbol, path
            )
            raise
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            self._logger.error(
                "[%s] Failed to deserialise model from %s: %s",
                self.symbol,
                path,
                exc,
            )
            raise

        try:
            self._model         = payload["model"]
            self._label_encoder = payload["label_encoder"]
            self._feature_names = payload["feature_names"]
        except KeyError as exc:
            raise KeyError(
                f"Saved model at '{path}' is missing required key: {exc}"
            ) from exc

        self._multipliers = payload.get("multipliers", _load_multipliers())
        self.symbol       = payload.get("symbol", self.symbol)
        self._metadata    = payload.get("metadata", {})
        self._is_trained  = True

        self._logger.info(
            "[%s] UniversalModel loaded from %s | classes=%s",
            self.symbol,
            path,
            (
                list(self._label_encoder.classes_)
                if self._label_encoder is not None
                else "n/a"
            ),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        status = "trained" if self._is_trained else "untrained"
        return f"UniversalModel(symbol={self.symbol!r}, status={status})"
