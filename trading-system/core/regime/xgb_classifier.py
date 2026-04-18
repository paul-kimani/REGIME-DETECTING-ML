"""XGBoost regime classifier trained on HMM-generated labels."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Regime label constants
# ---------------------------------------------------------------------------
REGIME_NAMES: list[str] = [
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "VOLATILE",
    "SAFE_HAVEN",
]

# Default XGBoost hyperparameters
_DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class XGBRegimeClassifier:
    """XGBoost multi-class classifier for market regime detection.

    Wraps an :class:`xgboost.XGBClassifier` with optional HMM posterior
    probability features, Optuna hyperparameter search, and convenience
    methods for batch inference, feature-importance reporting, and
    model persistence.

    Parameters
    ----------
    symbol:
        Instrument identifier used in log messages (e.g. ``"XAUUSD"``).
    n_classes:
        Number of regime classes.  Use 4 for all instruments and 5 for
        Gold (adds the ``SAFE_HAVEN`` class).
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        n_classes: int = 4,
    ) -> None:
        self.symbol = symbol
        self.n_classes = n_classes

        self._logger = get_logger(__name__)
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_names: Optional[list[str]] = None
        self._feature_importances: Optional[pd.DataFrame] = None
        self._best_params: dict = {}
        self._hmm_cols_used: bool = False
        self._n_hmm_cols: int = 0
        self._is_trained: bool = False
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_hmm_features(
        self,
        features: pd.DataFrame,
        hmm_probs: Optional[np.ndarray],
    ) -> pd.DataFrame:
        """Return *features* with HMM posterior columns appended if provided.

        Columns are named ``hmm_prob_0``, ``hmm_prob_1``, … up to the number
        of columns in *hmm_probs*.

        Parameters
        ----------
        features:
            Feature DataFrame to augment.
        hmm_probs:
            2-D array of shape ``(n_samples, n_states)`` or ``None``.

        Returns
        -------
        pd.DataFrame
            Either the original *features* or a copy with HMM columns appended.

        Raises
        ------
        ValueError
            If the row count of *hmm_probs* does not match *features*.
        """
        if hmm_probs is None:
            return features

        if hmm_probs.ndim == 1:
            hmm_probs = hmm_probs.reshape(-1, 1)

        if hmm_probs.shape[0] != len(features):
            raise ValueError(
                f"hmm_probs has {hmm_probs.shape[0]} rows but features has "
                f"{len(features)} rows."
            )

        hmm_cols = {
            f"hmm_prob_{i}": hmm_probs[:, i]
            for i in range(hmm_probs.shape[1])
        }
        return features.assign(**hmm_cols)

    def _build_model(self, params: dict) -> xgb.XGBClassifier:
        """Construct an :class:`xgboost.XGBClassifier` from *params*.

        Always injects ``objective``, ``num_class``, and ``eval_metric``
        regardless of what is present in *params*.

        Parameters
        ----------
        params:
            Hyperparameter dict to pass to the classifier constructor.

        Returns
        -------
        xgb.XGBClassifier
        """
        merged = {**params}
        merged["objective"] = "multi:softprob"
        merged["num_class"] = self.n_classes
        merged["eval_metric"] = "mlogloss"
        return xgb.XGBClassifier(**merged)

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Convert string regime labels to integer class indices.

        Parameters
        ----------
        labels:
            Series of regime-name strings (e.g. ``"TREND_UP"``).

        Returns
        -------
        np.ndarray of int with values in ``[0, n_classes)``.

        Raises
        ------
        ValueError
            If any label is not present in :data:`REGIME_NAMES`.
        """
        label_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
        encoded = labels.map(label_to_idx)
        unknown = labels[encoded.isna()]
        if not unknown.empty:
            raise ValueError(
                f"[{self.symbol}] Unknown regime labels found: "
                f"{unknown.unique().tolist()}"
            )
        return encoded.to_numpy(dtype=int)

    def _store_feature_importances(self, model: xgb.XGBClassifier) -> None:
        """Compute and cache the feature-importance DataFrame.

        Parameters
        ----------
        model:
            A fitted :class:`xgboost.XGBClassifier`.
        """
        importances = model.feature_importances_
        names = (
            self._feature_names
            if self._feature_names is not None
            else [f"f{i}" for i in range(len(importances))]
        )
        self._feature_importances = (
            pd.DataFrame({"feature_name": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Public API — training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        hmm_probs: Optional[np.ndarray] = None,
        use_optuna: bool = False,
        n_trials: int = 50,
    ) -> None:
        """Train the XGBoost multi-class classifier.

        If *hmm_probs* is provided the HMM posterior columns are appended to
        the feature matrix before training and the same transformation will be
        applied at inference time.

        When *use_optuna* is ``True``, 20 % of the data is held out as a
        validation set and Optuna maximises accuracy over *n_trials* trials.
        The best hyperparameters are then used to refit on the full dataset.

        Parameters
        ----------
        features:
            DataFrame of shape ``(n_samples, n_features)``.  Must not
            contain NaN or infinite values after HMM column augmentation.
        labels:
            Series of regime-name strings aligned to *features*.
        hmm_probs:
            Optional 2-D array of shape ``(n_samples, n_states)`` containing
            HMM posterior probabilities.
        use_optuna:
            Whether to run Optuna hyperparameter search before final fit.
        n_trials:
            Number of Optuna trials (ignored when *use_optuna* is ``False``).

        Raises
        ------
        ValueError
            If *features* and *labels* have different lengths, or if any
            label is not a recognised regime name.
        RuntimeError
            If Optuna or XGBoost raise an unrecoverable error.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"[{self.symbol}] features ({len(features)}) and labels "
                f"({len(labels)}) must have the same length."
            )

        self._logger.info(
            "[%s] Starting XGBoost training — n_samples=%d, n_classes=%d, "
            "use_optuna=%s",
            self.symbol,
            len(features),
            self.n_classes,
            use_optuna,
        )

        # Augment with HMM columns if provided
        X = self._append_hmm_features(features, hmm_probs)
        self._hmm_cols_used = hmm_probs is not None
        self._n_hmm_cols = hmm_probs.shape[1] if hmm_probs is not None else 0
        self._feature_names = list(X.columns)

        y = self._encode_labels(labels)
        X_arr = X.to_numpy(dtype=float)

        # ------------------------------------------------------------------
        # Optuna search
        # ------------------------------------------------------------------
        if use_optuna:
            import optuna  # local import — optional dependency

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_arr,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            def _objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": 0,
                }
                clf = self._build_model(params)
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_val)
                accuracy = float((preds == y_val).mean())
                return accuracy

            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

            self._best_params = {**_DEFAULT_XGB_PARAMS, **study.best_params}
            self._logger.info(
                "[%s] Optuna finished. Best accuracy=%.4f | params=%s",
                self.symbol,
                study.best_value,
                study.best_params,
            )
        else:
            self._best_params = dict(_DEFAULT_XGB_PARAMS)

        # ------------------------------------------------------------------
        # Final fit on full data
        # ------------------------------------------------------------------
        model = self._build_model(self._best_params)
        try:
            model.fit(X_arr, y)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.symbol}] XGBoost fit failed: {exc}"
            ) from exc

        self._model = model
        self._store_feature_importances(model)
        self._is_trained = True

        self._metadata = {
            "training_date": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": int(len(features)),
            "n_classes": self.n_classes,
            "symbol": self.symbol,
            "hmm_cols_used": self._hmm_cols_used,
            "n_hmm_cols": self._n_hmm_cols,
            "use_optuna": use_optuna,
            "best_params": self._best_params,
        }

        self._logger.info(
            "[%s] XGBoost training complete — %d features.",
            self.symbol,
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Public API — inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        hmm_probs: Optional[np.ndarray] = None,
    ) -> dict:
        """Predict the market regime for the most recent row of *features*.

        Parameters
        ----------
        features:
            DataFrame with the same columns used during training (before HMM
            augmentation).  Only the **last row** is used for the returned
            prediction.
        hmm_probs:
            Optional HMM posterior array.  Must be provided if and only if
            HMM columns were used during training.

        Returns
        -------
        dict with keys:

        - ``"regime"`` (str): Predicted regime label.
        - ``"probabilities"`` (dict[str, float]): Class probabilities keyed
          by regime name.
        - ``"confidence"`` (float): Maximum class probability.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                f"[{self.symbol}] Model is not trained. Call train() first."
            )

        X = self._append_hmm_features(features, hmm_probs)
        X_arr = X.to_numpy(dtype=float)[-1:, :]  # single row

        proba: np.ndarray = self._model.predict_proba(X_arr)[0]
        class_idx = int(np.argmax(proba))
        regime = REGIME_NAMES[class_idx]
        confidence = float(proba[class_idx])

        probabilities = {
            REGIME_NAMES[i]: float(proba[i])
            for i in range(self.n_classes)
        }

        return {
            "regime": regime,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    def predict_batch(
        self,
        features: pd.DataFrame,
        hmm_probs: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Predict the regime for every row of *features*.

        Parameters
        ----------
        features:
            DataFrame of shape ``(n_samples, n_features)``.
        hmm_probs:
            Optional HMM posterior array of shape ``(n_samples, n_states)``.

        Returns
        -------
        list[dict]
            One prediction dict per row (same format as :meth:`predict`).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                f"[{self.symbol}] Model is not trained. Call train() first."
            )

        X = self._append_hmm_features(features, hmm_probs)
        X_arr = X.to_numpy(dtype=float)

        proba_matrix: np.ndarray = self._model.predict_proba(X_arr)
        results: list[dict] = []
        for proba in proba_matrix:
            class_idx = int(np.argmax(proba))
            results.append(
                {
                    "regime": REGIME_NAMES[class_idx],
                    "probabilities": {
                        REGIME_NAMES[i]: float(proba[i])
                        for i in range(self.n_classes)
                    },
                    "confidence": float(proba[class_idx]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name`` (str), ``importance`` (float).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._feature_importances is None:
            raise RuntimeError(
                f"[{self.symbol}] Model is not trained. Call train() first."
            )
        return self._feature_importances.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Persist the trained model and metadata to *path* via joblib.

        Parameters
        ----------
        path:
            Filesystem path for the output file (e.g. ``"models/xgb.pkl"``).

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
            "feature_names": self._feature_names,
            "feature_importances": self._feature_importances,
            "best_params": self._best_params,
            "hmm_cols_used": self._hmm_cols_used,
            "n_hmm_cols": self._n_hmm_cols,
            "n_classes": self.n_classes,
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

        self._logger.info("[%s] Model saved to %s", self.symbol, path)

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
            self._model = payload["model"]
            self._feature_names = payload["feature_names"]
            self._feature_importances = payload["feature_importances"]
            self._best_params = payload.get("best_params", {})
            self._hmm_cols_used = payload.get("hmm_cols_used", False)
            self._n_hmm_cols = payload.get("n_hmm_cols", 0)
            self.n_classes = payload.get("n_classes", self.n_classes)
            self.symbol = payload.get("symbol", self.symbol)
            self._metadata = payload.get("metadata", {})
        except KeyError as exc:
            raise KeyError(
                f"Saved model at '{path}' is missing required key: {exc}"
            ) from exc

        self._is_trained = True
        self._logger.info(
            "[%s] Model loaded from %s | features=%d",
            self.symbol,
            path,
            len(self._feature_names) if self._feature_names else 0,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        status = "trained" if self._is_trained else "untrained"
        return (
            f"XGBRegimeClassifier(symbol={self.symbol!r}, "
            f"n_classes={self.n_classes}, status={status})"
        )
