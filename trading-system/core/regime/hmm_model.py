"""HMM-based regime discovery model using hmmlearn GaussianHMM."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Regime label constants
# ---------------------------------------------------------------------------
REGIME_TREND_UP   = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGE      = "RANGE"
REGIME_VOLATILE   = "VOLATILE"
REGIME_SAFE_HAVEN = "SAFE_HAVEN"   # Gold only

# Feature columns the HMM is trained on (in this order)
_HMM_FEATURE_COLS: list[str] = [
    "log_return",
    "atr_ratio",
    "realised_vol_20",
    "volume_ratio",
]


class HMMRegimeModel:
    """Hidden Markov Model for unsupervised market-regime discovery.

    Fits a Gaussian HMM on a compact four-feature representation:
    ``[log_return, atr_ratio, realised_vol_20, volume_ratio]``.

    After training, states are automatically mapped to human-readable
    regime labels via :meth:`_map_states`.

    Parameters
    ----------
    n_states:
        Number of hidden states (4 for all symbols, 5 for Gold to capture
        the additional ``SAFE_HAVEN`` regime).
    covariance_type:
        HMM covariance structure passed straight through to
        :class:`hmmlearn.hmm.GaussianHMM`.
    n_iter:
        Maximum EM iterations.
    random_state:
        Seed for reproducibility.
    symbol:
        Instrument identifier (e.g. ``"XAUUSD"``).  Used only for logging
        and metadata — does not change behaviour by itself.
    """

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        symbol: str = "UNKNOWN",
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.symbol = symbol

        self._logger = get_logger(__name__)

        self._hmm: Optional[GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self.state_mapping: dict[int, str] = {}
        self._is_trained: bool = False
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_hmm_features(
        self,
        features: np.ndarray,
        feature_names: Optional[list[str]],
    ) -> np.ndarray:
        """Return the four-column sub-matrix used for HMM training/inference.

        If *feature_names* is provided the columns are selected by name;
        otherwise the first four columns are assumed to be in the correct
        order ``[log_return, atr_ratio, realised_vol_20, volume_ratio]``.

        Parameters
        ----------
        features:
            Full feature matrix with shape ``(n_samples, n_features)``.
        feature_names:
            Optional list of column names matching axis-1 of *features*.

        Returns
        -------
        np.ndarray
            Sub-matrix of shape ``(n_samples, 4)``.

        Raises
        ------
        ValueError
            If any required column name is missing from *feature_names*.
        """
        if feature_names is not None:
            missing = [c for c in _HMM_FEATURE_COLS if c not in feature_names]
            if missing:
                raise ValueError(
                    f"HMM feature columns missing from feature_names: {missing}"
                )
            indices = [feature_names.index(col) for col in _HMM_FEATURE_COLS]
            return features[:, indices]

        if features.shape[1] < 4:
            raise ValueError(
                f"Expected at least 4 feature columns, got {features.shape[1]}"
            )
        return features[:, :4]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        features: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """Train the HMM on a feature matrix.

        Selects the four HMM-specific columns
        ``[log_return, atr_ratio, realised_vol_20, volume_ratio]``,
        standardises them, fits a :class:`~hmmlearn.hmm.GaussianHMM`, and
        auto-maps discovered states to regime labels via :meth:`_map_states`.

        Parameters
        ----------
        features:
            2-D array of shape ``(n_samples, n_features)``.  Must not
            contain NaN or infinite values.
        feature_names:
            Optional list of column names.  When provided, columns are
            selected by name rather than position.

        Raises
        ------
        ValueError
            If *features* contains fewer rows than twice the number of
            states, or if required columns are missing.
        RuntimeError
            If the HMM fails to converge.
        """
        self._logger.info(
            "[%s] Starting HMM training — n_states=%d, n_samples=%d",
            self.symbol,
            self.n_states,
            features.shape[0],
        )

        if features.shape[0] < self.n_states * 2:
            raise ValueError(
                f"Too few samples ({features.shape[0]}) to train a "
                f"{self.n_states}-state HMM."
            )

        X = self._select_hmm_features(features, feature_names)

        # Remove any remaining NaN / inf rows
        valid_mask = np.isfinite(X).all(axis=1)
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            self._logger.warning(
                "[%s] Dropping %d rows with NaN/inf before HMM training.",
                self.symbol,
                n_dropped,
            )
        X = X[valid_mask]

        if X.shape[0] < self.n_states * 2:
            raise ValueError(
                f"After dropping invalid rows only {X.shape[0]} samples remain — "
                "too few to fit the HMM."
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

        try:
            self._hmm.fit(X_scaled)
        except Exception as exc:  # hmmlearn can raise various runtime errors
            raise RuntimeError(f"HMM fitting failed: {exc}") from exc

        if not self._hmm.monitor_.converged:
            self._logger.warning(
                "[%s] HMM did not converge after %d iterations.",
                self.symbol,
                self.n_iter,
            )

        self._is_trained = True

        # Map integer states -> regime names using full (possibly larger) X
        X_map = self._select_hmm_features(features, feature_names)
        X_map = X_map[np.isfinite(X_map).all(axis=1)]
        self._map_states(X_map)

        self._metadata = {
            "training_date": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": int(X.shape[0]),
            "feature_names": feature_names if feature_names is not None else _HMM_FEATURE_COLS,
            "symbol": self.symbol,
            "n_states": self.n_states,
            "converged": bool(self._hmm.monitor_.converged),
        }

        self._logger.info(
            "[%s] HMM training complete. State mapping: %s",
            self.symbol,
            self.state_mapping,
        )

    def predict(
        self,
        features: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Viterbi decoding and posterior-probability estimation.

        Parameters
        ----------
        features:
            2-D array of shape ``(n_samples, n_features)``.
        feature_names:
            Optional list of column names.

        Returns
        -------
        state_sequence : np.ndarray of str, shape ``(n_samples,)``
            Mapped regime labels for each bar (Viterbi path).
        state_probabilities : np.ndarray of float, shape ``(n_samples, n_states)``
            Posterior (smoothed) state probabilities.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._hmm is None or self._scaler is None:
            raise RuntimeError(
                "Model is not trained. Call train() before predict()."
            )

        X = self._select_hmm_features(features, feature_names)
        X_scaled = self._scaler.transform(X)

        viterbi_states: np.ndarray = self._hmm.predict(X_scaled)
        state_probs: np.ndarray = self._hmm.predict_proba(X_scaled)

        mapped_labels = np.array(
            [self.state_mapping.get(s, REGIME_RANGE) for s in viterbi_states],
            dtype=object,
        )
        return mapped_labels, state_probs

    def predict_latest(
        self,
        features: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Predict the regime for the most recent bar (last row).

        Parameters
        ----------
        features:
            2-D array of shape ``(n_samples, n_features)``.  Only the last
            row is used for the returned prediction; however the full matrix
            is passed through :meth:`predict` so that the HMM's transition
            dynamics are respected.
        feature_names:
            Optional list of column names.

        Returns
        -------
        dict with keys:
            - ``"regime"`` (str): Predicted regime label for the latest bar.
            - ``"probabilities"`` (dict[str, float]): Posterior probability
              for every known regime label.
            - ``"confidence"`` (float): Maximum posterior probability.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._hmm is None or self._scaler is None:
            raise RuntimeError(
                "Model is not trained. Call train() before predict_latest()."
            )

        mapped_labels, state_probs = self.predict(features, feature_names)

        latest_regime: str = str(mapped_labels[-1])
        latest_probs: np.ndarray = state_probs[-1]

        # Build per-label probabilities (aggregate if two states share a label)
        label_probs: dict[str, float] = {}
        for state_idx, prob in enumerate(latest_probs):
            label = self.state_mapping.get(state_idx, REGIME_RANGE)
            label_probs[label] = label_probs.get(label, 0.0) + float(prob)

        confidence = float(latest_probs.max())

        return {
            "regime": latest_regime,
            "probabilities": label_probs,
            "confidence": confidence,
        }

    def _map_states(self, features: np.ndarray) -> None:
        """Auto-map HMM integer states to regime-name strings.

        Uses the **un-scaled** four-column feature matrix to compute per-state
        statistics, then applies the following heuristic:

        1. Decode all bars with the trained HMM (using the stored scaler).
        2. For each state compute: mean ``log_return``, mean ``atr_ratio``,
           mean ``realised_vol_20``.
        3. The state with the **lowest** ``atr_ratio`` AND ``realised_vol_20``
           is labelled ``RANGE``.
        4. The state with the **highest** ``realised_vol_20`` is labelled
           ``VOLATILE``.
        5. Among the remaining states, the one with the highest ``atr_ratio``
           and **positive** mean return → ``TREND_UP``; the other →
           ``TREND_DOWN``.
        6. For ``n_states == 5`` the one remaining state is labelled
           ``SAFE_HAVEN``.
        7. Fallback: if any assignment is still ambiguous (e.g. too few
           distinct states), states are labelled in index order:
           TREND_UP / TREND_DOWN / RANGE / VOLATILE [/ SAFE_HAVEN].

        The result is stored in ``self.state_mapping``.

        Parameters
        ----------
        features:
            Un-scaled 2-D array of shape ``(n_samples, 4)`` with columns
            ``[log_return, atr_ratio, realised_vol_20, volume_ratio]``.
        """
        assert self._hmm is not None and self._scaler is not None

        X_scaled = self._scaler.transform(features)
        try:
            states: np.ndarray = self._hmm.predict(X_scaled)
        except Exception as exc:
            self._logger.error(
                "[%s] _map_states: HMM predict failed (%s). Using fallback mapping.",
                self.symbol,
                exc,
            )
            self._apply_fallback_mapping()
            return

        # Column indices in the 4-col sub-matrix
        IDX_LOG_RETURN   = 0
        IDX_ATR_RATIO    = 1
        IDX_REALISED_VOL = 2

        unique_states = np.unique(states)
        if len(unique_states) < 2:
            self._logger.warning(
                "[%s] Only %d unique HMM state(s) found. Using fallback mapping.",
                self.symbol,
                len(unique_states),
            )
            self._apply_fallback_mapping()
            return

        # Compute per-state statistics
        state_stats: dict[int, dict[str, float]] = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                # State never visited — use zeros
                state_stats[s] = {
                    "mean_return": 0.0,
                    "mean_atr_ratio": 0.0,
                    "mean_realised_vol": 0.0,
                }
            else:
                state_stats[s] = {
                    "mean_return": float(features[mask, IDX_LOG_RETURN].mean()),
                    "mean_atr_ratio": float(features[mask, IDX_ATR_RATIO].mean()),
                    "mean_realised_vol": float(features[mask, IDX_REALISED_VOL].mean()),
                }

        all_states = list(range(self.n_states))

        # Step 1: RANGE — lowest atr_ratio AND lowest realised_vol
        # Use a composite score (rank sum) to handle ties
        atr_ranks = sorted(all_states, key=lambda s: state_stats[s]["mean_atr_ratio"])
        vol_ranks = sorted(all_states, key=lambda s: state_stats[s]["mean_realised_vol"])
        range_scores = {
            s: atr_ranks.index(s) + vol_ranks.index(s) for s in all_states
        }
        range_state = min(all_states, key=lambda s: range_scores[s])

        # Step 2: VOLATILE — highest realised_vol
        volatile_state = max(all_states, key=lambda s: state_stats[s]["mean_realised_vol"])

        # If RANGE and VOLATILE landed on the same state, break the tie:
        # give VOLATILE priority and assign RANGE to the second-lowest vol state
        if range_state == volatile_state:
            remaining_for_range = [s for s in all_states if s != volatile_state]
            range_state = min(
                remaining_for_range,
                key=lambda s: range_scores[s],
            )

        assigned = {range_state: REGIME_RANGE, volatile_state: REGIME_VOLATILE}
        remaining = [s for s in all_states if s not in assigned]

        # Step 3: TREND_UP / TREND_DOWN among remaining states
        if len(remaining) >= 2:
            # Highest atr_ratio among remaining → trend; sign of return picks direction
            trend_state = max(remaining, key=lambda s: state_stats[s]["mean_atr_ratio"])
            non_trend = [s for s in remaining if s != trend_state]

            if state_stats[trend_state]["mean_return"] >= 0:
                assigned[trend_state] = REGIME_TREND_UP
                # Secondary trend state: other remaining gets TREND_DOWN
                if non_trend:
                    assigned[non_trend[0]] = REGIME_TREND_DOWN
            else:
                assigned[trend_state] = REGIME_TREND_DOWN
                if non_trend:
                    assigned[non_trend[0]] = REGIME_TREND_UP

            # Any further remaining states (n_states == 5 Gold case)
            remaining_after_trend = [s for s in remaining if s not in assigned]
        elif len(remaining) == 1:
            # Only one unassigned state — assign both TREND labels heuristically
            s = remaining[0]
            if state_stats[s]["mean_return"] >= 0:
                assigned[s] = REGIME_TREND_UP
            else:
                assigned[s] = REGIME_TREND_DOWN
            remaining_after_trend = []
        else:
            remaining_after_trend = []

        # Step 4: SAFE_HAVEN for 5-state Gold model
        if self.n_states == 5 and remaining_after_trend:
            assigned[remaining_after_trend[0]] = REGIME_SAFE_HAVEN
            remaining_after_trend = remaining_after_trend[1:]

        # Patch any still-unassigned states with a fallback label
        fallback_pool = [REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_VOLATILE]
        if self.n_states == 5:
            fallback_pool.append(REGIME_SAFE_HAVEN)
        used_labels = set(assigned.values())
        for s in all_states:
            if s not in assigned:
                for label in fallback_pool:
                    if label not in used_labels:
                        assigned[s] = label
                        used_labels.add(label)
                        break
                else:
                    # All canonical labels exhausted — reuse VOLATILE
                    assigned[s] = REGIME_VOLATILE

        self.state_mapping = assigned
        self._logger.debug(
            "[%s] State mapping resolved: %s | stats: %s",
            self.symbol,
            self.state_mapping,
            {
                s: {k: f"{v:.5f}" for k, v in stats.items()}
                for s, stats in state_stats.items()
            },
        )

    def _apply_fallback_mapping(self) -> None:
        """Map states to regime labels in ascending index order.

        Used when statistical heuristics cannot distinguish states.
        """
        labels = [
            REGIME_TREND_UP,
            REGIME_TREND_DOWN,
            REGIME_RANGE,
            REGIME_VOLATILE,
            REGIME_SAFE_HAVEN,
        ]
        self.state_mapping = {i: labels[i] for i in range(self.n_states)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Persist the trained model to disk.

        Saves a dict with keys ``"hmm"``, ``"scaler"``, ``"state_mapping"``,
        and ``"metadata"`` using :func:`joblib.dump`.

        Parameters
        ----------
        path:
            File-system path for the output file (e.g. ``"models/hmm.pkl"``).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        OSError
            If the file cannot be written.
        """
        if not self._is_trained or self._hmm is None or self._scaler is None:
            raise RuntimeError(
                "Model is not trained. Call train() before save_model()."
            )

        payload = {
            "hmm": self._hmm,
            "scaler": self._scaler,
            "state_mapping": self.state_mapping,
            "metadata": self._metadata,
        }
        try:
            joblib.dump(payload, path)
        except OSError as exc:
            self._logger.error("[%s] Failed to save model to %s: %s", self.symbol, path, exc)
            raise

        self._logger.info("[%s] Model saved to %s", self.symbol, path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model from disk.

        Restores the HMM, scaler, state mapping, and metadata from the file
        produced by :meth:`save_model`.

        Parameters
        ----------
        path:
            Path to the saved ``.pkl`` file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If the file is missing expected keys.
        """
        try:
            payload: dict = joblib.load(path)
        except FileNotFoundError:
            self._logger.error("[%s] Model file not found: %s", self.symbol, path)
            raise
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            self._logger.error(
                "[%s] Failed to deserialise model from %s: %s", self.symbol, path, exc
            )
            raise

        try:
            self._hmm = payload["hmm"]
            self._scaler = payload["scaler"]
            self.state_mapping = payload["state_mapping"]
            self._metadata = payload.get("metadata", {})
        except KeyError as exc:
            raise KeyError(
                f"Saved model at '{path}' is missing required key: {exc}"
            ) from exc

        self._is_trained = True
        self._logger.info(
            "[%s] Model loaded from %s | mapping: %s",
            self.symbol,
            path,
            self.state_mapping,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state_names(self) -> list[str]:
        """Return the regime label for every HMM state, ordered by state index.

        Returns
        -------
        list[str]
            E.g. ``["TREND_UP", "RANGE", "VOLATILE", "TREND_DOWN"]`` for a
            4-state model.

        Raises
        ------
        RuntimeError
            If the model has not been trained (state mapping is empty).
        """
        if not self.state_mapping:
            raise RuntimeError(
                "State mapping is not available. Train or load the model first."
            )
        return [self.state_mapping[i] for i in range(self.n_states)]

    def __repr__(self) -> str:  # pragma: no cover
        status = "trained" if self._is_trained else "untrained"
        return (
            f"HMMRegimeModel(symbol={self.symbol!r}, n_states={self.n_states}, "
            f"covariance_type={self.covariance_type!r}, status={status})"
        )
