"""DriftDetector — PSI-based feature drift and regime distribution shift detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PSI_STABLE: float = 0.10        # PSI below this is considered stable
_PSI_SLIGHT: float = 0.20        # PSI above this triggers WARNING log
_PSI_SIGNIFICANT: float = 0.25   # PSI above this triggers ERROR log (critical drift)
_EPSILON: float = 1e-4            # Small value added to avoid log(0)

_REGIME_SHIFT_THRESHOLD: float = 0.20   # abs frequency diff (20 pp) flags a shifted regime
_REGIME_SHIFT_MAX_DIFF_RETRAIN: float = 0.30  # max_diff above this triggers retrain

_MIN_PSI_RETRAIN_FEATURES: int = 3       # >= 3 features with PSI > 0.20 → retrain


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """PSI-based feature drift and regime distribution shift detection."""

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)

        Steps:
        1. Create bucket edges from *expected* using percentile-based breaks.
        2. Bin both *expected* and *actual* into those buckets.
        3. Add a small epsilon (1e-4) to each frequency to avoid log(0).
        4. Accumulate PSI contributions per bucket.

        Interpretation:
        - PSI < 0.10  → stable (negligible distribution shift)
        - 0.10–0.20  → slight shift (monitor closely)
        - PSI > 0.20  → significant shift (consider retraining)

        Args:
            expected: 1-D array representing the training / reference distribution.
            actual:   1-D array representing the live / current distribution.
            buckets:  Number of equal-probability buckets to use (default 10).

        Returns:
            PSI as a non-negative float.
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        if len(expected) == 0 or len(actual) == 0:
            logger.warning("compute_psi: received empty array — returning 0.0")
            return 0.0

        # --- Build percentile-based bucket edges from the expected distribution ---
        percentiles = np.linspace(0, 100, buckets + 1)
        edges = np.percentile(expected, percentiles)

        # Ensure strictly increasing edges (collapse duplicates by adding tiny offset)
        edges = np.unique(edges)
        if len(edges) < 2:
            logger.warning(
                "compute_psi: degenerate expected distribution (all identical values) "
                "— returning 0.0"
            )
            return 0.0

        # Extend the outermost edges to capture all actual values
        edges[0] = -np.inf
        edges[-1] = np.inf

        # --- Compute frequencies ---
        expected_counts, _ = np.histogram(expected, bins=edges)
        actual_counts, _ = np.histogram(actual, bins=edges)

        n_expected = float(len(expected))
        n_actual = float(len(actual))

        expected_pct = (expected_counts / n_expected) + _EPSILON
        actual_pct = (actual_counts / n_actual) + _EPSILON

        # --- PSI contribution per bucket ---
        psi_contributions = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi = float(np.sum(psi_contributions))

        logger.debug("compute_psi: buckets=%d PSI=%.4f", len(edges) - 1, psi)
        return psi

    # ------------------------------------------------------------------
    # Feature drift
    # ------------------------------------------------------------------

    def check_feature_drift(
        self,
        training_features: pd.DataFrame,
        live_features: pd.DataFrame,
        top_n: int = 15,
    ) -> dict:
        """Compute PSI for each feature column.

        When the DataFrame has more than *top_n* columns the method uses only
        the first *top_n* columns (assumes they are already sorted by importance
        by the caller).

        Args:
            training_features: DataFrame of training-time feature values.
            live_features:     DataFrame of live feature values (same columns).
            top_n:             Maximum number of features to evaluate.

        Returns:
            Dict mapping ``feature_name`` → ``psi_score``.
        """
        if training_features.empty or live_features.empty:
            logger.warning(
                "check_feature_drift: one or both DataFrames are empty — "
                "returning empty results"
            )
            return {}

        # Restrict to shared columns, then trim to top_n
        shared_cols = [
            c for c in training_features.columns if c in live_features.columns
        ]
        if not shared_cols:
            logger.error(
                "check_feature_drift: no shared columns between training and live DataFrames"
            )
            return {}

        cols_to_check = shared_cols[:top_n] if len(shared_cols) > top_n else shared_cols

        results: dict[str, float] = {}
        for col in cols_to_check:
            train_vals = training_features[col].dropna().to_numpy(dtype=float)
            live_vals = live_features[col].dropna().to_numpy(dtype=float)

            if len(train_vals) == 0 or len(live_vals) == 0:
                logger.debug(
                    "check_feature_drift: skipping %s — no non-null values", col
                )
                continue

            psi = self.compute_psi(train_vals, live_vals)
            results[col] = psi

            if psi > _PSI_SIGNIFICANT:
                logger.error(
                    "check_feature_drift: CRITICAL drift on feature '%s' — "
                    "PSI=%.4f (> %.2f)",
                    col,
                    psi,
                    _PSI_SIGNIFICANT,
                )
            elif psi > _PSI_SLIGHT:
                logger.warning(
                    "check_feature_drift: significant drift on feature '%s' — "
                    "PSI=%.4f (> %.2f)",
                    col,
                    psi,
                    _PSI_SLIGHT,
                )
            else:
                logger.debug(
                    "check_feature_drift: '%s' stable — PSI=%.4f", col, psi
                )

        logger.info(
            "check_feature_drift: evaluated %d feature(s). "
            "WARNING (PSI>%.2f): %d  CRITICAL (PSI>%.2f): %d",
            len(results),
            _PSI_SLIGHT,
            sum(1 for v in results.values() if v > _PSI_SLIGHT),
            _PSI_SIGNIFICANT,
            sum(1 for v in results.values() if v > _PSI_SIGNIFICANT),
        )
        return results

    # ------------------------------------------------------------------
    # Regime distribution shift
    # ------------------------------------------------------------------

    def check_regime_distribution_shift(
        self,
        training_regimes: pd.Series,
        live_regimes: pd.Series,
    ) -> dict:
        """Compare regime frequency distributions between training and live data.

        A regime is flagged as *shifted* when its frequency differs by more than
        20 percentage points between the two datasets.

        Args:
            training_regimes: Categorical series of regime labels in training data.
            live_regimes:     Categorical series of regime labels in live data.

        Returns:
            Dict with keys:
            - ``"shift_detected"`` (bool): True when any regime exceeds the threshold.
            - ``"regime_diffs"`` (dict):   ``{regime_label: abs_diff_pct}`` for all regimes.
            - ``"max_diff"`` (float):      Largest absolute percentage-point difference.
            - ``"shifted_regimes"`` (list): Regime labels that exceeded the threshold.
        """
        if training_regimes.empty or live_regimes.empty:
            logger.warning(
                "check_regime_distribution_shift: empty series — returning no-shift result"
            )
            return {
                "shift_detected": False,
                "regime_diffs": {},
                "max_diff": 0.0,
                "shifted_regimes": [],
            }

        # Compute frequency fractions (value_counts normalises to sum=1)
        train_freq: pd.Series = training_regimes.value_counts(normalize=True)
        live_freq: pd.Series = live_regimes.value_counts(normalize=True)

        # Union of all known regimes
        all_regimes = set(train_freq.index) | set(live_freq.index)

        regime_diffs: dict[str, float] = {}
        for regime in all_regimes:
            train_pct = float(train_freq.get(regime, 0.0))
            live_pct = float(live_freq.get(regime, 0.0))
            regime_diffs[str(regime)] = abs(live_pct - train_pct)

        shifted_regimes = [
            r for r, diff in regime_diffs.items() if diff > _REGIME_SHIFT_THRESHOLD
        ]
        max_diff = float(max(regime_diffs.values())) if regime_diffs else 0.0
        shift_detected = len(shifted_regimes) > 0

        if shift_detected:
            logger.warning(
                "check_regime_distribution_shift: shift detected — "
                "max_diff=%.3f shifted_regimes=%s",
                max_diff,
                shifted_regimes,
            )
        else:
            logger.info(
                "check_regime_distribution_shift: stable — max_diff=%.3f",
                max_diff,
            )

        return {
            "shift_detected": shift_detected,
            "regime_diffs": regime_diffs,
            "max_diff": max_diff,
            "shifted_regimes": shifted_regimes,
        }

    # ------------------------------------------------------------------
    # Retrain decision
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        psi_results: dict,
        performance_drift: dict,
        regime_shift: dict,
    ) -> tuple[bool, str]:
        """Decide whether model retraining is needed based on all drift signals.

        Retrain is triggered when ANY of the following conditions hold:
        1. Any feature PSI > 0.25 (critical drift).
        2. >= 3 features with PSI > 0.20 (widespread moderate drift).
        3. Regime shift detected *and* ``max_diff`` > 0.30.
        4. Any performance drift value indicates a CRITICAL alert (|drift| > 35%).

        Args:
            psi_results:       Output of :meth:`check_feature_drift`.
            performance_drift: Output of :meth:`~performance_monitor.PerformanceMonitor.compute_drift`.
            regime_shift:      Output of :meth:`check_regime_distribution_shift`.

        Returns:
            Tuple ``(retrain_needed: bool, reason: str)``.
            *reason* is an empty string when no retraining is needed.
        """
        reasons: list[str] = []

        # --- Condition 1: any critical PSI ---
        critical_features = [
            f for f, psi in psi_results.items() if psi > _PSI_SIGNIFICANT
        ]
        if critical_features:
            reasons.append(
                f"critical PSI (>{_PSI_SIGNIFICANT}) on features: "
                + ", ".join(critical_features)
            )

        # --- Condition 2: widespread moderate drift ---
        moderate_features = [
            f for f, psi in psi_results.items() if psi > _PSI_SLIGHT
        ]
        if len(moderate_features) >= _MIN_PSI_RETRAIN_FEATURES:
            reasons.append(
                f"{len(moderate_features)} features with PSI > {_PSI_SLIGHT}: "
                + ", ".join(moderate_features)
            )

        # --- Condition 3: severe regime shift ---
        if regime_shift.get("shift_detected") and regime_shift.get("max_diff", 0.0) > _REGIME_SHIFT_MAX_DIFF_RETRAIN:
            reasons.append(
                f"regime distribution shift detected — "
                f"max_diff={regime_shift['max_diff']:.3f} "
                f"(>{_REGIME_SHIFT_MAX_DIFF_RETRAIN}), "
                f"shifted: {regime_shift.get('shifted_regimes', [])}"
            )

        # --- Condition 4: CRITICAL performance degradation ---
        # A drift value indicates a CRITICAL alert when its abs value > 35%.
        # For max_drawdown, positive drift is bad (drawdown grew).
        critical_perf_metrics: list[str] = []
        for metric, drift_pct in performance_drift.items():
            if metric == "total_trades":
                continue
            is_bad = (drift_pct > 35.0) if metric == "max_drawdown" else (drift_pct < -35.0)
            if is_bad:
                critical_perf_metrics.append(f"{metric}={drift_pct:+.1f}%")
        if critical_perf_metrics:
            reasons.append(
                "CRITICAL performance drift: " + ", ".join(critical_perf_metrics)
            )

        if reasons:
            full_reason = "; ".join(reasons)
            logger.warning(
                "DriftDetector.should_retrain: RETRAIN NEEDED — %s", full_reason
            )
            return True, full_reason

        logger.info("DriftDetector.should_retrain: no retraining needed")
        return False, ""
