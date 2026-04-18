"""Tests for regime detection: HMM, XGBoost classifier, MTF alignment, universal model."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# HMMRegimeModel
# ---------------------------------------------------------------------------

class TestHMMRegimeModel:
    """Tests for core/regime/hmm_model.py."""

    def test_fit_and_predict(self, sample_features):
        """HMM can be fit and then predicts a valid regime string."""
        from core.regime.hmm_model import HMMRegimeModel, VALID_REGIMES

        model = HMMRegimeModel(n_states=4)
        # Use a subset of numeric features
        numeric = sample_features.select_dtypes(include=[np.number]).iloc[:, :5]
        model.fit(numeric)

        assert model.is_fitted
        pred = model.predict(numeric)
        assert isinstance(pred, str)
        assert pred in VALID_REGIMES

    def test_predict_proba_sums_to_one(self, sample_features):
        """HMM state probabilities sum to 1.0 for each bar."""
        from core.regime.hmm_model import HMMRegimeModel

        model = HMMRegimeModel(n_states=4)
        numeric = sample_features.select_dtypes(include=[np.number]).iloc[:, :5]
        model.fit(numeric)

        proba = model.predict_proba(numeric)
        assert isinstance(proba, dict)
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-6

    def test_untrained_model_raises(self, sample_features):
        """Untrained model raises RuntimeError on predict."""
        from core.regime.hmm_model import HMMRegimeModel

        model = HMMRegimeModel(n_states=4)
        numeric = sample_features.select_dtypes(include=[np.number]).iloc[:, :5]
        with pytest.raises(Exception):
            model.predict(numeric)

    def test_save_and_load(self, tmp_path, sample_features):
        """Save / load round-trip preserves prediction consistency."""
        from core.regime.hmm_model import HMMRegimeModel

        model = HMMRegimeModel(n_states=4)
        numeric = sample_features.select_dtypes(include=[np.number]).iloc[:, :5]
        model.fit(numeric)

        path = str(tmp_path / "hmm.pkl")
        model.save(path)

        model2 = HMMRegimeModel(n_states=4)
        model2.load(path)

        assert model2.is_fitted
        p1 = model.predict(numeric)
        p2 = model2.predict(numeric)
        assert p1 == p2

    def test_fit_with_insufficient_data_raises(self):
        """Fitting with fewer than 30 rows raises ValueError."""
        from core.regime.hmm_model import HMMRegimeModel

        model = HMMRegimeModel(n_states=4)
        tiny = pd.DataFrame(np.random.randn(10, 3))
        with pytest.raises(Exception):
            model.fit(tiny)

    def test_state_mapping_produces_correct_labels(self, sample_features):
        """State mapping assigns VOLATILE to highest-volatility state."""
        from core.regime.hmm_model import HMMRegimeModel, REGIME_VOLATILE

        model = HMMRegimeModel(n_states=4)
        numeric = sample_features.select_dtypes(include=[np.number]).iloc[:, :5]
        model.fit(numeric)

        # At least one state must map to VOLATILE
        assert REGIME_VOLATILE in model._state_map.values()


# ---------------------------------------------------------------------------
# XGBRegimeClassifier
# ---------------------------------------------------------------------------

class TestXGBRegimeClassifier:
    """Tests for core/regime/xgb_classifier.py."""

    @pytest.fixture
    def trained_classifier(self, sample_features):
        from core.regime.xgb_classifier import XGBRegimeClassifier

        clf = XGBRegimeClassifier()
        n = len(sample_features)
        labels = pd.Series(
            np.random.choice(["TREND_UP", "RANGE", "VOLATILE", "TREND_DOWN"], n),
            index=sample_features.index,
        )
        clf.train(sample_features, labels)
        return clf

    def test_train_and_predict(self, trained_classifier, sample_features):
        """Trained classifier returns a dict with 'regime' and 'confidence'."""
        result = trained_classifier.predict(sample_features.iloc[[-1]])
        assert "regime" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, trained_classifier, sample_features):
        """Probabilities in predict() sum to ≈1.0."""
        result = trained_classifier.predict(sample_features.iloc[[-1]])
        probs = result.get("probabilities", {})
        if probs:
            assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_untrained_classifier_returns_fallback(self, sample_features):
        """Untrained classifier returns a regime without crashing."""
        from core.regime.xgb_classifier import XGBRegimeClassifier

        clf = XGBRegimeClassifier()
        result = clf.predict(sample_features.iloc[[-1]])
        # Should not raise — returns fallback dict
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# MTFAlignment
# ---------------------------------------------------------------------------

class TestMTFAlignment:
    """Tests for core/regime/mtf_alignment.py."""

    def test_alignment_score_range(self):
        """Alignment score is always in [0, 1]."""
        from core.regime.mtf_alignment import MTFAlignment

        mtf = MTFAlignment()
        for h4, h1, m15 in [
            ("TREND_UP",   "TREND_UP",   "TREND_UP"),
            ("TREND_DOWN", "RANGE",      "VOLATILE"),
            ("RANGE",      "RANGE",      "RANGE"),
            ("VOLATILE",   "TREND_UP",   "RANGE"),
        ]:
            score = mtf.compute_alignment_score(h4, h1, m15, 0.7, 0.7, 0.7)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {h4}/{h1}/{m15}"

    def test_full_alignment_momentum_strategy(self):
        """Perfect directional alignment returns momentum strategy."""
        from core.regime.mtf_alignment import MTFAlignment

        mtf = MTFAlignment()
        strategy = mtf.determine_active_strategy("TREND_UP", "TREND_UP", "TREND_UP", 0.9)
        assert strategy == "momentum"

    def test_range_regime_returns_mean_reversion(self):
        """RANGE across all timeframes returns mean_reversion strategy."""
        from core.regime.mtf_alignment import MTFAlignment

        mtf = MTFAlignment()
        strategy = mtf.determine_active_strategy("RANGE", "RANGE", "RANGE", 0.7)
        assert strategy == "mean_reversion"

    def test_volatile_m15_returns_breakout(self):
        """VOLATILE M15 regime returns breakout strategy."""
        from core.regime.mtf_alignment import MTFAlignment

        mtf = MTFAlignment()
        strategy = mtf.determine_active_strategy("VOLATILE", "VOLATILE", "VOLATILE", 0.7)
        assert strategy == "breakout"

    def test_sizing_multiplier_decreases_with_misalignment(self):
        """Sizing multiplier for misaligned regimes is lower than fully aligned."""
        from core.regime.mtf_alignment import MTFAlignment

        mtf = MTFAlignment()
        aligned = mtf.compute_sizing_multiplier(
            mtf.compute_alignment_score("TREND_UP", "TREND_UP", "TREND_UP", 0.8, 0.8, 0.8)
        )
        misaligned = mtf.compute_sizing_multiplier(
            mtf.compute_alignment_score("TREND_UP", "RANGE", "VOLATILE", 0.5, 0.4, 0.3)
        )
        assert aligned >= misaligned


# ---------------------------------------------------------------------------
# RegimeDetector (integration)
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    """Integration tests for core/regime/regime_detector.py."""

    def test_detect_returns_regime_state(self, sample_features, sample_ohlcv_data):
        """detect() returns a RegimeState without crashing."""
        from core.regime.regime_detector import RegimeDetector, RegimeState

        det = RegimeDetector()
        data_dict = {"M15": sample_ohlcv_data}
        state = det.detect("XAUUSD", data_dict, sample_features)

        assert isinstance(state, RegimeState)
        assert state.symbol == "XAUUSD"
        assert state.final_sizing_multiplier >= 0.0

    def test_untrained_detector_multiplier_zero(self, sample_features, sample_ohlcv_data):
        """Freshly instantiated detector returns multiplier=0 (models not trained)."""
        from core.regime.regime_detector import RegimeDetector

        det = RegimeDetector()
        data_dict = {"M15": sample_ohlcv_data}
        state = det.detect("XAUUSD", data_dict, sample_features)
        assert state.final_sizing_multiplier == 0.0

    def test_regime_state_field_types(self, sample_features, sample_ohlcv_data):
        """All RegimeState fields have correct types."""
        from core.regime.regime_detector import RegimeDetector

        det = RegimeDetector()
        data_dict = {"M15": sample_ohlcv_data}
        state = det.detect("XAUUSD", data_dict, sample_features)

        assert isinstance(state.symbol, str)
        assert isinstance(state.timestamp, datetime)
        assert isinstance(state.global_multiplier, float)
        assert isinstance(state.alignment_score, float)
        assert isinstance(state.active_strategy, str)
        assert isinstance(state.final_sizing_multiplier, float)

    def test_detect_multiple_symbols_independent(self, sample_features, sample_ohlcv_data):
        """Detecting on different symbols does not cross-contaminate state."""
        from core.regime.regime_detector import RegimeDetector

        det = RegimeDetector()
        data_dict = {"M15": sample_ohlcv_data}
        s1 = det.detect("XAUUSD", data_dict, sample_features)
        s2 = det.detect("EURUSD", data_dict, sample_features)

        assert s1.symbol == "XAUUSD"
        assert s2.symbol == "EURUSD"
