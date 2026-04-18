"""Tests for signal engine: momentum, mean-reversion, breakout modules, and router."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a features DataFrame with deliberately extreme values so
# the signal conditions trigger without needing real market data.
# ---------------------------------------------------------------------------


def _make_extreme_features(n: int = 100, direction: str = "LONG") -> pd.DataFrame:
    """Return a small feature DataFrame designed to trigger signals."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base_price = 1800.0

    feat = pd.DataFrame(index=idx)
    feat["close"]       = base_price + rng.normal(0, 0.5, n)
    feat["high"]        = feat["close"] + rng.uniform(0.5, 1.5, n)
    feat["low"]         = feat["close"] - rng.uniform(0.5, 1.5, n)
    feat["volume"]      = rng.integers(200, 2000, n).astype(float)
    feat["atr_14"]      = np.full(n, 5.0)
    feat["atr_50_mean"] = np.full(n, 5.0)

    if direction == "LONG":
        feat["z_score_50"]   = np.full(n, -2.5)   # oversold
        feat["rsi_7"]        = np.full(n, 20.0)    # oversold
        feat["stoch_k"]      = np.full(n, 10.0)    # oversold
        feat["cci_14"]       = np.full(n, -200.0)  # oversold
        feat["adx_14"]       = np.full(n, 15.0)    # weak trend
        feat["macd_hist"]    = np.full(n, 0.1)
        feat["macd_hist_slope"] = np.full(n, 0.05)
    else:
        feat["z_score_50"]   = np.full(n, 2.5)    # overbought
        feat["rsi_7"]        = np.full(n, 80.0)
        feat["stoch_k"]      = np.full(n, 90.0)
        feat["cci_14"]       = np.full(n, 200.0)
        feat["adx_14"]       = np.full(n, 15.0)
        feat["macd_hist"]    = np.full(n, -0.1)
        feat["macd_hist_slope"] = np.full(n, -0.05)

    feat["hurst_100"]   = np.full(n, 0.35)        # strong mean-reverting
    feat["bb_width"]    = np.full(n, 0.002)
    feat["atr_ratio"]   = np.full(n, 0.60)        # compression
    feat["volume_ratio"] = np.full(n, 2.0)        # elevated vol for breakout
    feat["ema_20"]      = feat["close"]
    feat["ema_50"]      = feat["close"]
    feat["ema_200"]     = feat["close"]

    return feat.fillna(0.0)


def _make_regime_state(multiplier: float = 1.0, strategy: str = "momentum"):
    """Build a minimal RegimeState namespace."""
    from core.regime.regime_detector import RegimeState

    return RegimeState(
        symbol="XAUUSD",
        timestamp=datetime.now(timezone.utc),
        global_risk_state="NORMAL",
        global_multiplier=1.0,
        h4_regime="TREND_UP",
        h4_confidence=0.80,
        h1_regime="TREND_UP",
        h1_confidence=0.75,
        m15_regime="TREND_UP",
        m15_confidence=0.70,
        alignment_score=0.85,
        alignment_sizing_multiplier=1.0,
        active_strategy=strategy,
        regime_confirmed=True,
        regime_age_candles=20,
        regime_maturity_flag="mature",
        regime_age_multiplier=1.0,
        final_sizing_multiplier=multiplier,
        strategy_module=strategy,
    )


# ---------------------------------------------------------------------------
# MomentumModule
# ---------------------------------------------------------------------------


class TestMomentumModule:
    """Tests for core/signals/momentum_module.py."""

    def test_generate_labels_length(self, sample_ohlcv_data, sample_features):
        from core.signals.momentum_module import MomentumModule

        mod = MomentumModule("XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data, sample_features)
        assert len(labels) == len(sample_ohlcv_data)

    def test_generate_labels_valid_values(self, sample_ohlcv_data, sample_features):
        """Labels are 0, 1, or 2 only."""
        from core.signals.momentum_module import MomentumModule, LABEL_SHORT, LABEL_NOTRADE, LABEL_LONG

        mod = MomentumModule("XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data, sample_features)
        assert set(labels.unique()).issubset({LABEL_SHORT, LABEL_NOTRADE, LABEL_LONG})

    def test_last_n_bars_forced_notrade(self, sample_ohlcv_data, sample_features):
        """Last label_forward_candles bars are always NOTRADE (1)."""
        from core.signals.momentum_module import MomentumModule, LABEL_NOTRADE

        mod = MomentumModule("XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data, sample_features)
        n_fwd = mod._label_fwd_m15
        assert all(labels.values[-n_fwd:] == LABEL_NOTRADE)

    def test_train_sets_trained_flag(self, sample_ohlcv_data_large):
        """train() marks the model as trained."""
        from core.signals.momentum_module import MomentumModule
        from core.data.feature_engineer import FeatureEngineer

        mod = MomentumModule("XAUUSD")
        fe  = FeatureEngineer()
        features = fe.compute(sample_ohlcv_data_large, symbol="XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data_large, features)
        mod.train(features, labels)
        assert mod._xgb_trained

    def test_predict_returns_none_when_untrained(self, sample_features):
        """predict() returns None when model is not yet trained."""
        from core.signals.momentum_module import MomentumModule

        mod = MomentumModule("XAUUSD")
        rs  = _make_regime_state(1.0, "momentum")
        latest_bar = {"bid": 1810.0, "ask": 1810.5, "atr": 5.0,
                      "high": 1811.0, "low": 1809.0, "close": 1810.0, "pip_size": 0.01}
        result = mod.predict(sample_features, rs, latest_bar)
        assert result is None

    def test_predict_after_train_returns_signal_or_none(self, sample_ohlcv_data_large):
        """After training, predict() returns a SignalOutput or None (not an error)."""
        from core.signals.momentum_module import MomentumModule
        from core.signals.signal_router import SignalOutput
        from core.data.feature_engineer import FeatureEngineer

        mod = MomentumModule("XAUUSD")
        fe  = FeatureEngineer()
        features = fe.compute(sample_ohlcv_data_large, symbol="XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data_large, features)
        mod.train(features, labels)

        rs = _make_regime_state(1.0, "momentum")
        latest_bar = {"bid": 1810.0, "ask": 1810.5, "atr": 5.0,
                      "high": 1811.0, "low": 1809.0, "close": 1810.0, "pip_size": 0.01}
        result = mod.predict(features, rs, latest_bar)
        assert result is None or isinstance(result, SignalOutput)

    def test_save_load_roundtrip(self, tmp_path, sample_ohlcv_data_large):
        """Save / load preserves the trained state."""
        from core.signals.momentum_module import MomentumModule
        from core.data.feature_engineer import FeatureEngineer

        mod = MomentumModule("XAUUSD")
        fe  = FeatureEngineer()
        features = fe.compute(sample_ohlcv_data_large, symbol="XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data_large, features)
        mod.train(features, labels)

        path = str(tmp_path / "mom.pkl")
        mod.save_model(path)

        mod2 = MomentumModule("XAUUSD")
        mod2.load_model(path)
        assert mod2._xgb_trained
        assert set(mod2._feature_names) == set(mod._feature_names)


# ---------------------------------------------------------------------------
# MeanReversionModule
# ---------------------------------------------------------------------------


class TestMeanReversionModule:
    """Tests for core/signals/mean_reversion_module.py."""

    def test_gate1_zscore_triggers_long(self):
        """Gate 1 passes for a strongly negative z-score (LONG)."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        row = pd.Series({"z_score_50": -2.5, "hurst_100": 0.5})
        passes, direction, z = mod._check_gate1_zscore(row)
        assert passes
        assert direction == "LONG"

    def test_gate1_zscore_triggers_short(self):
        """Gate 1 passes for a strongly positive z-score (SHORT)."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        row = pd.Series({"z_score_50": 2.5, "hurst_100": 0.5})
        passes, direction, _ = mod._check_gate1_zscore(row)
        assert passes
        assert direction == "SHORT"

    def test_gate1_zscore_neutral_fails(self):
        """Gate 1 does not pass for neutral z-score."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        row = pd.Series({"z_score_50": 0.5, "hurst_100": 0.5})
        passes, _, _ = mod._check_gate1_zscore(row)
        assert not passes

    def test_gate1_hurst_adjustment_relaxes_threshold(self):
        """Strong mean-reversion (H < 0.40) triggers at lower z-score."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        # Without Hurst adjustment, -1.8 would NOT pass (threshold = -2.0)
        # With H=0.35 < 0.40, threshold becomes -1.70 → should pass
        row_strong = pd.Series({"z_score_50": -1.75, "hurst_100": 0.35})
        row_normal = pd.Series({"z_score_50": -1.75, "hurst_100": 0.50})
        p_strong, _, _ = mod._check_gate1_zscore(row_strong)
        p_normal, _, _ = mod._check_gate1_zscore(row_normal)
        assert p_strong
        assert not p_normal

    def test_gate2_oscillators_requires_min_agreement(self):
        """Gate 2 requires at least 2 oscillators to agree."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        # Only 1 oversold → should fail
        row_one = pd.Series({"rsi_7": 20.0, "stoch_k": 50.0, "cci_14": 0.0})
        # 2 oversold → should pass
        row_two = pd.Series({"rsi_7": 20.0, "stoch_k": 15.0, "cci_14": 0.0})
        assert not mod._check_gate2_oscillators(row_one, "LONG")
        assert mod._check_gate2_oscillators(row_two, "LONG")

    def test_gate3_adx_rejects_strong_trend(self):
        """Gate 3 rejects when ADX >= 30."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        row = pd.Series({"adx_14": 35.0, "macd_hist": 0.0, "macd_hist_slope": 0.0})
        assert not mod._check_gate3_adx(row, "LONG")

    def test_gate3_adx_passes_weak_trend(self):
        """Gate 3 passes when ADX < 30 and MACD not expanding against direction."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("EURUSD")
        row = pd.Series({"adx_14": 20.0, "macd_hist": 0.1, "macd_hist_slope": 0.05})
        assert mod._check_gate3_adx(row, "LONG")

    def test_generate_labels_binary(self, sample_ohlcv_data, sample_features):
        """generate_labels returns only 0 and 1."""
        from core.signals.mean_reversion_module import MeanReversionModule

        mod = MeanReversionModule("XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data, sample_features)
        assert set(labels.unique()).issubset({0, 1})

    def test_train_and_gate4_returns_probability(self, sample_ohlcv_data_large):
        """After training, Gate 4 returns a probability in [0, 1]."""
        from core.signals.mean_reversion_module import MeanReversionModule
        from core.data.feature_engineer import FeatureEngineer

        mod = MeanReversionModule("XAUUSD")
        fe  = FeatureEngineer()
        features = fe.compute(sample_ohlcv_data_large, symbol="XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data_large, features)
        mod.train(features, labels)

        row = features.iloc[-1]
        p = mod._check_gate4_ml(row)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# BreakoutModule
# ---------------------------------------------------------------------------


class TestBreakoutModule:
    """Tests for core/signals/breakout_module.py."""

    def test_detect_compression_false_when_no_squeeze(self, sample_features):
        """Compression detection returns False with normal volatility."""
        from core.signals.breakout_module import BreakoutModule

        mod = MomentumModule = BreakoutModule("XAUUSD")
        # Features with atr_ratio = 1.5 (above threshold) → no compression
        feat = sample_features.copy()
        feat["atr_ratio"] = 1.5
        result = mod._detect_compression(feat, n=20)
        assert not result["compressed"]

    def test_detect_compression_true_when_squeezed(self):
        """Compression detection fires when all three conditions met."""
        from core.signals.breakout_module import BreakoutModule

        mod = BreakoutModule("XAUUSD")
        n = 50
        rng = np.random.default_rng(3)
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")

        feat = pd.DataFrame(index=idx)
        feat["close"]       = 1800.0 + rng.normal(0, 0.1, n)
        feat["high"]        = feat["close"] + 0.3
        feat["low"]         = feat["close"] - 0.3
        feat["atr_14"]      = np.full(n, 1.0)
        feat["atr_50_mean"] = np.full(n, 5.0)   # much wider baseline → small range_ratio
        feat["atr_ratio"]   = np.full(n, 0.50)  # below 0.80 threshold
        feat["bb_width"]    = np.full(n, 0.001) # small

        result = mod._detect_compression(feat, n=20)
        assert result["compressed"]

    def test_detect_break_long(self):
        """Break detection fires LONG when close breaks above compression_high."""
        from core.signals.breakout_module import BreakoutModule

        mod = BreakoutModule("XAUUSD")
        compression = {
            "compressed": True,
            "compression_high": 1810.0,
            "compression_low":  1800.0,
        }
        row = pd.Series({
            "close": 1813.0,  # > 1810 + 0.30*5 = 1811.5
            "atr_14": 5.0,
            "volume_ratio": 2.0,  # above 1.50 threshold
        })
        result = mod._detect_break(row, compression)
        assert result["break_detected"]
        assert result["direction"] == "LONG"

    def test_detect_break_short(self):
        """Break detection fires SHORT when close breaks below compression_low."""
        from core.signals.breakout_module import BreakoutModule

        mod = BreakoutModule("XAUUSD")
        compression = {
            "compressed": True,
            "compression_high": 1810.0,
            "compression_low":  1800.0,
        }
        row = pd.Series({
            "close": 1797.0,  # < 1800 - 0.30*5 = 1798.5
            "atr_14": 5.0,
            "volume_ratio": 2.0,
        })
        result = mod._detect_break(row, compression)
        assert result["break_detected"]
        assert result["direction"] == "SHORT"

    def test_generate_labels_binary(self, sample_ohlcv_data, sample_features):
        """generate_labels returns only 0 and 1."""
        from core.signals.breakout_module import BreakoutModule

        mod = BreakoutModule("XAUUSD")
        labels = mod.generate_labels(sample_ohlcv_data, sample_features)
        assert set(labels.unique()).issubset({0, 1})

    def test_predict_insufficient_data_returns_none(self):
        """predict() returns None when fewer than 25 bars provided."""
        from core.signals.breakout_module import BreakoutModule

        mod = BreakoutModule("XAUUSD")
        tiny_feat = pd.DataFrame({"close": [1800.0] * 10, "atr_14": [5.0] * 10})
        rs = _make_regime_state(1.0, "breakout")
        lb = {"bid": 1800.0, "ask": 1800.5, "atr": 5.0, "close": 1800.0, "pip_size": 0.01}
        assert mod.predict(tiny_feat, rs, lb) is None


# ---------------------------------------------------------------------------
# SignalRouter
# ---------------------------------------------------------------------------


class TestSignalRouter:
    """Tests for core/signals/signal_router.py."""

    def _make_router_with_mocks(self):
        from core.signals.signal_router import SignalRouter, SignalOutput

        # Momentum module that always returns a LONG signal
        good_signal = SignalOutput(
            asset="XAUUSD",
            timeframe="M15",
            timestamp=datetime.now(timezone.utc),
            signal="LONG",
            module="MOMENTUM",
            confidence=0.75,
            entry_price=1810.0,
            stop_loss=1800.0,
            take_profit_1=1820.0,
            take_profit_2=1830.0,
            atr=5.0,
            stop_distance_pips=1000.0,
            rr_ratio=2.0,
        )

        mom  = MagicMock()
        mr   = MagicMock()
        bo   = MagicMock()
        mom.predict.return_value = good_signal
        mr.predict.return_value  = None
        bo.predict.return_value  = None

        return SignalRouter(mom, mr, bo), good_signal

    def test_routes_to_momentum_when_strategy_is_momentum(self, sample_features):
        """Router calls momentum module when active_strategy == 'momentum'."""
        from core.signals.signal_router import SignalOutput

        router, expected_signal = self._make_router_with_mocks()
        rs = _make_regime_state(1.0, "momentum")
        lb = {"bid": 1809.5, "ask": 1810.5, "atr": 5.0, "close": 1810.0,
              "high": 1811.0, "low": 1809.0, "pip_size": 0.01}

        result = router.route("XAUUSD", "M15", sample_features, rs, lb)
        assert result is not None
        assert isinstance(result, SignalOutput)
        assert result.signal == "LONG"

    def test_returns_none_when_multiplier_zero(self, sample_features):
        """Router gates on final_sizing_multiplier == 0."""
        router, _ = self._make_router_with_mocks()
        rs = _make_regime_state(0.0, "momentum")
        lb = {"bid": 1809.5, "ask": 1810.5, "atr": 5.0, "close": 1810.0,
              "high": 1811.0, "low": 1809.0, "pip_size": 0.01}
        result = router.route("XAUUSD", "M15", sample_features, rs, lb)
        assert result is None

    def test_validation_rejects_invalid_stop(self, sample_features):
        """Router rejects signal where stop_loss >= entry for LONG."""
        from core.signals.signal_router import SignalRouter, SignalOutput

        bad_signal = SignalOutput(
            asset="XAUUSD",
            timeframe="M15",
            timestamp=datetime.now(timezone.utc),
            signal="LONG",
            module="MOMENTUM",
            confidence=0.75,
            entry_price=1810.0,
            stop_loss=1815.0,   # WRONG: stop above entry for LONG
            take_profit_1=1820.0,
            take_profit_2=1830.0,
            atr=5.0,
            stop_distance_pips=500.0,
            rr_ratio=2.0,
        )
        mom = MagicMock()
        mom.predict.return_value = bad_signal
        router = SignalRouter(mom, MagicMock(), MagicMock())
        rs = _make_regime_state(1.0, "momentum")
        lb = {"bid": 1809.5, "ask": 1810.5, "atr": 5.0, "close": 1810.0,
              "high": 1811.0, "low": 1809.0, "pip_size": 0.01}
        result = router.route("XAUUSD", "M15", sample_features, rs, lb)
        assert result is None

    def test_magic_number_in_range(self, sample_features):
        """Magic number returned by router is 8 digits (10M–99M)."""
        router, _ = self._make_router_with_mocks()
        rs = _make_regime_state(1.0, "momentum")
        lb = {"bid": 1809.5, "ask": 1810.5, "atr": 5.0, "close": 1810.0,
              "high": 1811.0, "low": 1809.0, "pip_size": 0.01}
        result = router.route("XAUUSD", "M15", sample_features, rs, lb)
        assert result is not None
        assert 10_000_000 <= result.magic_number <= 99_999_999

    def test_routes_to_mean_reversion_strategy(self, sample_features):
        """Router calls MR module when active_strategy == 'mean_reversion'."""
        from core.signals.signal_router import SignalRouter, SignalOutput

        good_signal = SignalOutput(
            asset="XAUUSD",
            timeframe="M15",
            timestamp=datetime.now(timezone.utc),
            signal="SHORT",
            module="MEAN_REVERSION",
            confidence=0.65,
            entry_price=1810.0,
            stop_loss=1820.0,
            take_profit_1=1805.0,
            take_profit_2=1800.0,
            atr=5.0,
            stop_distance_pips=1000.0,
            rr_ratio=2.0,
        )
        mom = MagicMock()
        mr  = MagicMock()
        bo  = MagicMock()
        mr.predict.return_value  = good_signal
        mom.predict.return_value = None
        bo.predict.return_value  = None

        router = SignalRouter(mom, mr, bo)
        rs = _make_regime_state(1.0, "mean_reversion")
        lb = {"bid": 1809.5, "ask": 1810.5, "atr": 5.0, "close": 1810.0,
              "high": 1811.0, "low": 1809.0, "pip_size": 0.01}
        result = router.route("XAUUSD", "M15", sample_features, rs, lb)
        assert result is not None
        assert result.module == "MEAN_REVERSION"
