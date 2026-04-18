"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# OHLCV / Feature data fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, base_price: float = 1800.0, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Args:
        n:          Number of bars.
        base_price: Starting close price.
        symbol:     Instrument symbol (unused in generation but useful for callers).

    Returns:
        DataFrame with columns: open, high, low, close, volume.
        Index is a DatetimeIndex at 15-minute intervals.
    """
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.001, n)
    close   = base_price * np.cumprod(1 + returns)
    noise   = rng.uniform(0.0005, 0.002, n)
    high    = close * (1 + noise)
    low     = close * (1 - noise)
    open_   = np.roll(close, 1)
    open_[0] = close[0]
    volume  = rng.integers(100, 5000, n).astype(float)

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """300 bars of synthetic M15 OHLCV for XAUUSD."""
    return _make_ohlcv(n=300, base_price=1800.0)


@pytest.fixture
def sample_ohlcv_data_large() -> pd.DataFrame:
    """1500 bars of synthetic M15 OHLCV (for training tests)."""
    return _make_ohlcv(n=1500, base_price=1800.0)


@pytest.fixture
def sample_features(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Feature DataFrame built from sample_ohlcv_data.

    Uses FeatureEngineer if available; falls back to minimal hand-crafted
    features so tests can run without the full dependency tree.
    """
    df = sample_ohlcv_data
    n  = len(df)
    rng = np.random.default_rng(0)

    feat = pd.DataFrame(index=df.index)

    # Price / basic
    feat["close"]       = df["close"]
    feat["high"]        = df["high"]
    feat["low"]         = df["low"]
    feat["volume"]      = df["volume"]

    # ATR (simple range proxy)
    feat["atr_14"]      = (df["high"] - df["low"]).rolling(14).mean().fillna(0.001)
    feat["atr_50_mean"] = (df["high"] - df["low"]).rolling(50).mean().fillna(0.001)

    # Trend
    feat["ema_20"]      = df["close"].ewm(span=20).mean()
    feat["ema_50"]      = df["close"].ewm(span=50).mean()
    feat["ema_200"]     = df["close"].ewm(span=200).mean()

    # Momentum
    feat["rsi_7"]       = 50.0 + rng.normal(0, 15, n)
    feat["rsi_14"]      = 50.0 + rng.normal(0, 10, n)
    feat["macd_hist"]   = rng.normal(0, 0.5, n)
    feat["macd_hist_slope"] = rng.normal(0, 0.1, n)

    # Mean reversion
    feat["z_score_50"]  = rng.normal(0, 1.5, n)
    feat["hurst_100"]   = rng.uniform(0.35, 0.65, n)
    feat["stoch_k"]     = rng.uniform(0, 100, n)
    feat["stoch_d"]     = rng.uniform(0, 100, n)
    feat["cci_14"]      = rng.normal(0, 80, n)
    feat["adx_14"]      = rng.uniform(10, 50, n)

    # Breakout
    feat["bb_width"]    = rng.uniform(0.001, 0.01, n)
    feat["atr_ratio"]   = rng.uniform(0.5, 1.5, n)
    feat["volume_ratio"] = rng.uniform(0.5, 3.0, n)

    # Misc
    feat["atr_50_mean"] = feat["atr_50_mean"].fillna(0.001)

    return feat.fillna(method="ffill").fillna(0.0)


# ---------------------------------------------------------------------------
# Regime state fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_regime_state():
    """A healthy RegimeState with multiplier=1.0 and TREND_UP strategy."""
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
        active_strategy="momentum",
        regime_confirmed=True,
        regime_age_candles=20,
        regime_maturity_flag="mature",
        regime_age_multiplier=1.0,
        final_sizing_multiplier=1.0,
        strategy_module="momentum",
    )


# ---------------------------------------------------------------------------
# Signal fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_signal(sample_regime_state):
    """A valid LONG signal from the Momentum module."""
    from core.signals.signal_router import SignalOutput

    return SignalOutput(
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
        regime_context=sample_regime_state,
        sizing_inputs={
            "final_sizing_multiplier": 1.0,
            "global_multiplier": 1.0,
            "alignment_multiplier": 1.0,
            "age_multiplier": 1.0,
            "confidence": 0.75,
            "regime": "TREND_UP",
            "global_risk_state": "NORMAL",
        },
        magic_number=12345678,
    )


# ---------------------------------------------------------------------------
# TradeOrder fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trade_order(sample_signal):
    """A TradeOrder built from sample_signal."""
    from core.risk import TradeOrder

    return TradeOrder(
        signal_id=1,
        magic_number=sample_signal.magic_number,
        symbol=sample_signal.asset,
        timeframe=sample_signal.timeframe,
        direction=sample_signal.signal,
        module=sample_signal.module,
        lot_size=0.10,
        entry_price=sample_signal.entry_price,
        stop_loss=sample_signal.stop_loss,
        take_profit_1=sample_signal.take_profit_1,
        take_profit_2=sample_signal.take_profit_2,
        atr=sample_signal.atr,
        stop_distance_pips=sample_signal.stop_distance_pips,
        rr_ratio=sample_signal.rr_ratio,
        account_balance=100_000.0,
        base_risk_pct=0.01,
        kelly_multiplier=0.30,
        volatility_scalar=1.0,
        regime_age_multiplier=1.0,
        alignment_multiplier=1.0,
        correlation_multiplier=1.0,
        global_multiplier=1.0,
        final_risk_pct=0.01,
        risk_amount_currency=1_000.0,
        regime_context=sample_signal.regime_context,
    )


# ---------------------------------------------------------------------------
# Mock MT5 client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mt5_client():
    """MagicMock that mimics MT5Client's public API."""
    client = MagicMock()
    client.health_check.return_value = {
        "mt5_connected": True,
        "account_server": "Demo-Server",
    }
    client.account_info.return_value = {
        "balance": 100_000.0,
        "equity":  100_000.0,
        "margin":  0.0,
        "free_margin": 100_000.0,
    }
    client.symbol_info_tick.return_value = {
        "bid": 1809.50,
        "ask": 1810.50,
        "time": int(datetime.now(timezone.utc).timestamp()),
    }
    client.symbol_info.return_value = {
        "symbol": "XAUUSD",
        "digits": 2,
        "point": 0.01,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "trade_contract_size": 100.0,
    }
    client.positions_get.return_value = []
    client.orders_get.return_value = []
    return client


# ---------------------------------------------------------------------------
# Test config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config():
    """Minimal config namespace for tests that need config access."""
    from types import SimpleNamespace

    return SimpleNamespace(
        mode=False,
        assets=[
            SimpleNamespace(symbol="XAUUSD", hmm_states=5),
            SimpleNamespace(symbol="EURUSD", hmm_states=4),
        ],
        risk=SimpleNamespace(
            base_risk_per_trade=0.01,
            max_risk_per_trade=0.015,
            max_portfolio_heat=0.06,
            max_correlated_risk=0.03,
            kelly_fraction=0.50,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=10.0,
        ),
        sizing=SimpleNamespace(
            base_risk_per_trade=0.01,
            max_risk_per_trade=0.015,
            kelly_fraction=0.50,
        ),
    )
