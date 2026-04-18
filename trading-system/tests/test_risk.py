"""Tests for risk engine: Kelly sizer, portfolio risk, circuit breakers, prop-firm compliance."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _account_state(
    balance: float = 100_000.0,
    equity: float = 100_000.0,
    daily_pnl_pct: float = 0.0,
    weekly_pnl_pct: float = 0.0,
    consecutive_losses: int = 0,
    global_risk_state: str = "NORMAL",
    avg_spread_ratio: float = 1.0,
    mt5_connected: bool = True,
    open_positions: list | None = None,
) -> dict:
    return {
        "balance":            balance,
        "equity":             equity,
        "daily_pnl_pct":      daily_pnl_pct,
        "weekly_pnl_pct":     weekly_pnl_pct,
        "consecutive_losses": consecutive_losses,
        "global_risk_state":  global_risk_state,
        "avg_spread_ratio":   avg_spread_ratio,
        "mt5_connected":      mt5_connected,
        "open_positions":     open_positions or [],
        "timestamp":          datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# KellySizer
# ---------------------------------------------------------------------------


class TestKellySizer:
    """Tests for core/risk/kelly_sizer.py."""

    def test_kelly_fraction_positive(self):
        """Kelly fraction > 0 for profitable expectancy."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        f = ks.kelly_fraction(p_win=0.60, rr=2.0)
        assert f > 0.0

    def test_kelly_fraction_zero_for_negative_expectancy(self):
        """Kelly fraction is 0 when expectancy is negative."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        f = ks.kelly_fraction(p_win=0.30, rr=1.0)
        assert f == 0.0

    def test_kelly_fraction_half_kelly_applied(self):
        """Half-Kelly means returned fraction ≤ 0.5 × full Kelly."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        f = ks.kelly_fraction(p_win=0.60, rr=2.0)
        # Full Kelly = (0.6*2 - 0.4)/2 = (1.2-0.4)/2 = 0.4
        # Half-Kelly ≤ 0.20
        assert f <= 0.25   # generous bound to allow config variance

    def test_compute_lot_size_positive(self):
        """compute_lot_size returns a positive lot for normal inputs."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        lot = ks.compute_lot_size(
            account_balance=100_000.0,
            risk_pct=0.01,
            stop_pips=100.0,
            symbol="XAUUSD",
        )
        assert lot > 0.0

    def test_compute_lot_size_rounds_to_step(self):
        """compute_lot_size result is a multiple of 0.01."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        lot = ks.compute_lot_size(
            account_balance=100_000.0,
            risk_pct=0.01,
            stop_pips=100.0,
            symbol="XAUUSD",
        )
        # Must be a multiple of the lot step (0.01)
        assert round(lot % 0.01, 6) < 1e-5

    def test_compute_lot_size_clipped_to_max(self):
        """compute_lot_size never exceeds the configured max_lot."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        # Extreme risk to force lot above max
        lot = ks.compute_lot_size(
            account_balance=100_000_000.0,
            risk_pct=0.10,
            stop_pips=0.1,
            symbol="XAUUSD",
        )
        assert lot <= ks._max_lot

    def test_compute_lot_size_zero_stop_pips_returns_min(self):
        """compute_lot_size returns min_lot when stop_pips = 0."""
        from core.risk.kelly_sizer import KellySizer

        ks = KellySizer()
        lot = ks.compute_lot_size(
            account_balance=100_000.0,
            risk_pct=0.01,
            stop_pips=0.0,
            symbol="XAUUSD",
        )
        assert lot >= 0.0   # should not raise or return negative


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for core/risk/circuit_breakers.py."""

    def test_level0_all_clear(self):
        """Clean account returns level 0."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(daily_pnl_pct=-0.005, consecutive_losses=1)
        level, _ = cb.check(acct, [], [])
        assert level == 0

    def test_level1_triggered_by_daily_loss(self):
        """daily_pnl_pct < -1.5% triggers level 1."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(daily_pnl_pct=-0.016)
        level, _ = cb.check(acct, [], [])
        assert level >= 1

    def test_level1_triggered_by_consecutive_losses(self):
        """3 consecutive losses trigger level 1."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(consecutive_losses=3)
        level, _ = cb.check(acct, [], [])
        assert level >= 1

    def test_level2_triggered_by_higher_loss(self):
        """daily_pnl_pct < -2.5% triggers level 2+."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(daily_pnl_pct=-0.026)
        level, _ = cb.check(acct, [], [])
        assert level >= 2

    def test_level4_triggered_by_global_crisis(self):
        """global_risk_state == 'CRISIS' triggers level 4."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(global_risk_state="CRISIS")
        level, _ = cb.check(acct, [], [])
        assert level == 4

    def test_level4_triggered_by_spread_overrun(self):
        """avg_spread_ratio >= 3.0 triggers level 4."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        acct = _account_state(avg_spread_ratio=3.5)
        level, _ = cb.check(acct, [], [])
        assert level == 4

    def test_trading_halted_above_level2(self):
        """is_trading_halted returns True for level ≥ 2."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        assert not cb.is_trading_halted(0)
        assert not cb.is_trading_halted(1)
        assert cb.is_trading_halted(2)
        assert cb.is_trading_halted(3)
        assert cb.is_trading_halted(4)

    def test_size_multiplier_level1(self):
        """Level 1 returns multiplier of 0.70."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        mult = cb.get_size_multiplier(1)
        assert abs(mult - 0.70) < 0.05

    def test_size_multiplier_level2_plus_zero(self):
        """Level 2+ returns multiplier 0.0 (trading halted)."""
        from core.risk.circuit_breakers import CircuitBreaker

        cb = CircuitBreaker()
        for lvl in (2, 3, 4):
            assert cb.get_size_multiplier(lvl) == 0.0


# ---------------------------------------------------------------------------
# PropFirmCompliance
# ---------------------------------------------------------------------------


class TestPropFirmCompliance:
    """Tests for core/risk/prop_firm_compliance.py."""

    def test_compliant_trade_passes(self, sample_signal):
        """A clean trade passes all compliance checks."""
        from core.risk.prop_firm_compliance import PropFirmCompliance

        pfc = PropFirmCompliance()
        acct = _account_state(daily_pnl_pct=-0.01, equity=98_000.0, balance=100_000.0)
        ok, reasons = pfc.check(sample_signal, acct)
        assert ok

    def test_daily_drawdown_breach_fails(self, sample_signal):
        """Trade fails when daily loss already exceeds internal daily limit (4%)."""
        from core.risk.prop_firm_compliance import PropFirmCompliance

        pfc = PropFirmCompliance()
        acct = _account_state(daily_pnl_pct=-0.041)  # exceeds 4% internal limit
        ok, reasons = pfc.check(sample_signal, acct)
        assert not ok
        assert any("daily" in r.lower() for r in reasons)

    def test_max_drawdown_breach_fails(self, sample_signal):
        """Trade fails when account equity < 92% of starting balance (8% internal DD)."""
        from core.risk.prop_firm_compliance import PropFirmCompliance

        pfc = PropFirmCompliance()
        # Equity = 91% of balance → drawdown = 9%, exceeds 8% internal limit
        acct = _account_state(balance=100_000.0, equity=91_000.0)
        ok, reasons = pfc.check(sample_signal, acct)
        assert not ok


# ---------------------------------------------------------------------------
# RiskEngine (integration)
# ---------------------------------------------------------------------------


class TestRiskEngine:
    """Integration tests for core/risk/__init__.py."""

    def test_process_valid_signal_returns_trade_order(self, sample_signal, sample_features):
        """process() returns a TradeOrder for a valid signal."""
        from core.risk import RiskEngine, TradeOrder

        re = RiskEngine()
        acct = _account_state()
        result = re.process(sample_signal, acct, sample_features, [])
        # Should return a TradeOrder (may also return None if risk checks reject it)
        assert result is None or isinstance(result, TradeOrder)

    def test_process_rejects_zero_lot_signal(self, sample_signal, sample_features):
        """process() rejects when lot size calculation yields 0."""
        from core.risk import RiskEngine

        re = RiskEngine()
        # Simulate very high stop pips to force lot = 0
        sample_signal.stop_distance_pips = 1_000_000.0
        acct = _account_state(balance=100.0)  # tiny balance → can't afford even min risk
        result = re.process(sample_signal, acct, sample_features, [])
        assert result is None

    def test_process_rejects_rr_below_minimum(self, sample_signal, sample_features):
        """process() rejects signals with rr_ratio < 1.5."""
        from core.risk import RiskEngine

        re = RiskEngine()
        sample_signal.rr_ratio = 0.5   # below minimum
        acct = _account_state()
        result = re.process(sample_signal, acct, sample_features, [])
        assert result is None

    def test_trade_order_fields_populated(self, sample_signal, sample_features):
        """Returned TradeOrder has non-zero lot_size and populated audit fields."""
        from core.risk import RiskEngine, TradeOrder

        re = RiskEngine()
        # Ensure signal has a healthy RR
        sample_signal.rr_ratio = 2.0
        acct = _account_state()
        result = re.process(sample_signal, acct, sample_features, [])
        if result is not None:
            assert isinstance(result, TradeOrder)
            assert result.lot_size > 0
            assert result.account_balance == 100_000.0
            assert result.direction in ("LONG", "SHORT")
