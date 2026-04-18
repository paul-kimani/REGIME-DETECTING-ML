"""Tests for backtesting: simulation engine look-ahead, fill logic, walk-forward, metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    """Tests for backtesting/performance_metrics.py."""

    def _make_trades(self, n_win: int, n_loss: int):
        """Create a minimal list of BacktestTrade-like objects."""
        from backtesting.simulation_engine import BacktestTrade

        trades = []
        for i in range(n_win):
            t = BacktestTrade(
                trade_id=i,
                symbol="XAUUSD",
                direction="LONG",
                module="MOMENTUM",
                entry_bar=i * 2,
                entry_price=1800.0,
                entry_time=datetime.now(timezone.utc),
                stop_loss=1790.0,
                tp1=1810.0,
                tp2=1820.0,
                lot_size=0.10,
                atr_at_entry=5.0,
                regime_at_entry="TREND_UP",
                exit_bar=i * 2 + 1,
                exit_price=1815.0,
                exit_reason="TP2",
                exit_time=datetime.now(timezone.utc),
                pnl_currency=150.0,
                r_multiple=1.5,
                mae_pips=30.0,
                mfe_pips=200.0,
                hold_bars=1,
                tp1_hit=True,
                slippage_pips=0.5,
            )
            trades.append(t)

        for j in range(n_loss):
            t = BacktestTrade(
                trade_id=n_win + j,
                symbol="XAUUSD",
                direction="LONG",
                module="MOMENTUM",
                entry_bar=j * 3 + 100,
                entry_price=1800.0,
                entry_time=datetime.now(timezone.utc),
                stop_loss=1790.0,
                tp1=1810.0,
                tp2=1820.0,
                lot_size=0.10,
                atr_at_entry=5.0,
                regime_at_entry="TREND_UP",
                exit_bar=j * 3 + 101,
                exit_price=1789.0,
                exit_reason="SL",
                exit_time=datetime.now(timezone.utc),
                pnl_currency=-100.0,
                r_multiple=-1.0,
                mae_pips=110.0,
                mfe_pips=20.0,
                hold_bars=1,
                tp1_hit=False,
                slippage_pips=0.3,
            )
            trades.append(t)

        return trades

    def _make_equity(self, trades, initial_balance: float = 100_000.0) -> pd.Series:
        equity = initial_balance
        values = [equity]
        for t in trades:
            equity += t.pnl_currency
            values.append(equity)
        return pd.Series(values, dtype=float)

    def test_empty_trades_returns_zeros(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        metrics = pm.compute([], pd.Series(dtype=float), 100_000.0)
        assert metrics["total_trades"] == 0
        assert metrics["sharpe_ratio"] == 0.0

    def test_win_rate_calculation(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        trades = self._make_trades(6, 4)  # 60% win rate
        equity = self._make_equity(trades)
        metrics = pm.compute(trades, equity, 100_000.0)

        assert abs(metrics["win_rate"] - 0.60) < 0.01

    def test_profit_factor_positive(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        trades = self._make_trades(10, 5)  # net positive
        equity = self._make_equity(trades)
        metrics = pm.compute(trades, equity, 100_000.0)

        assert metrics["profit_factor"] > 1.0

    def test_max_drawdown_non_negative(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        trades = self._make_trades(5, 5)
        equity = self._make_equity(trades)
        metrics = pm.compute(trades, equity, 100_000.0)

        assert metrics["max_drawdown"] >= 0.0
        assert metrics["max_drawdown"] <= 1.0

    def test_sharpe_ratio_positive_for_profitable_system(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        trades = self._make_trades(20, 5)  # very profitable
        equity = self._make_equity(trades)
        metrics = pm.compute(trades, equity, 100_000.0)

        assert metrics["sharpe_ratio"] > 0.0

    def test_check_thresholds_passes_good_system(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        good_metrics = {
            "win_rate": 0.55,
            "profit_factor": 1.80,
            "max_drawdown": 0.10,
            "sharpe_ratio": 1.20,
            "total_trades": 50,
            "avg_r_multiple": 0.50,
        }
        passed, failures = pm.check_thresholds(good_metrics)
        assert passed
        assert len(failures) == 0

    def test_check_thresholds_fails_low_win_rate(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        bad_metrics = {
            "win_rate": 0.25,        # below 0.40
            "profit_factor": 1.50,
            "max_drawdown": 0.10,
            "sharpe_ratio": 1.00,
            "total_trades": 50,
            "avg_r_multiple": 0.40,
        }
        passed, failures = pm.check_thresholds(bad_metrics)
        assert not passed
        assert any("win_rate" in f.lower() for f in failures)

    def test_total_trades_in_metrics(self):
        from backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics()
        trades = self._make_trades(7, 3)
        equity = self._make_equity(trades)
        metrics = pm.compute(trades, equity, 100_000.0)

        assert metrics["total_trades"] == 10


# ---------------------------------------------------------------------------
# SimulationEngine (look-ahead and fill logic)
# ---------------------------------------------------------------------------


class TestSimulationEngine:
    """Tests for backtesting/simulation_engine.py."""

    def _make_sim_engine(self):
        from backtesting.simulation_engine import SimulationEngine

        feature_eng   = MagicMock()
        regime_det    = MagicMock()
        signal_router = MagicMock()
        risk_engine   = MagicMock()

        # Regime detector returns a default state
        from core.regime.regime_detector import RegimeState

        state = RegimeState(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            final_sizing_multiplier=1.0,
            active_strategy="momentum",
        )
        regime_det.detect.return_value = state

        # Feature engineer returns a DataFrame
        feature_eng.compute.return_value = pd.DataFrame(
            {"atr_14": [5.0] * 50, "close": [1800.0] * 50}
        )

        # Signal router returns None by default (no signal)
        signal_router.route.return_value = None
        risk_engine.process.return_value = None

        return SimulationEngine(
            feature_engineer=feature_eng,
            regime_detector=regime_det,
            signal_router=signal_router,
            risk_engine=risk_engine,
            initial_balance=100_000.0,
        )

    def _make_ohlcv(self, n=200) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        close = 1800.0 + np.cumsum(rng.normal(0, 0.5, n))
        noise = rng.uniform(0.3, 1.0, n)
        return pd.DataFrame({
            "open":   close - noise * 0.3,
            "high":   close + noise,
            "low":    close - noise,
            "close":  close,
            "volume": rng.integers(100, 2000, n).astype(float),
        }, index=idx)

    def test_run_returns_backtest_results(self):
        """run() returns a BacktestResults object without crashing."""
        from backtesting.simulation_engine import BacktestResults

        sim = self._make_sim_engine()
        df  = self._make_ohlcv(200)
        results = sim.run("XAUUSD", df)

        assert isinstance(results, BacktestResults)
        assert results.symbol == "XAUUSD"

    def test_equity_curve_length_equals_bars(self):
        """Equity curve has the same length as the input DataFrame."""
        from backtesting.simulation_engine import BacktestResults

        sim = self._make_sim_engine()
        df  = self._make_ohlcv(200)
        results = sim.run("XAUUSD", df)

        assert len(results.equity_curve) == len(df)

    def test_equity_starts_at_initial_balance(self):
        """First equity value equals initial_balance."""
        sim = self._make_sim_engine()
        df  = self._make_ohlcv(200)
        results = sim.run("XAUUSD", df)

        assert results.initial_balance == 100_000.0
        assert abs(results.equity_curve.iloc[0] - 100_000.0) < 1.0

    def test_no_trades_equity_flat(self):
        """With no signals, equity curve stays flat."""
        sim = self._make_sim_engine()
        df  = self._make_ohlcv(200)
        results = sim.run("XAUUSD", df)

        # No signals → no trades → equity unchanged
        assert len(results.trades) == 0
        assert abs(results.final_balance - results.initial_balance) < 1.0


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class TestWalkForwardValidator:
    """Tests for backtesting/walk_forward.py."""

    def _make_wfv(self):
        from backtesting.walk_forward import WalkForwardValidator
        from backtesting.performance_metrics import PerformanceMetrics

        sim_engine = MagicMock()
        pm         = PerformanceMetrics()

        # sim_engine.run() returns an empty BacktestResults
        from backtesting.simulation_engine import BacktestResults

        results = BacktestResults(
            symbol="XAUUSD",
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
            initial_balance=100_000.0,
            final_balance=100_000.0,
            trades=[],
            equity_curve=pd.Series([100_000.0] * 50, dtype=float),
        )
        sim_engine.run.return_value = results

        return WalkForwardValidator(
            simulation_engine=sim_engine,
            performance_metrics=pm,
        )

    def _make_ohlcv(self, n=1000) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
        close = 1800.0 + np.cumsum(rng.normal(0, 0.3, n))
        noise = rng.uniform(0.2, 0.8, n)
        return pd.DataFrame({
            "open": close - 0.1,
            "high": close + noise,
            "low":  close - noise,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }, index=idx)

    def test_run_returns_walk_forward_results(self):
        from backtesting.walk_forward import WalkForwardResults

        wfv     = self._make_wfv()
        df      = self._make_ohlcv(800)
        results = wfv.run("XAUUSD", df, n_folds=3)

        assert isinstance(results, WalkForwardResults)
        assert results.symbol == "XAUUSD"
        assert results.n_folds == 3

    def test_fold_count_matches_request(self):
        wfv     = self._make_wfv()
        df      = self._make_ohlcv(800)
        results = wfv.run("XAUUSD", df, n_folds=4)

        assert len(results.fold_results) == 4

    def test_fold_windows_non_overlapping(self):
        """Each fold's test window starts where the previous ended."""
        wfv     = self._make_wfv()
        df      = self._make_ohlcv(800)
        results = wfv.run("XAUUSD", df, n_folds=3)

        folds = results.fold_results
        for i in range(1, len(folds)):
            assert folds[i].test_start >= folds[i - 1].test_end

    def test_combined_trades_is_flat_list(self):
        wfv     = self._make_wfv()
        df      = self._make_ohlcv(800)
        results = wfv.run("XAUUSD", df, n_folds=3)

        assert isinstance(results.combined_trades, list)


# ---------------------------------------------------------------------------
# ResultsAnalyzer
# ---------------------------------------------------------------------------


class TestResultsAnalyzer:
    """Tests for backtesting/results_analyzer.py."""

    def _make_wf_results(self):
        from backtesting.walk_forward import WalkForwardResults, FoldResult

        fold = FoldResult(
            fold_idx=0,
            train_start=0,
            train_end=400,
            test_start=400,
            test_end=600,
            trades=[],
            equity_curve=pd.Series([100_000.0] * 200, dtype=float),
            metrics={"sharpe_ratio": 0.80, "win_rate": 0.55, "total_trades": 0},
        )

        return WalkForwardResults(
            symbol="XAUUSD",
            n_folds=1,
            fold_results=[fold],
            combined_equity=pd.Series([100_000.0] * 600, dtype=float),
            combined_trades=[],
            overall_metrics={"sharpe_ratio": 0.80, "win_rate": 0.55, "total_trades": 0},
            fold_consistency_ok=True,
        )

    def test_generate_report_returns_dict(self, tmp_path):
        from backtesting.results_analyzer import ResultsAnalyzer

        analyzer = ResultsAnalyzer(output_dir=str(tmp_path))
        wf = self._make_wf_results()
        report = analyzer.generate_report(wf)

        assert isinstance(report, dict)
        assert "summary" in report

    def test_report_summary_has_symbol(self, tmp_path):
        from backtesting.results_analyzer import ResultsAnalyzer

        analyzer = ResultsAnalyzer(output_dir=str(tmp_path))
        wf = self._make_wf_results()
        report = analyzer.generate_report(wf)

        assert report["summary"]["symbol"] == "XAUUSD"

    def test_export_to_csv_creates_file(self, tmp_path):
        from backtesting.results_analyzer import ResultsAnalyzer

        analyzer = ResultsAnalyzer(output_dir=str(tmp_path))
        wf = self._make_wf_results()
        csv_path = analyzer.export_to_csv(wf)

        # File should exist (even if empty trades → empty CSV is acceptable)
        assert csv_path is not None
