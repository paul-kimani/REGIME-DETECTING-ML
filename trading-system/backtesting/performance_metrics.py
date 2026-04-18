"""PerformanceMetrics — Sharpe, Sortino, Calmar, profit factor, R-multiple, drawdown stats."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# M15 bars per year: 252 trading days * 6.5 trading hours * 4 bars/hour = 6,552
# The spec calls for 252 * 96 (i.e. 24-hour sessions, as in FX/metals).
_BARS_PER_YEAR: int = 252 * 96     # 24,192

# Minimum trade count for meaningful statistics
_MIN_TRADES: int = 20

# Thresholds for check_thresholds()
_THRESHOLDS: dict[str, tuple] = {
    "win_rate":      (0.40, ">="),
    "profit_factor": (1.20, ">="),
    "max_drawdown":  (0.25, "<="),
    "sharpe_ratio":  (0.50, ">="),
    "total_trades":  (_MIN_TRADES, ">="),
    "avg_r_multiple": (0.10, ">="),
}


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------


class PerformanceMetrics:
    """Compute comprehensive performance statistics from a list of BacktestTrade objects.

    All methods accept the plain list of :class:`~backtesting.simulation_engine.BacktestTrade`
    objects produced by :class:`~backtesting.simulation_engine.SimulationEngine`.
    """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        trades: list,
        equity_curve: pd.Series,
        initial_balance: float = 100_000.0,
    ) -> dict:
        """Compute all performance metrics.

        Args:
            trades:          List of closed :class:`BacktestTrade` objects.
            equity_curve:    Equity series indexed by bar number, produced by
                             :class:`~backtesting.simulation_engine.SimulationEngine`.
            initial_balance: Starting account balance in account currency.

        Returns:
            Flat dict of metrics.  Returns a zeroed dict if *trades* is empty.
        """
        if not trades or equity_curve.empty:
            logger.info("PerformanceMetrics.compute(): no trades or empty equity curve — returning zeros")
            return self._empty_metrics()

        # ----------------------------------------------------------------
        # Basic building blocks
        # ----------------------------------------------------------------
        n_trades: int = len(trades)
        pnls: np.ndarray = np.array([t.pnl_currency for t in trades], dtype=float)
        r_multiples: np.ndarray = np.array([t.r_multiple for t in trades], dtype=float)
        winners: np.ndarray = pnls[pnls > 0]
        losers:  np.ndarray = pnls[pnls < 0]

        win_rate: float = float(len(winners)) / n_trades if n_trades > 0 else 0.0

        gross_profit: float = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss:   float = float(abs(losers.sum())) if len(losers) > 0 else 0.0

        if gross_loss > 0:
            profit_factor: float = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        expectancy: float = float(np.mean(r_multiples)) if n_trades > 0 else 0.0
        avg_r_multiple: float = expectancy

        avg_win: float  = float(np.mean(r_multiples[pnls > 0])) if len(winners) > 0 else 0.0
        avg_loss: float = float(np.mean(r_multiples[pnls < 0])) if len(losers)  > 0 else 0.0

        largest_loss: float = float(min(pnls)) if n_trades > 0 else 0.0

        # ----------------------------------------------------------------
        # Equity curve metrics
        # ----------------------------------------------------------------
        equity: pd.Series = equity_curve.dropna().astype(float)
        final_balance: float = float(equity.iloc[-1]) if not equity.empty else initial_balance
        n_bars: int = len(equity)

        total_return: float = (
            (final_balance - initial_balance) / initial_balance
            if initial_balance > 0 else 0.0
        )

        annualized_return: float = self._annualize(total_return, n_bars)

        # Returns series (bar-to-bar changes in equity)
        bar_returns: pd.Series = equity.pct_change().dropna()

        # ----------------------------------------------------------------
        # Drawdown
        # ----------------------------------------------------------------
        dd_series: pd.Series = self._compute_drawdown_series(equity)
        max_drawdown: float  = float(dd_series.min()) if not dd_series.empty else 0.0
        max_drawdown = abs(max_drawdown)   # store as positive fraction

        max_dd_duration: int = self._compute_drawdown_duration(dd_series)

        # ----------------------------------------------------------------
        # Sharpe ratio  (annualised, rf = 0)
        # ----------------------------------------------------------------
        if len(bar_returns) > 1 and bar_returns.std(ddof=1) > 0:
            sharpe_ratio = float(
                bar_returns.mean() / bar_returns.std(ddof=1) * np.sqrt(_BARS_PER_YEAR)
            )
        else:
            sharpe_ratio = 0.0

        # ----------------------------------------------------------------
        # Sortino ratio  (downside deviation only)
        # ----------------------------------------------------------------
        sortino_ratio: float = self._compute_sortino(bar_returns)

        # ----------------------------------------------------------------
        # Calmar ratio
        # ----------------------------------------------------------------
        if max_drawdown > 0:
            calmar_ratio: float = annualized_return / max_drawdown
        else:
            calmar_ratio = float("inf") if annualized_return > 0 else 0.0

        # ----------------------------------------------------------------
        # Consecutive losses
        # ----------------------------------------------------------------
        consecutive_losses_max: int = self._max_consecutive_losses(pnls)

        # ----------------------------------------------------------------
        # Slippage and hold time
        # ----------------------------------------------------------------
        slippages: np.ndarray = np.array([t.slippage_pips for t in trades], dtype=float)
        avg_slippage: float   = float(np.mean(slippages)) if len(slippages) > 0 else 0.0

        hold_bars_arr: np.ndarray = np.array([t.hold_bars for t in trades], dtype=float)
        avg_hold_time: float      = float(np.mean(hold_bars_arr)) if n_trades > 0 else 0.0

        # ----------------------------------------------------------------
        # Trade frequency per week
        # ----------------------------------------------------------------
        # Estimate weeks from equity curve length (M15 bars)
        bars_per_week: float = 96 * 5.0     # 5 trading days
        weeks: float = n_bars / bars_per_week if n_bars > 0 else 1.0
        trade_frequency_per_week: float = n_trades / weeks if weeks > 0 else 0.0

        metrics = {
            "total_trades":              n_trades,
            "total_return":              total_return,
            "annualized_return":         annualized_return,
            "profit_factor":             profit_factor,
            "expectancy":                expectancy,
            "max_drawdown":              max_drawdown,
            "max_drawdown_duration":     max_dd_duration,
            "sharpe_ratio":              sharpe_ratio,
            "sortino_ratio":             sortino_ratio,
            "calmar_ratio":              calmar_ratio,
            "win_rate":                  win_rate,
            "avg_r_multiple":            avg_r_multiple,
            "avg_win":                   avg_win,
            "avg_loss":                  avg_loss,
            "largest_loss":              largest_loss,
            "consecutive_losses_max":    consecutive_losses_max,
            "avg_slippage":              avg_slippage,
            "avg_hold_time_candles":     avg_hold_time,
            "trade_frequency_per_week":  trade_frequency_per_week,
            "gross_profit":              gross_profit,
            "gross_loss":                gross_loss,
            "final_balance":             final_balance,
            "initial_balance":           initial_balance,
        }

        logger.info(
            "PerformanceMetrics.compute(): %d trades | win_rate=%.2f%% | "
            "pf=%.2f | sharpe=%.2f | max_dd=%.2f%% | calmar=%.2f",
            n_trades,
            win_rate * 100,
            profit_factor if profit_factor != float("inf") else 9999,
            sharpe_ratio,
            max_drawdown * 100,
            calmar_ratio if calmar_ratio != float("inf") else 9999,
        )

        return metrics

    # ------------------------------------------------------------------
    # Breakdowns
    # ------------------------------------------------------------------

    def breakdown_by_module(self, trades: list) -> dict:
        """Return per-module metrics dict.

        Args:
            trades: List of :class:`BacktestTrade` objects.

        Returns:
            Dict keyed by module name (e.g. ``"MOMENTUM"``) whose values are
            metrics dicts from :meth:`compute` computed on that module's trades.
        """
        return self._breakdown(trades, key_fn=lambda t: t.module.upper())

    def breakdown_by_regime(self, trades: list) -> dict:
        """Return per-regime metrics dict.

        Args:
            trades: List of :class:`BacktestTrade` objects.

        Returns:
            Dict keyed by ``regime_at_entry`` (e.g. ``"trend_up"``) whose
            values are metrics dicts from :meth:`compute`.
        """
        return self._breakdown(trades, key_fn=lambda t: t.regime_at_entry)

    def _breakdown(self, trades: list, key_fn) -> dict:
        """Generic breakdown helper.

        Args:
            trades: Full trade list.
            key_fn: Callable that maps a trade to a string bucket key.

        Returns:
            Dict mapping bucket key to a metrics dict.
        """
        buckets: dict[str, list] = {}
        for trade in trades:
            key = key_fn(trade)
            buckets.setdefault(key, []).append(trade)

        result: dict = {}
        for bucket_name, bucket_trades in buckets.items():
            # Build a minimal equity stub from cumulative PnL for this bucket
            pnl_series = pd.Series(
                [t.pnl_currency for t in bucket_trades],
                dtype=float,
            )
            initial = 100_000.0
            equity_stub = (initial + pnl_series.cumsum()).reset_index(drop=True)
            equity_stub = pd.Series(
                np.concatenate([[initial], equity_stub.values]),
                dtype=float,
            )
            result[bucket_name] = self.compute(
                bucket_trades, equity_stub, initial_balance=initial
            )

        return result

    # ------------------------------------------------------------------
    # Threshold validation
    # ------------------------------------------------------------------

    def check_thresholds(self, metrics: dict) -> tuple:
        """Validate metrics against minimum acceptable performance thresholds.

        Minimums:
        - win_rate >= 0.40
        - profit_factor >= 1.20
        - max_drawdown <= 0.25  (stored as positive fraction)
        - sharpe_ratio >= 0.50
        - total_trades >= 20
        - avg_r_multiple >= 0.10

        Args:
            metrics: Output of :meth:`compute`.

        Returns:
            Tuple of ``(passed: bool, failures: list[str])``.
            *passed* is ``True`` only when the failures list is empty.
        """
        failures: list[str] = []

        for metric_name, (threshold, direction) in _THRESHOLDS.items():
            value = metrics.get(metric_name)
            if value is None:
                failures.append(f"{metric_name}: missing from metrics dict")
                continue

            # Handle Inf values
            numeric_value = float(value) if value != float("inf") else 1e9

            if direction == ">=":
                if numeric_value < threshold:
                    failures.append(
                        f"{metric_name}={numeric_value:.4f} < minimum {threshold}"
                    )
            elif direction == "<=":
                if numeric_value > threshold:
                    failures.append(
                        f"{metric_name}={numeric_value:.4f} > maximum {threshold}"
                    )

        passed = len(failures) == 0

        if passed:
            logger.info("check_thresholds: all thresholds passed")
        else:
            logger.info("check_thresholds: %d failure(s): %s", len(failures), failures)

        return passed, failures

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _compute_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Compute drawdown at each bar as a fraction from the running peak.

        Drawdown values are <= 0 (e.g. -0.10 means 10% below the peak).

        Args:
            equity: Equity curve as a float Series.

        Returns:
            pd.Series of drawdown fractions aligned to *equity*'s index.
        """
        if equity.empty:
            return pd.Series(dtype=float)

        running_peak: pd.Series = equity.cummax()
        # Avoid divide-by-zero when peak is 0
        safe_peak = running_peak.replace(0.0, np.nan)
        drawdown: pd.Series = (equity - running_peak) / safe_peak
        return drawdown.fillna(0.0)

    def _annualize(
        self,
        total_return: float,
        n_bars: int,
        bars_per_year: int = _BARS_PER_YEAR,
    ) -> float:
        """Geometric annualisation: ``(1 + r)^(bars_per_year / n_bars) - 1``.

        Returns 0.0 when *n_bars* is 0 or negative.

        Args:
            total_return:  Total fractional return over the period.
            n_bars:        Number of bars in the period.
            bars_per_year: Number of bars in one year (default 252 * 96).

        Returns:
            Annualised return as a fraction.
        """
        if n_bars <= 0:
            return 0.0

        exponent = bars_per_year / n_bars
        try:
            annualised = (1.0 + total_return) ** exponent - 1.0
        except (OverflowError, ZeroDivisionError):
            annualised = 0.0

        return float(annualised)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_sortino(self, bar_returns: pd.Series) -> float:
        """Compute annualised Sortino ratio (rf = 0).

        Uses only the returns below zero as the downside deviation.

        Args:
            bar_returns: Series of bar-to-bar percentage returns.

        Returns:
            Annualised Sortino ratio, or 0.0 when insufficient data.
        """
        if len(bar_returns) < 2:
            return 0.0

        mean_return = float(bar_returns.mean())
        downside: pd.Series = bar_returns[bar_returns < 0]

        if len(downside) < 2:
            # No losing bars — return a large positive value capped at 10
            return 10.0 if mean_return > 0 else 0.0

        downside_std = float(downside.std(ddof=1))
        if downside_std <= 0:
            return 0.0

        return float(mean_return / downside_std * np.sqrt(_BARS_PER_YEAR))

    def _compute_drawdown_duration(self, dd_series: pd.Series) -> int:
        """Compute the longest period (in bars) spent in a drawdown.

        A drawdown period is a contiguous run of bars where drawdown < 0.

        Args:
            dd_series: Output of :meth:`_compute_drawdown_series`.

        Returns:
            Maximum number of consecutive bars spent below the prior peak.
        """
        if dd_series.empty:
            return 0

        max_duration: int = 0
        current: int = 0

        for val in dd_series:
            if val < 0:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0

        return max_duration

    def _max_consecutive_losses(self, pnls: np.ndarray) -> int:
        """Find the longest consecutive streak of losing trades.

        Args:
            pnls: Array of per-trade PnL values.

        Returns:
            Maximum consecutive loss count.
        """
        max_streak: int = 0
        current: int = 0

        for pnl in pnls:
            if pnl < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        return max_streak

    @staticmethod
    def _empty_metrics() -> dict:
        """Return a zeroed metrics dict for the case of no trades.

        Returns:
            Dict with all expected keys set to 0 or sensible defaults.
        """
        return {
            "total_trades":              0,
            "total_return":              0.0,
            "annualized_return":         0.0,
            "profit_factor":             0.0,
            "expectancy":                0.0,
            "max_drawdown":              0.0,
            "max_drawdown_duration":     0,
            "sharpe_ratio":              0.0,
            "sortino_ratio":             0.0,
            "calmar_ratio":              0.0,
            "win_rate":                  0.0,
            "avg_r_multiple":            0.0,
            "avg_win":                   0.0,
            "avg_loss":                  0.0,
            "largest_loss":              0.0,
            "consecutive_losses_max":    0,
            "avg_slippage":              0.0,
            "avg_hold_time_candles":     0.0,
            "trade_frequency_per_week":  0.0,
            "gross_profit":              0.0,
            "gross_loss":                0.0,
            "final_balance":             0.0,
            "initial_balance":           0.0,
        }
