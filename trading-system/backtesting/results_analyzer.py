"""ResultsAnalyzer — generates report, equity curve, drawdown, and R-distribution plots."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.performance_metrics import PerformanceMetrics
from backtesting.walk_forward import WalkForwardResults
from core.utils.logger import get_logger

logger = get_logger(__name__)

_DPI: int = 150


# ---------------------------------------------------------------------------
# ResultsAnalyzer
# ---------------------------------------------------------------------------


class ResultsAnalyzer:
    """Generates reports, charts, and CSV exports from walk-forward results.

    Args:
        output_dir: Directory where all reports and plot files will be saved.
                    Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pm = PerformanceMetrics()
        logger.info("ResultsAnalyzer initialised: output_dir=%s", self._output_dir)

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def generate_report(self, wf_results: WalkForwardResults) -> dict:
        """Generate a full structured report dict.

        The returned structure is::

            {
                "summary": {
                    "symbol": str,
                    "n_folds": int,
                    "total_trades": int,
                    "overall_metrics": dict,
                    "fold_consistency_ok": bool,
                    "fold_consistency_reason": str,
                },
                "fold_breakdown": [
                    {
                        "fold": int,
                        "train_bars": int,
                        "test_bars": int,
                        "trades": int,
                        "metrics": dict,
                    },
                    ...
                ],
                "module_breakdown": dict,
                "regime_breakdown": dict,
                "plots": {
                    "equity_curve": str,
                    "drawdown": str,
                    "r_distribution": str,
                    "regime_performance": str,
                },
            }

        Args:
            wf_results: Completed :class:`~backtesting.walk_forward.WalkForwardResults`.

        Returns:
            Nested dict containing the full report.
        """
        logger.info(
            "[%s] Generating report: %d folds, %d total trades",
            wf_results.symbol,
            wf_results.n_folds,
            len(wf_results.combined_trades),
        )

        # ── Summary ──────────────────────────────────────────────────────────
        summary: dict[str, Any] = {
            "symbol": wf_results.symbol,
            "n_folds": wf_results.n_folds,
            "total_trades": len(wf_results.combined_trades),
            "overall_metrics": wf_results.overall_metrics,
            "fold_consistency_ok": wf_results.fold_consistency_ok,
            "fold_consistency_reason": wf_results.fold_consistency_reason,
        }

        # ── Fold breakdown ────────────────────────────────────────────────────
        fold_breakdown: list[dict[str, Any]] = []
        for fr in wf_results.fold_results:
            fold_breakdown.append(
                {
                    "fold": fr.fold_idx,
                    "train_bars": fr.train_end - fr.train_start,
                    "test_bars": fr.test_end - fr.test_start,
                    "trades": len(fr.trades),
                    "metrics": fr.metrics,
                }
            )

        # ── Module and regime breakdowns ──────────────────────────────────────
        module_breakdown: dict = {}
        regime_breakdown: dict = {}
        if wf_results.combined_trades:
            try:
                module_breakdown = self._pm.breakdown_by_module(
                    wf_results.combined_trades
                )
            except Exception as exc:
                logger.warning("breakdown_by_module failed: %s", exc)

            try:
                regime_breakdown = self._pm.breakdown_by_regime(
                    wf_results.combined_trades
                )
            except Exception as exc:
                logger.warning("breakdown_by_regime failed: %s", exc)

        # ── Plots ─────────────────────────────────────────────────────────────
        equity_path = self.plot_equity_curve(
            wf_results.combined_equity,
            title=f"Equity_Curve_{wf_results.symbol}",
        )
        drawdown_path = self.plot_drawdown(wf_results.combined_equity)
        r_dist_path = self.plot_r_distribution(wf_results.combined_trades)
        regime_perf_path = self.plot_regime_performance(wf_results.combined_trades)

        plots: dict[str, str] = {
            "equity_curve": equity_path,
            "drawdown": drawdown_path,
            "r_distribution": r_dist_path,
            "regime_performance": regime_perf_path,
        }

        report = {
            "summary": summary,
            "fold_breakdown": fold_breakdown,
            "module_breakdown": module_breakdown,
            "regime_breakdown": regime_breakdown,
            "plots": plots,
        }

        logger.info("[%s] Report generated successfully", wf_results.symbol)
        return report

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        equity: pd.Series,
        title: str = "Equity Curve",
    ) -> str:
        """Plot equity curve and save to ``reports/{title}.png``.

        The chart includes:

        - Blue line for the equity curve.
        - Red shading below the starting balance for drawdown periods.
        - Grid, title, and axis labels.

        Args:
            equity: Equity values indexed by bar number.
            title:  Plot title; also used as the file-name stem.

        Returns:
            Absolute path to the saved PNG file.
        """
        safe_title = title.replace(" ", "_")
        file_path = self._output_dir / f"{safe_title}.png"

        fig, ax = plt.subplots(figsize=(12, 5))

        if equity.empty:
            ax.set_title(title)
            ax.set_xlabel("Bar")
            ax.set_ylabel("Equity")
            plt.tight_layout()
            fig.savefig(str(file_path), dpi=_DPI)
            plt.close(fig)
            logger.warning("plot_equity_curve: equity series is empty — blank chart saved")
            return str(file_path)

        x = np.arange(len(equity))
        y = equity.values.astype(float)
        start_balance = float(y[0])

        # Red shading below starting balance
        ax.fill_between(
            x, y, start_balance,
            where=(y < start_balance),
            color="red", alpha=0.25, label="Drawdown vs start",
        )

        ax.plot(x, y, color="steelblue", linewidth=1.5, label="Equity")
        ax.axhline(start_balance, color="gray", linestyle="--", linewidth=0.8)

        ax.set_title(title)
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        fig.savefig(str(file_path), dpi=_DPI)
        plt.close(fig)

        logger.info("plot_equity_curve: saved to %s", file_path)
        return str(file_path)

    def plot_drawdown(self, equity: pd.Series) -> str:
        """Plot drawdown curve and save to ``reports/drawdown.png``.

        Shows a red-filled area representing the drawdown from the running peak
        at each bar.

        Args:
            equity: Equity values indexed by bar number.

        Returns:
            Absolute path to the saved PNG file.
        """
        file_path = self._output_dir / "drawdown.png"

        fig, ax = plt.subplots(figsize=(12, 4))

        if equity.empty:
            ax.set_title("Drawdown")
            ax.set_xlabel("Bar")
            ax.set_ylabel("Drawdown (%)")
            plt.tight_layout()
            fig.savefig(str(file_path), dpi=_DPI)
            plt.close(fig)
            logger.warning("plot_drawdown: equity series is empty — blank chart saved")
            return str(file_path)

        eq = equity.values.astype(float)
        running_peak = np.maximum.accumulate(eq)
        # Avoid division by zero
        safe_peak = np.where(running_peak == 0.0, np.nan, running_peak)
        drawdown_pct = ((eq - running_peak) / safe_peak) * 100.0
        drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0)

        x = np.arange(len(drawdown_pct))
        ax.fill_between(x, drawdown_pct, 0, color="red", alpha=0.5, label="Drawdown")
        ax.plot(x, drawdown_pct, color="darkred", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.6)

        min_dd = float(drawdown_pct.min())
        ax.set_title(f"Drawdown (Max: {min_dd:.2f}%)")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        fig.savefig(str(file_path), dpi=_DPI)
        plt.close(fig)

        logger.info("plot_drawdown: saved to %s", file_path)
        return str(file_path)

    def plot_r_distribution(self, trades: list) -> str:
        """Histogram of R-multiples from all trades.

        Features:

        - Green bars for positive R, red bars for negative R.
        - Vertical line at 0 and at the mean R.

        Saved to ``reports/r_distribution.png``.

        Args:
            trades: List of :class:`~backtesting.simulation_engine.BacktestTrade`
                    objects.

        Returns:
            Absolute path to the saved PNG file.
        """
        file_path = self._output_dir / "r_distribution.png"

        fig, ax = plt.subplots(figsize=(10, 5))

        if not trades:
            ax.set_title("R-Multiple Distribution (no trades)")
            ax.set_xlabel("R-Multiple")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            fig.savefig(str(file_path), dpi=_DPI)
            plt.close(fig)
            logger.warning("plot_r_distribution: no trades — blank chart saved")
            return str(file_path)

        r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
        mean_r = float(np.mean(r_multiples))

        # Compute bin edges
        n_bins = max(20, len(trades) // 5)
        bin_edges = np.linspace(r_multiples.min(), r_multiples.max(), n_bins + 1)

        # Separate positive and negative buckets
        pos_mask = r_multiples >= 0
        neg_mask = ~pos_mask

        if pos_mask.any():
            ax.hist(
                r_multiples[pos_mask],
                bins=bin_edges,
                color="green",
                alpha=0.7,
                label="Positive R",
            )
        if neg_mask.any():
            ax.hist(
                r_multiples[neg_mask],
                bins=bin_edges,
                color="red",
                alpha=0.7,
                label="Negative R",
            )

        ax.axvline(0, color="black", linewidth=1.2, linestyle="-", label="Zero")
        ax.axvline(
            mean_r,
            color="navy",
            linewidth=1.5,
            linestyle="--",
            label=f"Mean R = {mean_r:.2f}",
        )

        ax.set_title(f"R-Multiple Distribution (n={len(trades)}, mean={mean_r:.2f})")
        ax.set_xlabel("R-Multiple")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        fig.savefig(str(file_path), dpi=_DPI)
        plt.close(fig)

        logger.info("plot_r_distribution: saved to %s", file_path)
        return str(file_path)

    def plot_regime_performance(self, trades: list) -> str:
        """Bar chart of win rate and average R-multiple per regime.

        Groups trades by ``regime_at_entry`` and plots two side-by-side bars
        per regime: win rate (%) and average R-multiple.

        Saved to ``reports/regime_performance.png``.

        Args:
            trades: List of :class:`~backtesting.simulation_engine.BacktestTrade`
                    objects.

        Returns:
            Absolute path to the saved PNG file.
        """
        file_path = self._output_dir / "regime_performance.png"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if not trades:
            for ax in axes:
                ax.set_title("Regime Performance (no trades)")
            plt.tight_layout()
            fig.savefig(str(file_path), dpi=_DPI)
            plt.close(fig)
            logger.warning("plot_regime_performance: no trades — blank chart saved")
            return str(file_path)

        # Aggregate per regime
        regime_stats: dict[str, dict[str, list]] = {}
        for trade in trades:
            regime = str(trade.regime_at_entry)
            if regime not in regime_stats:
                regime_stats[regime] = {"pnls": [], "r_multiples": []}
            regime_stats[regime]["pnls"].append(trade.pnl_currency)
            regime_stats[regime]["r_multiples"].append(trade.r_multiple)

        regimes = sorted(regime_stats.keys())
        win_rates = []
        avg_rs = []

        for regime in regimes:
            pnls = np.array(regime_stats[regime]["pnls"], dtype=float)
            rs = np.array(regime_stats[regime]["r_multiples"], dtype=float)
            win_rates.append(float(np.mean(pnls > 0)) * 100.0)
            avg_rs.append(float(np.mean(rs)))

        x = np.arange(len(regimes))
        bar_width = 0.6

        # Win rate chart
        ax_wr = axes[0]
        colours_wr = ["green" if wr >= 50 else "red" for wr in win_rates]
        bars = ax_wr.bar(x, win_rates, width=bar_width, color=colours_wr, alpha=0.8)
        ax_wr.axhline(50, color="black", linestyle="--", linewidth=0.8)
        ax_wr.set_title("Win Rate by Regime (%)")
        ax_wr.set_xlabel("Regime")
        ax_wr.set_ylabel("Win Rate (%)")
        ax_wr.set_xticks(x)
        ax_wr.set_xticklabels(regimes, rotation=30, ha="right")
        ax_wr.set_ylim(0, 100)
        ax_wr.grid(True, axis="y", alpha=0.4)
        for bar_rect, val in zip(bars, win_rates):
            ax_wr.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                bar_rect.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

        # Avg R chart
        ax_r = axes[1]
        colours_r = ["green" if r >= 0 else "red" for r in avg_rs]
        bars_r = ax_r.bar(x, avg_rs, width=bar_width, color=colours_r, alpha=0.8)
        ax_r.axhline(0, color="black", linewidth=0.8)
        ax_r.set_title("Average R-Multiple by Regime")
        ax_r.set_xlabel("Regime")
        ax_r.set_ylabel("Average R-Multiple")
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(regimes, rotation=30, ha="right")
        ax_r.grid(True, axis="y", alpha=0.4)
        for bar_rect, val in zip(bars_r, avg_rs):
            va = "bottom" if val >= 0 else "top"
            offset = 0.01 if val >= 0 else -0.01
            ax_r.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                val + offset,
                f"{val:.2f}R",
                ha="center", va=va, fontsize=8,
            )

        plt.tight_layout()
        fig.savefig(str(file_path), dpi=_DPI)
        plt.close(fig)

        logger.info("plot_regime_performance: saved to %s", file_path)
        return str(file_path)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_to_csv(self, trades: list, path: str) -> None:
        """Export trade list to CSV.

        Converts :class:`~backtesting.simulation_engine.BacktestTrade`
        dataclass instances to dicts before writing.

        Args:
            trades: List of :class:`~backtesting.simulation_engine.BacktestTrade`
                    objects.
            path:   Destination file path (created/overwritten).
        """
        if not trades:
            logger.warning("export_to_csv: no trades to export — empty file written to %s", path)
            pd.DataFrame().to_csv(path, index=False)
            return

        rows: list[dict] = []
        for trade in trades:
            try:
                rows.append(asdict(trade))
            except TypeError:
                # Fallback: manual __dict__ conversion for non-dataclass objects
                rows.append(vars(trade))

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("export_to_csv: %d trades written to %s", len(trades), path)
