"""WalkForwardValidator — expanding-window walk-forward cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    train_start: int        # bar index
    train_end: int
    test_start: int
    test_end: int
    trades: list
    equity_curve: pd.Series
    metrics: dict


@dataclass
class WalkForwardResults:
    """Aggregated results from all walk-forward folds."""

    symbol: str
    n_folds: int
    fold_results: list              # list of FoldResult
    combined_equity: pd.Series     # stitched test periods
    combined_trades: list
    overall_metrics: dict = field(default_factory=dict)
    fold_consistency_ok: bool = True
    fold_consistency_reason: str = ""


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Expanding-window walk-forward cross-validation.

    Splits data into N folds where:
    - Training window expands on each fold.
    - Test window is fixed size (next ~15 % of data after training).
    - Models are retrained on each fold's training data.
    - Simulation is run on each fold's test period only.

    Args:
        simulation_engine:   :class:`~backtesting.simulation_engine.SimulationEngine`
                             instance used to run each test fold.
        performance_metrics: :class:`~backtesting.performance_metrics.PerformanceMetrics`
                             instance used to compute fold and overall metrics.
        regime_detector:     Optional regime detector with a
                             ``train_all(data_dict, symbols)`` method that is
                             called before each fold's simulation.
        mlflow_tracker:      Optional MLflow tracker with a
                             ``log_fold(fold_idx, metrics)`` method.
    """

    def __init__(
        self,
        simulation_engine,
        performance_metrics,
        regime_detector=None,
        mlflow_tracker=None,
    ) -> None:
        self._engine = simulation_engine
        self._pm = performance_metrics
        self._rd = regime_detector
        self._mlflow = mlflow_tracker

        logger.info(
            "WalkForwardValidator initialised: regime_detector=%s mlflow=%s",
            regime_detector is not None,
            mlflow_tracker is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        full_data: pd.DataFrame,
        n_folds: int = 4,
    ) -> WalkForwardResults:
        """Run full walk-forward validation.

        Data split for *n_folds* = 4::

            Initial train : first 40 % of data
            Each fold adds ~15 % more data to the training window
            Test period   : next ~15 % after each training window

        Per fold:

        1. Slice ``train_data = full_data[0:train_end]``.
        2. Retrain ``regime_detector`` on *train_data* if available
           via ``regime_detector.train_all({symbol: train_data}, [symbol])``.
        3. Run ``simulation_engine.run(symbol, full_data, start_idx=test_start)``
           then retain only trades whose ``entry_bar`` falls within
           ``[test_start, test_end)``.
        4. Compute fold metrics using ``performance_metrics.compute()``.
        5. Log the fold to *mlflow_tracker* if available.

        After all folds:

        - Stitch combined equity curve from test periods.
        - Compute ``overall_metrics``.
        - Call :meth:`check_fold_consistency`.

        Args:
            symbol:    Instrument identifier, e.g. ``"XAUUSD"``.
            full_data: Full OHLCV DataFrame sorted ascending.
            n_folds:   Number of walk-forward folds (default 4).

        Returns:
            :class:`WalkForwardResults` populated with all fold data.
        """
        if full_data.empty:
            raise ValueError("full_data must not be empty")
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")

        n_bars: int = len(full_data)
        boundaries = self._compute_fold_boundaries(n_bars, n_folds)
        logger.info(
            "[%s] WalkForward.run(): %d bars, %d folds", symbol, n_bars, n_folds
        )

        fold_results: list[FoldResult] = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(
            boundaries
        ):
            logger.info(
                "[%s] Fold %d/%d — train=[%d:%d] test=[%d:%d]",
                symbol, fold_idx + 1, n_folds,
                train_start, train_end, test_start, test_end,
            )

            try:
                fold_result = self._run_fold(
                    symbol=symbol,
                    full_data=full_data,
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            except Exception as exc:
                logger.error(
                    "[%s] Fold %d failed with exception: %s — inserting empty fold",
                    symbol, fold_idx, exc,
                )
                fold_result = FoldResult(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    trades=[],
                    equity_curve=pd.Series(dtype=float),
                    metrics=self._pm.compute([], pd.Series(dtype=float)),
                )

            fold_results.append(fold_result)

            # Log to mlflow if available
            if self._mlflow is not None:
                try:
                    self._mlflow.log_fold(fold_idx, fold_result.metrics)
                except Exception as exc:
                    logger.warning(
                        "[%s] mlflow.log_fold(%d) failed: %s", symbol, fold_idx, exc
                    )

        # ------------------------------------------------------------------
        # Aggregate across folds
        # ------------------------------------------------------------------
        combined_trades: list = []
        for fr in fold_results:
            combined_trades.extend(fr.trades)

        combined_equity = self._stitch_equity(fold_results)

        overall_metrics = self._pm.compute(
            combined_trades, combined_equity
        )

        wf_results = WalkForwardResults(
            symbol=symbol,
            n_folds=n_folds,
            fold_results=fold_results,
            combined_equity=combined_equity,
            combined_trades=combined_trades,
            overall_metrics=overall_metrics,
        )

        self.check_fold_consistency(wf_results)

        logger.info(
            "[%s] WalkForward complete: %d total trades | sharpe=%.2f | "
            "max_dd=%.2f%% | consistency_ok=%s",
            symbol,
            overall_metrics.get("total_trades", 0),
            overall_metrics.get("sharpe_ratio", 0.0),
            overall_metrics.get("max_drawdown", 0.0) * 100,
            wf_results.fold_consistency_ok,
        )

        return wf_results

    def check_fold_consistency(self, results: WalkForwardResults) -> bool:
        """Check no single fold is carrying performance.

        Criteria:

        - Any fold's Sharpe ratio deviating more than 40 % from the mean is
          flagged as inconsistent.
        - Any fold with 0 trades is flagged.

        Sets ``results.fold_consistency_ok`` and
        ``results.fold_consistency_reason`` in-place.

        Args:
            results: :class:`WalkForwardResults` to inspect.

        Returns:
            ``True`` if all folds are consistent.
        """
        issues: list[str] = []

        fold_sharpes: list[float] = []
        for fr in results.fold_results:
            n_trades = fr.metrics.get("total_trades", 0)
            if n_trades == 0:
                issues.append(f"Fold {fr.fold_idx} has 0 trades")
            fold_sharpes.append(fr.metrics.get("sharpe_ratio", 0.0))

        if len(fold_sharpes) > 1:
            mean_sharpe = float(np.mean(fold_sharpes))
            if mean_sharpe != 0.0:
                for fr, sh in zip(results.fold_results, fold_sharpes):
                    deviation = abs(sh - mean_sharpe) / abs(mean_sharpe)
                    if deviation > 0.40:
                        issues.append(
                            f"Fold {fr.fold_idx} Sharpe={sh:.2f} deviates "
                            f"{deviation * 100:.1f}% from mean {mean_sharpe:.2f}"
                        )

        if issues:
            results.fold_consistency_ok = False
            results.fold_consistency_reason = "; ".join(issues)
            logger.warning(
                "[%s] Fold consistency FAILED: %s",
                results.symbol,
                results.fold_consistency_reason,
            )
        else:
            results.fold_consistency_ok = True
            results.fold_consistency_reason = "All folds consistent"
            logger.info("[%s] Fold consistency OK", results.symbol)

        return results.fold_consistency_ok

    def _compute_fold_boundaries(
        self,
        n_bars: int,
        n_folds: int,
    ) -> list[tuple[int, int, int, int]]:
        """Return list of ``(train_start, train_end, test_start, test_end)`` tuples.

        Data partition logic:

        - Initial training window: first 40 % of bars.
        - Remaining 60 % split into 2 * n_folds equal slices (half for
          incremental training growth, half for test windows).
        - For each fold k (0-based) the training window expands by one slice
          beyond the initial block, and the test window immediately follows.

        Args:
            n_bars:  Total number of bars in the dataset.
            n_folds: Number of folds.

        Returns:
            List of ``(train_start, train_end, test_start, test_end)`` index
            tuples (all bar-level, end is exclusive).
        """
        # 40 % initial training, remaining 60 % split into n_folds pairs
        initial_train_end: int = int(n_bars * 0.40)
        remaining: int = n_bars - initial_train_end
        fold_size: int = max(1, remaining // n_folds)

        boundaries: list[tuple[int, int, int, int]] = []

        for k in range(n_folds):
            train_start = 0
            train_end = initial_train_end + k * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_bars)

            # Clamp so we never go beyond the dataset
            if test_start >= n_bars:
                logger.warning(
                    "_compute_fold_boundaries: fold %d test_start=%d >= n_bars=%d — skipping",
                    k, test_start, n_bars,
                )
                break

            boundaries.append((train_start, train_end, test_start, test_end))

        return boundaries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        symbol: str,
        full_data: pd.DataFrame,
        fold_idx: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
    ) -> FoldResult:
        """Execute a single walk-forward fold.

        Args:
            symbol:      Instrument identifier.
            full_data:   Complete OHLCV DataFrame.
            fold_idx:    Zero-based fold index.
            train_start: Start bar index of training window (always 0).
            train_end:   Exclusive end of training window / start of test.
            test_start:  Inclusive start of test window.
            test_end:    Exclusive end of test window.

        Returns:
            Populated :class:`FoldResult`.

        Raises:
            ValueError: If there is insufficient data to train or test.
        """
        if train_end <= train_start:
            raise ValueError(
                f"Fold {fold_idx}: train_end={train_end} <= train_start={train_start}"
            )
        if test_end <= test_start:
            raise ValueError(
                f"Fold {fold_idx}: test_end={test_end} <= test_start={test_start}"
            )

        # 1. Slice training data
        train_data: pd.DataFrame = full_data.iloc[train_start:train_end]

        # 2. Retrain regime detector on training window
        if self._rd is not None:
            try:
                self._rd.train_all({symbol: train_data}, [symbol])
                logger.info(
                    "[%s] Fold %d: regime_detector retrained on %d bars",
                    symbol, fold_idx, len(train_data),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Fold %d: regime_detector.train_all() failed: %s",
                    symbol, fold_idx, exc,
                )

        # 3. Run simulation starting from test_start on the full dataset
        #    so the simulation engine has its warmup context but we only
        #    keep trades whose entry_bar falls within [test_start, test_end).
        sim_results = self._engine.run(
            symbol=symbol,
            data=full_data,
            start_idx=test_start,
        )

        # Filter trades to those that started within the test window
        fold_trades = [
            t for t in sim_results.trades
            if test_start <= t.entry_bar < test_end
        ]

        # Extract equity curve slice for the test window
        equity_slice = sim_results.equity_curve.loc[
            sim_results.equity_curve.index.isin(range(test_start, test_end))
        ]
        if equity_slice.empty:
            # Fall back to whatever equity was produced
            equity_slice = sim_results.equity_curve

        # 4. Compute fold metrics
        metrics = self._pm.compute(fold_trades, equity_slice)

        logger.info(
            "[%s] Fold %d: %d trades in test window [%d:%d] | "
            "sharpe=%.2f | win_rate=%.2f%%",
            symbol, fold_idx, len(fold_trades),
            test_start, test_end,
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("win_rate", 0.0) * 100,
        )

        return FoldResult(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            trades=fold_trades,
            equity_curve=equity_slice,
            metrics=metrics,
        )

    def _stitch_equity(self, fold_results: list[FoldResult]) -> pd.Series:
        """Concatenate per-fold equity curves into a single continuous series.

        Each fold's equity curve is re-based so that it starts at the last
        value of the previous fold, producing a seamless combined curve.

        Args:
            fold_results: Ordered list of :class:`FoldResult` objects.

        Returns:
            Combined equity :class:`pd.Series` indexed by bar number.
        """
        if not fold_results:
            return pd.Series(dtype=float)

        combined_parts: list[pd.Series] = []
        offset: float = 0.0

        for fr in fold_results:
            eq = fr.equity_curve.dropna()
            if eq.empty:
                continue

            if combined_parts:
                # Re-base: shift so that this fold starts where the last ended
                last_val = float(combined_parts[-1].iloc[-1])
                first_val = float(eq.iloc[0])
                offset = last_val - first_val

            rebased = eq + offset
            combined_parts.append(rebased)

        if not combined_parts:
            return pd.Series(dtype=float)

        return pd.concat(combined_parts).reset_index(drop=True)
