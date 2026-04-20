"""Backtesting CLI entry point — runs walk-forward validation and generates reports.

Run from the project root on Windows:
    python run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2024-01-01

MetaTrader5 is used directly for live data fetching if the DB is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to Python path so all core imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.utils.config import get_config
from core.utils.logger import get_logger

log = get_logger("run_backtest")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting for the hybrid ML trading system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Instrument symbol to backtest (e.g. XAUUSD, EURUSD).",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Backtest start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Backtest end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of walk-forward folds.",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Directory to write report files and charts.",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=100_000.0,
        help="Starting account balance in account currency.",
    )
    parser.add_argument(
        "--timeframe",
        default="M15",
        help="Primary timeframe for backtesting.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating plot PNG files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    timeframe: str,
) -> Optional[dict[str, object]]:
    """Fetch OHLCV data for the backtest period.

    Tries the local PostgreSQL database first, falls back to fetching directly
    from MT5 if the DB returns insufficient data.

    Args:
        symbol:    Instrument symbol.
        start:     Backtest start (UTC-aware datetime).
        end:       Backtest end (UTC-aware datetime).
        timeframe: Primary timeframe string.

    Returns:
        Dict mapping timeframe → DataFrame, or None if data cannot be obtained.
    """
    data: dict[str, object] = {}
    timeframes = ["M15", "H1", "H4"]

    # ── Attempt 1: PostgreSQL ─────────────────────────────────────────────
    try:
        from core.data.db_manager import DatabaseManager
        db = DatabaseManager()
        db.connect()

        for tf in timeframes:
            try:
                df = db.get_candles(symbol, tf, start, end)
                if df is not None and len(df) > 0:
                    data[tf] = df
                    log.info("Loaded %d bars from DB: %s %s", len(df), symbol, tf)
            except Exception as exc:
                log.debug("DB fetch skipped for %s %s: %s", symbol, tf, exc)

        db.disconnect()
    except Exception as exc:
        log.warning("DB unavailable: %s — will try MT5 direct", exc)

    # ── Attempt 2: MT5 direct ─────────────────────────────────────────────
    missing = [tf for tf in timeframes if tf not in data or len(data[tf]) < 100]
    if missing:
        try:
            from core.execution.mt5_connector import MT5Connector, MT5ConnectionError
            from core.data.data_pipeline import DataPipeline

            mt5 = MT5Connector()
            login    = int(os.getenv("MT5_LOGIN", "0")) or None
            password = os.getenv("MT5_PASSWORD") or None
            server   = os.getenv("MT5_SERVER")   or None
            path     = os.getenv("MT5_TERMINAL_PATH") or None
            mt5.initialize(path=path, login=login, password=password, server=server)

            if not mt5.is_connected():
                raise MT5ConnectionError("MT5 terminal not connected")

            pipeline = DataPipeline(mt5, None)
            days = max((end - start).days, 1)
            tf_bars = {"M15": days * 96, "H1": days * 24, "H4": days * 6}

            for tf in missing:
                try:
                    bars = min(tf_bars.get(tf, 500), 50_000)
                    df = pipeline.fetch_historical(
                        symbol, tf,
                        start_date=start,
                        bars=bars,
                        store_to_db=False,   # DB may be unavailable
                    )
                    if df is not None and len(df) > 0:
                        # fetch_historical returns a DataFrame with a 'timestamp'
                        # column (not index) — filter by that column
                        if "timestamp" in df.columns:
                            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
                            df = df[mask].reset_index(drop=True)
                        if len(df) > 0:
                            data[tf] = df
                            log.info("Loaded %d bars from MT5: %s %s", len(df), symbol, tf)
                except Exception as fetch_exc:
                    log.warning("MT5 fetch failed %s %s: %s", symbol, tf, fetch_exc)

            mt5.shutdown()
        except Exception as exc:
            log.warning("MT5 unavailable for data fetch: %s", exc)

    if timeframe not in data or len(data[timeframe]) < 200:
        log.error(
            "Insufficient data for %s %s (need ≥200 bars, got %d)",
            symbol,
            timeframe,
            len(data.get(timeframe, [])),
        )
        return None

    return data


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------


def _build_components(symbol: str, initial_balance: float) -> dict:
    """Instantiate all components needed to run the simulation."""
    from core.data.feature_engineer import FeatureEngineer
    from core.regime.regime_detector import RegimeDetector
    from core.signals.momentum_module import MomentumModule
    from core.signals.mean_reversion_module import MeanReversionModule
    from core.signals.breakout_module import BreakoutModule
    from core.signals.signal_router import SignalRouter
    from core.risk import RiskEngine
    from backtesting.simulation_engine import SimulationEngine
    from backtesting.performance_metrics import PerformanceMetrics
    from backtesting.walk_forward import WalkForwardValidator

    feature_eng = FeatureEngineer()
    regime_det  = RegimeDetector()
    momentum    = MomentumModule(symbol)
    mr_mod      = MeanReversionModule(symbol)
    breakout    = BreakoutModule(symbol)
    router      = SignalRouter(momentum, mr_mod, breakout)
    risk_engine = RiskEngine()
    pm          = PerformanceMetrics()

    for mod, name in [
        (momentum, "MomentumModule"),
        (mr_mod,   "MeanReversionModule"),
        (breakout, "BreakoutModule"),
    ]:
        model_path = Path("models/xgboost") / f"{symbol}_{name}.pkl"
        if model_path.exists():
            try:
                mod.load_model(str(model_path))
                log.info("Loaded saved model: %s", model_path)
            except Exception as exc:
                log.warning("Could not load %s: %s", model_path, exc)

    sim_engine = SimulationEngine(
        feature_engineer=feature_eng,
        regime_detector=regime_det,
        signal_router=router,
        risk_engine=risk_engine,
        initial_balance=initial_balance,
    )

    return {
        "feature_eng":  feature_eng,
        "regime_det":   regime_det,
        "momentum":     momentum,
        "mr_mod":       mr_mod,
        "breakout":     breakout,
        "router":       router,
        "risk_engine":  risk_engine,
        "sim_engine":   sim_engine,
        "perf_metrics": pm,
        "validator":    WalkForwardValidator(sim_engine, pm, regime_det),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load data, run walk-forward validation, and write reports."""
    args = _parse_args()

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        log.error("Invalid date format: %s", exc)
        sys.exit(1)

    if start_dt >= end_dt:
        log.error("--start must be before --end")
        sys.exit(1)

    log.info(
        "=== Backtest: %s | %s → %s | folds=%d ===",
        args.symbol, args.start, args.end, args.folds,
    )

    try:
        get_config()
    except Exception as exc:
        log.warning("Config load warning: %s — proceeding with defaults", exc)

    # ── Load data ─────────────────────────────────────────────────────────
    data = _load_data(args.symbol, start_dt, end_dt, args.timeframe)
    if data is None:
        log.critical("Cannot obtain data for %s — aborting", args.symbol)
        sys.exit(1)

    primary_df = data[args.timeframe]
    log.info(
        "Primary data: %d bars (%s → %s)",
        len(primary_df),
        primary_df.index[0] if len(primary_df) > 0 else "?",
        primary_df.index[-1] if len(primary_df) > 0 else "?",
    )

    # ── Build components ──────────────────────────────────────────────────
    comps = _build_components(args.symbol, args.balance)

    # ── Run walk-forward validation ───────────────────────────────────────
    log.info("Running walk-forward validation (%d folds) …", args.folds)
    try:
        wf_results = comps["validator"].run(
            symbol=args.symbol,
            df=primary_df,
            n_folds=args.folds,
            mtf_data=data,
        )
    except Exception as exc:
        log.critical("Walk-forward validation failed: %s", exc, exc_info=True)
        sys.exit(1)

    log.info(
        "Walk-forward complete | %d folds | %d total trades | consistency=%s",
        wf_results.n_folds,
        len(wf_results.combined_trades),
        "OK" if wf_results.fold_consistency_ok else f"FAIL: {wf_results.fold_consistency_reason}",
    )

    # ── Generate report ───────────────────────────────────────────────────
    from backtesting.results_analyzer import ResultsAnalyzer

    output_dir = Path(args.output)
    analyzer   = ResultsAnalyzer(output_dir=str(output_dir))
    report     = analyzer.generate_report(wf_results)

    summary = report.get("summary", {})
    metrics = summary.get("overall_metrics", {})
    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT — {args.symbol}")
    print("=" * 60)
    print(f"  Period        : {args.start}  →  {args.end}")
    print(f"  Folds         : {wf_results.n_folds}")
    print(f"  Total trades  : {summary.get('total_trades', 0)}")
    print(f"  Consistency   : {'OK' if wf_results.fold_consistency_ok else 'FAIL'}")
    if wf_results.fold_consistency_reason:
        print(f"  Reason        : {wf_results.fold_consistency_reason}")
    print("-" * 60)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<25} : {val:.4f}")
        else:
            print(f"  {key:<25} : {val}")
    print("=" * 60 + "\n")

    if not args.no_plots:
        try:
            analyzer.save_plots(wf_results)
            log.info("Charts saved to %s", output_dir)
        except Exception as exc:
            log.warning("Chart generation failed: %s", exc)

    try:
        csv_path = analyzer.export_to_csv(wf_results)
        log.info("Trade log exported: %s", csv_path)
    except Exception as exc:
        log.warning("CSV export failed: %s", exc)

    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"backtest_{args.symbol}_{ts}.json"
        json_path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("JSON report written: %s", json_path)
    except Exception as exc:
        log.warning("JSON report write failed: %s", exc)

    if not wf_results.fold_consistency_ok:
        log.warning("Fold consistency check failed — review before live deployment")
        sys.exit(2)

    log.info("=== Backtest complete ===")


if __name__ == "__main__":
    main()
