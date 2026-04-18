"""Monitoring and retraining daemon — scheduled drift detection and model retraining pipeline.

Run from the project root on Windows:
    python run_monitor.py

MetaTrader5 is used directly (no HTTP bridge).  The MT5 terminal must be
open and logged in.  This daemon can run alongside run_live.py.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add project root to Python path so all core imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.utils.config import get_config
from core.utils.logger import get_logger

log = get_logger("run_monitor")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()

_counter_lock    = threading.Lock()
_candle_count    = 0
_last_retrain_dt: Optional[datetime] = None


def _shutdown_handler(signum: int, frame) -> None:
    log.info("Shutdown signal %d received — stopping monitor daemon", signum)
    _shutdown_event.set()


# ---------------------------------------------------------------------------
# Retraining pipeline
# ---------------------------------------------------------------------------


def _full_retrain_pipeline(
    symbol:        str,
    data_pipeline,
    feature_eng,
    regime_det,
    momentum_mod,
    mr_mod,
    breakout_mod,
    db_manager,
    drift_detector,
    perf_monitor,
    mlflow_trk,
) -> None:
    """Ten-step full model retraining pipeline."""
    global _last_retrain_dt

    log.info("=== Starting full retrain pipeline for %s ===", symbol)
    start_pipeline = datetime.now(timezone.utc)

    # Step 1: Fetch training data
    log.info("[1/10] Fetching 3-month training data for %s …", symbol)
    training_data: dict[str, object] = {}
    for tf in ["M15", "H1", "H4"]:
        try:
            df = data_pipeline.fetch_latest(symbol, tf, count=13_000)
            if df is not None and len(df) > 200:
                training_data[tf] = df
                log.info("  %s %s: %d bars", symbol, tf, len(df))
        except Exception as exc:
            log.warning("  Fetch failed %s %s: %s", symbol, tf, exc)

    if "M15" not in training_data or len(training_data["M15"]) < 500:
        log.error("[1/10] Insufficient M15 data for %s — aborting retrain", symbol)
        return

    m15_df = training_data["M15"]

    # Step 2: Compute features
    log.info("[2/10] Computing features …")
    try:
        features = feature_eng.compute(m15_df, mtf_data=training_data, symbol=symbol)
    except Exception as exc:
        log.error("[2/10] Feature engineering failed: %s", exc)
        return

    if features is None or len(features) < 200:
        log.error("[2/10] Insufficient features — aborting retrain")
        return

    # Step 3: Regime HMM retraining
    log.info("[3/10] Training HMM regime model …")
    try:
        regime_det.train(symbol, training_data, features)
        log.info("[3/10] HMM training complete for %s", symbol)
    except Exception as exc:
        log.warning("[3/10] Regime HMM retrain failed: %s — continuing", exc)

    # Step 4: Label generation
    log.info("[4/10] Generating training labels …")
    try:
        mom_labels = momentum_mod.generate_labels(m15_df, features, timeframe="M15")
        mr_labels  = mr_mod.generate_labels(m15_df, features)
        bo_labels  = breakout_mod.generate_labels(m15_df, features)
    except Exception as exc:
        log.error("[4/10] Label generation failed: %s — aborting retrain", exc)
        return

    # Step 5–7: Train signal models
    for step, mod, labels, name in [
        (5, momentum_mod,  mom_labels, "Momentum"),
        (6, mr_mod,        mr_labels,  "Mean-Reversion"),
        (7, breakout_mod,  bo_labels,  "Breakout"),
    ]:
        log.info("[%d/10] Training %s XGBoost …", step, name)
        try:
            mod.train(features, labels)
            log.info("[%d/10] %s model trained", step, name)
        except Exception as exc:
            log.error("[%d/10] %s training failed: %s", step, name, exc)

    # Step 8: Walk-forward validation
    log.info("[8/10] Running 4-fold walk-forward validation …")
    wf_ok      = True
    sharpe_val = 0.0
    try:
        from core.risk import RiskEngine
        from core.signals.signal_router import SignalRouter
        from backtesting.simulation_engine import SimulationEngine
        from backtesting.performance_metrics import PerformanceMetrics
        from backtesting.walk_forward import WalkForwardValidator

        router     = SignalRouter(momentum_mod, mr_mod, breakout_mod)
        sim_engine = SimulationEngine(
            feature_engineer=feature_eng,
            regime_detector=regime_det,
            signal_router=router,
            risk_engine=RiskEngine(),
            initial_balance=100_000.0,
        )
        pm        = PerformanceMetrics()
        validator = WalkForwardValidator(sim_engine, pm, regime_det)

        wf_results = validator.run(
            symbol=symbol, df=m15_df, n_folds=4, mtf_data=training_data
        )
        sharpe_val = wf_results.overall_metrics.get("sharpe_ratio", 0.0)
        wf_ok      = wf_results.fold_consistency_ok
        log.info(
            "[8/10] WF results — sharpe=%.3f consistency=%s",
            sharpe_val,
            "OK" if wf_ok else f"FAIL: {wf_results.fold_consistency_reason}",
        )
    except Exception as exc:
        log.warning("[8/10] Walk-forward validation error: %s — proceeding", exc)

    # Step 9: MLflow logging
    log.info("[9/10] Logging to MLflow …")
    try:
        mlflow_trk.log_run(
            {"symbol": symbol, "train_bars": len(m15_df), "retrain_ts": start_pipeline.isoformat()},
            {"sharpe_ratio": sharpe_val, "fold_consistency": 1.0 if wf_ok else 0.0},
            stage="STAGING",
        )
        if sharpe_val >= 0.50 and wf_ok:
            mlflow_trk.promote_champion(symbol)
            log.info("[9/10] Models promoted to CHAMPION for %s (sharpe=%.3f)", symbol, sharpe_val)
        else:
            log.info("[9/10] Models remain in STAGING (sharpe=%.3f, wf_ok=%s)", sharpe_val, wf_ok)
    except Exception as exc:
        log.warning("[9/10] MLflow logging failed: %s — continuing", exc)

    # Step 10: Save models to disk
    log.info("[10/10] Saving models to disk …")
    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for mod, name in [
        (momentum_mod,  "MomentumModule"),
        (mr_mod,        "MeanReversionModule"),
        (breakout_mod,  "BreakoutModule"),
    ]:
        path = model_dir / f"{symbol}_{name}.pkl"
        try:
            mod.save_model(str(path))
            saved.append(name)
        except Exception as exc:
            log.warning("[10/10] Could not save %s: %s", name, exc)

    log.info("[10/10] Saved: %s", saved)

    with _counter_lock:
        _last_retrain_dt = datetime.now(timezone.utc)

    elapsed = (datetime.now(timezone.utc) - start_pipeline).total_seconds()
    log.info("=== Retrain pipeline complete for %s in %.1f s ===", symbol, elapsed)

    try:
        db_manager.log_system_event(
            event_type="retrain",
            severity="INFO",
            message=f"Retrain complete: {symbol} sharpe={sharpe_val:.3f} wf_ok={wf_ok}",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Monitoring jobs
# ---------------------------------------------------------------------------


def _job_candle_monitoring(
    symbol: str,
    data_pipeline,
    feature_eng,
    regime_det,
    drift_detector,
    perf_monitor,
    db_manager,
    grafana,
) -> None:
    global _candle_count

    with _counter_lock:
        _candle_count += 1
        count_snapshot = _candle_count

    try:
        df = data_pipeline.fetch_latest(symbol, "M15", count=200)
        if df is None or len(df) < 50:
            return

        features = feature_eng.compute(df, symbol=symbol)

        try:
            data_dict = {"M15": df}
            regime_state = regime_det.detect(symbol, data_dict, features)
            grafana.export_regime_state(symbol, regime_state)
        except Exception as exc:
            log.debug("Regime export error %s: %s", symbol, exc)

        if count_snapshot % 500 == 0:
            log.info("[DRIFT] 500-candle checkpoint for %s (candle #%d)", symbol, count_snapshot)
            _job_drift_check(
                symbol=symbol,
                features=features,
                drift_detector=drift_detector,
                db_manager=db_manager,
            )

    except Exception as exc:
        log.warning("Candle monitoring error %s: %s", symbol, exc)


def _job_drift_check(
    symbol: str,
    features,
    drift_detector,
    db_manager,
) -> bool:
    import numpy as np

    log.info("[DRIFT] Running PSI drift check for %s …", symbol)
    try:
        baseline_df = None
        try:
            end   = datetime.now(timezone.utc) - timedelta(days=30)
            start = end - timedelta(days=30)
            baseline_df = db_manager.get_features(symbol, "M15", start, end)
        except Exception as exc:
            log.debug("Baseline fetch failed: %s", exc)

        if baseline_df is None or len(baseline_df) < 50:
            log.debug("[DRIFT] No baseline for %s — skipping PSI", symbol)
            return False

        numeric_cols = [
            c for c in features.select_dtypes(include=[np.number]).columns
            if c in baseline_df.columns
        ]
        psi_scores: dict[str, float] = {}

        for col in numeric_cols:
            try:
                expected = baseline_df[col].dropna().values
                actual   = features[col].dropna().values
                if len(expected) > 10 and len(actual) > 10:
                    psi_scores[col] = drift_detector.compute_psi(expected, actual)
            except Exception:
                pass

        trigger = drift_detector.should_retrain(psi_scores, regime_psi=0.0)
        if trigger:
            log.warning("[DRIFT] Retrain triggered for %s — PSI drift detected", symbol)
            return True

        critical_count = sum(1 for v in psi_scores.values() if v > 0.25)
        log.info("[DRIFT] %s PSI check — critical features: %d/%d", symbol, critical_count, len(psi_scores))

    except Exception as exc:
        log.warning("[DRIFT] Drift check error for %s: %s", symbol, exc)

    return False


def _job_session_close(symbols, data_pipeline, feature_eng, regime_det, drift_detector,
                       perf_monitor, db_manager) -> None:
    log.info("[SESSION CLOSE] Running daily performance review …")
    for symbol in symbols:
        try:
            live_metrics     = db_manager.get_daily_performance(symbol)     or {}
            baseline_metrics = db_manager.get_baseline_performance(symbol) or {}
            if live_metrics and baseline_metrics:
                alerts = perf_monitor.check(live_metrics, baseline_metrics)
                for alert in alerts:
                    log.warning("[ALERT] %s %s — %s: drift=%.1f%%",
                                symbol, alert.level, alert.metric, alert.drift_pct)
                if any(a.level == "RETRAIN" for a in alerts):
                    log.warning("[SESSION CLOSE] RETRAIN triggered for %s via performance", symbol)
                    db_manager.log_system_event(
                        "performance_retrain_trigger", "WARNING",
                        f"{symbol}: performance retrain triggered",
                    )
        except Exception as exc:
            log.warning("[SESSION CLOSE] Review error for %s: %s", symbol, exc)
    log.info("[SESSION CLOSE] Daily review complete")


def _job_weekly_retrain(symbols: list[str], **pipeline_kwargs) -> None:
    log.info("[WEEKLY] Starting weekly retrain for all symbols: %s", symbols)
    for symbol in symbols:
        if _shutdown_event.is_set():
            break
        _full_retrain_pipeline(symbol=symbol, **pipeline_kwargs)
    log.info("[WEEKLY] Weekly retrain complete for all symbols")


# ---------------------------------------------------------------------------
# Scheduler thread
# ---------------------------------------------------------------------------


def _scheduler_loop(
    symbols: list[str],
    pipeline_kwargs: dict,
    session_close_hour_utc: int = 22,
) -> None:
    last_session_close_date: Optional[datetime] = None
    last_weekly_date:        Optional[datetime] = None

    log.info("[SCHEDULER] Scheduler thread started")

    while not _shutdown_event.is_set():
        now = datetime.now(timezone.utc)

        if (
            now.hour == session_close_hour_utc
            and now.minute < 5
            and (last_session_close_date is None or last_session_close_date.date() < now.date())
        ):
            last_session_close_date = now
            try:
                _job_session_close(
                    symbols=symbols,
                    data_pipeline=pipeline_kwargs["data_pipeline"],
                    feature_eng=pipeline_kwargs["feature_eng"],
                    regime_det=pipeline_kwargs["regime_det"],
                    drift_detector=pipeline_kwargs["drift_detector"],
                    perf_monitor=pipeline_kwargs["perf_monitor"],
                    db_manager=pipeline_kwargs["db_manager"],
                )
            except Exception as exc:
                log.error("[SCHEDULER] Session close job failed: %s", exc, exc_info=True)

        if (
            now.weekday() == 6
            and now.hour == 1
            and now.minute < 5
            and (last_weekly_date is None or last_weekly_date.date() < now.date())
        ):
            last_weekly_date = now
            try:
                kw = {k: v for k, v in pipeline_kwargs.items() if k != "data_pipeline"}
                _job_weekly_retrain(
                    symbols=symbols,
                    data_pipeline=pipeline_kwargs["data_pipeline"],
                    **kw,
                )
            except Exception as exc:
                log.error("[SCHEDULER] Weekly retrain failed: %s", exc, exc_info=True)

        _shutdown_event.wait(timeout=60)

    log.info("[SCHEDULER] Scheduler thread stopped")


# ---------------------------------------------------------------------------
# Candle polling threads
# ---------------------------------------------------------------------------


def _candle_poll_loop(
    symbol: str,
    data_pipeline,
    feature_eng,
    regime_det,
    drift_detector,
    perf_monitor,
    db_manager,
    grafana,
    pipeline_kwargs: dict,
) -> None:
    log.info("[CANDLE] Poll thread started for %s", symbol)
    last_candle_time = None

    while not _shutdown_event.is_set():
        try:
            df = data_pipeline.fetch_latest(symbol, "M15", count=5)
            if df is not None and len(df) > 0:
                latest_ts = df.index[-1]
                if last_candle_time is None or latest_ts != last_candle_time:
                    last_candle_time = latest_ts

                    retrain_needed = False
                    try:
                        features = feature_eng.compute(df, symbol=symbol)
                    except Exception:
                        features = None

                    if features is not None:
                        _job_candle_monitoring(
                            symbol=symbol,
                            data_pipeline=data_pipeline,
                            feature_eng=feature_eng,
                            regime_det=regime_det,
                            drift_detector=drift_detector,
                            perf_monitor=perf_monitor,
                            db_manager=db_manager,
                            grafana=grafana,
                        )

                        with _counter_lock:
                            count_snapshot = _candle_count
                        if count_snapshot > 0 and count_snapshot % 500 == 0:
                            retrain_needed = _job_drift_check(
                                symbol=symbol,
                                features=features,
                                drift_detector=drift_detector,
                                db_manager=db_manager,
                            )

                    if retrain_needed:
                        log.warning("[CANDLE] Retrain triggered for %s — starting pipeline", symbol)
                        threading.Thread(
                            target=_full_retrain_pipeline,
                            kwargs={"symbol": symbol, **pipeline_kwargs},
                            daemon=True,
                            name=f"retrain_{symbol}",
                        ).start()

        except Exception as exc:
            log.warning("[CANDLE] Poll error for %s: %s", symbol, exc)

        _shutdown_event.wait(timeout=15)

    log.info("[CANDLE] Poll thread stopped for %s", symbol)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Initialise all components and start the monitoring daemon."""
    signal.signal(signal.SIGINT,  _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    log.info("=== Monitor Daemon Startup ===")

    # ── 1. Config ─────────────────────────────────────────────────────────
    try:
        cfg = get_config()
    except Exception as exc:
        log.warning("Config load warning: %s — using defaults", exc)
        cfg = None

    # ── 2. Database ───────────────────────────────────────────────────────
    from core.data.db_manager import DatabaseManager
    db_manager = DatabaseManager()
    try:
        db_manager.connect()
        log.info("PostgreSQL connected")
    except Exception as exc:
        log.warning("DB unavailable: %s — monitoring without persistence", exc)

    # ── 3. MT5 Direct connection ──────────────────────────────────────────
    from core.execution.mt5_connector import MT5Connector, MT5ConnectionError

    mt5 = MT5Connector()
    try:
        login    = int(os.getenv("MT5_LOGIN", "0")) or None
        password = os.getenv("MT5_PASSWORD") or None
        server   = os.getenv("MT5_SERVER")   or None
        path     = os.getenv("MT5_TERMINAL_PATH") or None
        mt5.initialize(path=path, login=login, password=password, server=server)

        if mt5.is_connected():
            log.info("MT5 connected: %s", mt5.health_check().get("account_server", ""))
        else:
            log.warning("MT5 terminal not connected — data fetch may fail")
    except MT5ConnectionError as exc:
        log.warning("MT5 initialisation failed: %s — continuing", exc)

    # ── 4. Core components ────────────────────────────────────────────────
    from core.data.data_pipeline import DataPipeline
    from core.data.feature_engineer import FeatureEngineer
    from core.regime.regime_detector import RegimeDetector
    from core.signals.momentum_module import MomentumModule
    from core.signals.mean_reversion_module import MeanReversionModule
    from core.signals.breakout_module import BreakoutModule
    from core.monitoring.drift_detector import DriftDetector
    from core.monitoring.performance_monitor import PerformanceMonitor
    from core.monitoring.grafana_exporter import GrafanaExporter
    from core.monitoring.mlflow_tracker import MLflowTracker

    assets  = cfg.assets if cfg is not None else []
    symbols = [a.symbol for a in assets] if assets else [
        "XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "XAGUSD", "US30", "NAS100"
    ]

    data_pipeline  = DataPipeline(mt5, db_manager)
    feature_eng    = FeatureEngineer()
    regime_det     = RegimeDetector()
    drift_detector = DriftDetector()
    perf_monitor   = PerformanceMonitor()
    grafana        = GrafanaExporter()
    mlflow_trk     = MLflowTracker()

    momentum_mods: dict[str, MomentumModule]      = {}
    mr_mods:       dict[str, MeanReversionModule] = {}
    breakout_mods: dict[str, BreakoutModule]      = {}

    for sym in symbols:
        momentum_mods[sym] = MomentumModule(sym)
        mr_mods[sym]       = MeanReversionModule(sym)
        breakout_mods[sym] = BreakoutModule(sym)

    # ── 5. Load champion models ───────────────────────────────────────────
    log.info("Loading champion models …")
    try:
        regime_det.load_models(symbols)
    except Exception as exc:
        log.warning("Could not load regime models: %s", exc)

    for sym in symbols:
        for mod, name in [
            (momentum_mods[sym],  "MomentumModule"),
            (mr_mods[sym],        "MeanReversionModule"),
            (breakout_mods[sym],  "BreakoutModule"),
        ]:
            path = Path("models/xgboost") / f"{sym}_{name}.pkl"
            if path.exists():
                try:
                    mod.load_model(str(path))
                    log.info("Loaded %s for %s", name, sym)
                except Exception as exc:
                    log.warning("Could not load %s for %s: %s", name, sym, exc)

    # ── 6. Log startup ────────────────────────────────────────────────────
    try:
        db_manager.log_system_event(
            "monitor_startup", "INFO", f"Monitor daemon started. Watching: {symbols}"
        )
    except Exception:
        pass

    # ── 7. Shared pipeline kwargs ─────────────────────────────────────────
    pipeline_kwargs: dict = {
        "data_pipeline":  data_pipeline,
        "feature_eng":    feature_eng,
        "regime_det":     regime_det,
        "drift_detector": drift_detector,
        "perf_monitor":   perf_monitor,
        "db_manager":     db_manager,
        "mlflow_trk":     mlflow_trk,
    }

    scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        kwargs={
            "symbols":         symbols,
            "pipeline_kwargs": {**pipeline_kwargs, "grafana": grafana},
        },
        daemon=True,
        name="monitor_scheduler",
    )
    scheduler_thread.start()
    log.info("Scheduler thread started")

    # ── 8. Per-symbol candle polling threads ──────────────────────────────
    for sym in symbols:
        sym_kwargs = {
            **pipeline_kwargs,
            "grafana":      grafana,
            "momentum_mod": momentum_mods[sym],
            "mr_mod":       mr_mods[sym],
            "breakout_mod": breakout_mods[sym],
        }
        t = threading.Thread(
            target=_candle_poll_loop,
            kwargs={
                "symbol":          sym,
                "data_pipeline":   data_pipeline,
                "feature_eng":     feature_eng,
                "regime_det":      regime_det,
                "drift_detector":  drift_detector,
                "perf_monitor":    perf_monitor,
                "db_manager":      db_manager,
                "grafana":         grafana,
                "pipeline_kwargs": sym_kwargs,
            },
            daemon=True,
            name=f"monitor_candle_{sym}",
        )
        t.start()
        log.info("Candle poll thread started for %s", sym)

    log.info("=== Monitor daemon fully operational ===")

    while not _shutdown_event.is_set():
        time.sleep(5)

    # ── Graceful shutdown ─────────────────────────────────────────────────
    log.info("Shutting down monitor daemon …")
    _shutdown_event.set()
    time.sleep(3)

    mt5.shutdown()

    try:
        db_manager.log_system_event("monitor_shutdown", "INFO", "Monitor daemon stopped cleanly")
        db_manager.disconnect()
    except Exception as exc:
        log.warning("DB shutdown error: %s", exc)

    log.info("=== Monitor daemon stopped ===")


if __name__ == "__main__":
    main()
