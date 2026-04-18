"""Live trading entry point — initialises all layers and starts the main trading loop."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to Python path so all core imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils.config import get_config
from core.utils.logger import get_logger

log = get_logger("run_live")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()
_position_manager = None    # global ref for signal handler
_db_manager       = None


def _shutdown_handler(signum: int, frame) -> None:
    """Handle SIGTERM / SIGINT gracefully."""
    log.info("Shutdown signal %d received — stopping new signal generation", signum)
    _shutdown_event.set()


def main() -> None:
    """Full live trading startup and main event loop.

    Startup sequence
    ----------------
    1.  Load config and verify all required keys are present.
    2.  Connect to PostgreSQL database.
    3.  Verify MT5 bridge is reachable (3 retries, 5 s apart).
    4.  Initialise all system components.
    5.  Fetch historical warmup data and run initial regime detection.
    6.  Start PositionManager background thread.
    7.  Register per-candle callbacks for all assets × M15.
    8.  Spin in the main thread, waiting for the shutdown event.
    """
    global _position_manager, _db_manager

    # ── signals ──────────────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # ── 1. Config ────────────────────────────────────────────────────────────
    log.info("=== Trading System Startup ===")
    cfg = get_config()

    # ── 2. Database ───────────────────────────────────────────────────────────
    from core.data.db_manager import DatabaseManager
    _db_manager = DatabaseManager()
    try:
        _db_manager.connect()
        log.info("PostgreSQL connected")
    except Exception as exc:
        log.error("Database connection failed: %s — continuing without DB persistence", exc)

    # ── 3. MT5 Bridge ─────────────────────────────────────────────────────────
    from core.execution.mt5_client import MT5Client, MT5ConnectionError
    mt5 = MT5Client()
    for attempt in range(1, 4):
        try:
            health = mt5.health_check()
            if health.get("mt5_connected"):
                log.info("MT5 bridge connected: %s", health.get("account_server", ""))
                break
            log.warning("MT5 bridge unhealthy (attempt %d/3): %s", attempt, health)
        except MT5ConnectionError as exc:
            log.warning("MT5 bridge unreachable (attempt %d/3): %s", attempt, exc)
        if attempt < 3:
            time.sleep(5)
    else:
        log.critical("MT5 bridge unavailable after 3 attempts — aborting startup")
        sys.exit(1)

    # ── 4. Component initialisation ───────────────────────────────────────────
    from core.data.data_pipeline    import DataPipeline
    from core.data.feature_engineer import FeatureEngineer
    from core.regime.regime_detector import RegimeDetector
    from core.signals.momentum_module      import MomentumModule
    from core.signals.mean_reversion_module import MeanReversionModule
    from core.signals.breakout_module      import BreakoutModule
    from core.signals.signal_router        import SignalRouter
    from core.risk                          import RiskEngine
    from core.risk.circuit_breakers         import CircuitBreaker
    from core.risk.stop_target_engine       import StopTargetEngine
    from core.execution.pre_execution_validator import PreExecutionValidator
    from core.execution.order_manager           import OrderManager
    from core.execution.fill_monitor            import FillMonitor
    from core.execution.position_manager        import PositionManager
    from core.execution.trade_journal           import TradeJournal
    from core.monitoring.grafana_exporter       import GrafanaExporter
    from core.monitoring.mlflow_tracker         import MLflowTracker

    assets = cfg.assets  # list of asset config nodes

    # One signal module instance per symbol keeps model state separate
    momentum_modules   = {a.symbol: MomentumModule(a.symbol)   for a in assets}
    mr_modules         = {a.symbol: MeanReversionModule(a.symbol) for a in assets}
    breakout_modules   = {a.symbol: BreakoutModule(a.symbol)    for a in assets}

    feature_eng   = FeatureEngineer()
    regime_det    = RegimeDetector()
    circuit_brk   = CircuitBreaker()
    stop_eng      = StopTargetEngine()
    risk_engine   = RiskEngine()
    trade_journal = TradeJournal(db_manager=_db_manager)
    grafana       = GrafanaExporter()
    mlflow_trk    = MLflowTracker()
    data_pipeline = DataPipeline(mt5, _db_manager)

    # Try loading saved champion models
    symbols = [a.symbol for a in assets]
    try:
        regime_det.load_models(symbols)
        log.info("Regime models loaded for: %s", symbols)
    except Exception as exc:
        log.warning("Could not load regime models (%s) — will run without pre-trained models", exc)

    for sym in symbols:
        for mod_dict, ModClass in [
            (momentum_modules,  MomentumModule),
            (mr_modules,        MeanReversionModule),
            (breakout_modules,  BreakoutModule),
        ]:
            model_path = Path("models/xgboost") / f"{sym}_{type(mod_dict[sym]).__name__}.pkl"
            if model_path.exists():
                try:
                    mod_dict[sym].load_model(str(model_path))
                    log.info("Loaded signal model: %s", model_path)
                except Exception as exc:
                    log.warning("Could not load %s: %s", model_path, exc)

    # Position manager (needs circuit_brk, stop_eng, trade_journal, regime_det)
    _position_manager = PositionManager(
        mt5_client=mt5,
        stop_target_engine=stop_eng,
        circuit_breaker=circuit_brk,
        trade_journal=trade_journal,
        regime_detector=regime_det,
    )

    order_mgr   = OrderManager(mt5, trade_journal)
    fill_mon    = FillMonitor(mt5, _position_manager, trade_journal)
    pre_val     = PreExecutionValidator(mt5)

    # ── 5. Historical warmup ──────────────────────────────────────────────────
    log.info("Fetching historical warmup data …")
    warmup_data: dict[str, dict] = {}
    for sym in symbols:
        warmup_data[sym] = {}
        for tf in ["M15", "H1", "H4"]:
            try:
                df = data_pipeline.fetch_latest(sym, tf, count=500)
                warmup_data[sym][tf] = df
            except Exception as exc:
                log.warning("Warmup fetch failed for %s %s: %s", sym, tf, exc)

    # Run initial regime detection to warm state machines
    for sym in symbols:
        if sym in warmup_data and warmup_data[sym]:
            try:
                m15_df = warmup_data[sym].get("M15")
                if m15_df is not None and len(m15_df) > 50:
                    feats = feature_eng.compute(m15_df, symbol=sym)
                    regime_det.detect(sym, warmup_data[sym], feats)
            except Exception as exc:
                log.warning("Warmup regime detection failed for %s: %s", sym, exc)

    # ── 6. Start PositionManager thread ──────────────────────────────────────
    _position_manager.start()
    log.info("PositionManager thread started")

    # ── 7. Log startup event ─────────────────────────────────────────────────
    try:
        _db_manager.log_system_event(
            event_type="startup",
            severity="INFO",
            message=f"Trading system started. Assets: {symbols}",
        )
    except Exception as exc:
        log.warning("Could not log startup event: %s", exc)

    # ── 8. Register candle callbacks ─────────────────────────────────────────
    def _make_callback(symbol: str):
        """Factory to capture symbol in closure."""

        def on_candle_close(sym: str, timeframe: str, candle: dict) -> None:
            """Triggered on every new M15 candle close for *symbol*."""
            if _shutdown_event.is_set():
                return

            log.debug("%s %s candle close — processing", sym, timeframe)

            try:
                # a. Fetch latest bars for all timeframes
                data_dict: dict = {}
                for tf in ["M15", "H1", "H4"]:
                    try:
                        data_dict[tf] = data_pipeline.fetch_latest(sym, tf, count=200)
                    except Exception as fetch_exc:
                        log.warning("Fetch failed %s %s: %s", sym, tf, fetch_exc)

                m15 = data_dict.get("M15")
                if m15 is None or len(m15) < 50:
                    log.debug("%s — insufficient data, skipping", sym)
                    return

                # b. Compute features
                features = feature_eng.compute(m15, mtf_data=data_dict, symbol=sym)

                # c. Regime detection
                regime_state = regime_det.detect(sym, data_dict, features)
                _position_manager.on_candle_close(sym, timeframe, features, regime_state)

                # d. Circuit breaker check
                try:
                    account_info = mt5.account_info()
                except MT5ConnectionError:
                    account_info = {}

                account_state = {
                    "balance":         account_info.get("balance",    100_000),
                    "equity":          account_info.get("equity",     100_000),
                    "daily_pnl_pct":   0.0,
                    "weekly_pnl_pct":  0.0,
                    "consecutive_losses": 0,
                    "global_risk_state": regime_state.global_risk_state,
                    "avg_spread_ratio":  1.0,
                    "mt5_connected":     True,
                    "open_positions":    _position_manager.get_open_positions(),
                    "timestamp":         datetime.now(timezone.utc),
                }

                cb_level, cb_desc = circuit_brk.check(account_state, [], account_state["open_positions"])
                if circuit_brk.is_trading_halted(cb_level):
                    log.info("%s CB level %d — trading halted: %s", sym, cb_level, cb_desc)
                    grafana.export_account(account_state)
                    return

                # e. Signal router
                momentum_mod = momentum_modules.get(sym, MomentumModule(sym))
                mr_mod       = mr_modules.get(sym, MeanReversionModule(sym))
                bo_mod       = breakout_modules.get(sym, BreakoutModule(sym))
                router       = SignalRouter(momentum_mod, mr_mod, bo_mod)

                tick = {}
                try:
                    tick = mt5.symbol_info_tick(sym)
                except MT5ConnectionError:
                    pass

                latest_bar = {
                    "bid":      tick.get("bid",   float(m15["close"].iloc[-1])),
                    "ask":      tick.get("ask",   float(m15["close"].iloc[-1])),
                    "close":    float(m15["close"].iloc[-1]),
                    "high":     float(m15["high"].iloc[-1]),
                    "low":      float(m15["low"].iloc[-1]),
                    "atr":      float(features["atr_14"].iloc[-1]) if "atr_14" in features.columns else 0.001,
                    "pip_size": 0.01 if "XAU" in sym else (1.0 if sym in ("US30", "NAS100") else 0.0001),
                }

                signal = router.route(sym, timeframe, features, regime_state, latest_bar)
                if signal is None:
                    grafana.export_regime_state(sym, regime_state)
                    grafana.export_account(account_state)
                    return

                # f. Risk engine
                trade_order = risk_engine.process(signal, account_state, features,
                                                   account_state["open_positions"])
                if trade_order is None:
                    grafana.export_regime_state(sym, regime_state)
                    return

                # g. Pre-execution validation
                valid, failures = pre_val.validate(trade_order, regime_state)
                if not valid:
                    log.info("%s pre-exec failed: %s", sym, failures)
                    return

                # h. Place order
                ticket: Optional[int] = None
                if trade_order.module == "BREAKOUT":
                    ticket = order_mgr.place_stop_limit_order(trade_order)
                elif trade_order.module == "MEAN_REVERSION":
                    ticket = order_mgr.place_limit_order(trade_order)
                else:
                    ticket = order_mgr.place_limit_order(trade_order)

                if ticket is not None:
                    log.info("%s order placed ticket=%d lot=%.2f %s via %s",
                             sym, ticket, trade_order.lot_size,
                             trade_order.direction, trade_order.module)
                    # Monitor fill in background thread
                    threading.Thread(
                        target=_monitor_fill,
                        args=(ticket, trade_order, order_mgr, fill_mon),
                        daemon=True,
                    ).start()

                # i. Export to Redis
                grafana.export_regime_state(sym, regime_state)
                grafana.export_signal(signal)
                grafana.export_account(account_state)

            except Exception as exc:
                log.error("Candle callback error for %s: %s", sym, exc, exc_info=True)

        return on_candle_close

    # Register callbacks for every asset on M15
    for sym in symbols:
        callback = _make_callback(sym)
        cb_thread = threading.Thread(
            target=data_pipeline.on_new_candle,
            args=(sym, "M15", callback),
            daemon=True,
            name=f"candle_cb_{sym}",
        )
        cb_thread.start()
        log.info("Candle callback registered for %s M15", sym)

    log.info("=== System fully operational. Waiting for candles … ===")

    # ── Main thread: wait for shutdown ──────────────────────────────────────
    while not _shutdown_event.is_set():
        time.sleep(1)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    log.info("Shutting down …")
    if _position_manager is not None:
        _position_manager.stop()

    try:
        _db_manager.log_system_event("shutdown", "INFO", "Trading system stopped cleanly")
        _db_manager.disconnect()
    except Exception as exc:
        log.warning("Shutdown DB cleanup error: %s", exc)

    log.info("=== Trading System stopped ===")


def _monitor_fill(ticket: int, trade_order, order_mgr, fill_mon) -> None:
    """Background thread: monitor a pending order for fill/expiry."""
    result = order_mgr.monitor_pending(trade_order, ticket)
    if result == "filled":
        fill_mon.on_fill(ticket, trade_order)
    elif result in ("expired", "cancelled"):
        log.info("Order %d %s", ticket, result)
    else:
        log.warning("Order %d monitor returned: %s", ticket, result)


if __name__ == "__main__":
    main()
