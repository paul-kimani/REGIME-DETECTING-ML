-- =============================================================================
-- Trading System — PostgreSQL Schema
-- =============================================================================
-- All tables use UTC timestamps.
-- Partitioned tables use declarative partitioning by symbol for performance.
-- Run this script once against a fresh database:
--   psql -U trading_user -d trading -f schema.sql
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for text search on reason codes

-- =============================================================================
-- CANDLES — OHLCV per symbol per timeframe
-- Partitioned by symbol for read performance on large historical pulls.
-- =============================================================================
CREATE TABLE IF NOT EXISTS candles (
    id              BIGSERIAL,
    symbol          VARCHAR(20)     NOT NULL,
    timeframe       VARCHAR(5)      NOT NULL,   -- M15, H1, H4
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL DEFAULT 0,
    spread          DOUBLE PRECISION,           -- broker spread at bar open
    inserted_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, symbol)
) PARTITION BY LIST (symbol);

-- Create partitions for each tracked asset
CREATE TABLE IF NOT EXISTS candles_xauusd PARTITION OF candles FOR VALUES IN ('XAUUSD');
CREATE TABLE IF NOT EXISTS candles_xagusd PARTITION OF candles FOR VALUES IN ('XAGUSD');
CREATE TABLE IF NOT EXISTS candles_eurusd PARTITION OF candles FOR VALUES IN ('EURUSD');
CREATE TABLE IF NOT EXISTS candles_gbpusd PARTITION OF candles FOR VALUES IN ('GBPUSD');
CREATE TABLE IF NOT EXISTS candles_usdjpy PARTITION OF candles FOR VALUES IN ('USDJPY');
CREATE TABLE IF NOT EXISTS candles_us30   PARTITION OF candles FOR VALUES IN ('US30');
CREATE TABLE IF NOT EXISTS candles_nas100 PARTITION OF candles FOR VALUES IN ('NAS100');
CREATE TABLE IF NOT EXISTS candles_other  PARTITION OF candles DEFAULT;

-- Unique constraint to prevent duplicate bars
CREATE UNIQUE INDEX IF NOT EXISTS uidx_candles
    ON candles (symbol, timeframe, timestamp);

-- Index for time-range queries
CREATE INDEX IF NOT EXISTS idx_candles_ts
    ON candles (symbol, timeframe, timestamp DESC);

-- =============================================================================
-- FEATURES — Computed feature snapshots per candle
-- Used for model retraining and post-hoc analysis.
-- =============================================================================
CREATE TABLE IF NOT EXISTS features (
    id              BIGSERIAL       PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL,
    timeframe       VARCHAR(5)      NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL,
    feature_data    JSONB           NOT NULL,   -- full feature dict as JSON
    inserted_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_features
    ON features (symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_features_ts
    ON features (symbol, timeframe, timestamp DESC);

-- =============================================================================
-- REGIME STATES — Regime label per candle per symbol
-- =============================================================================
CREATE TABLE IF NOT EXISTS regime_states (
    id                      BIGSERIAL       PRIMARY KEY,
    symbol                  VARCHAR(20)     NOT NULL,
    timeframe               VARCHAR(5)      NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,

    -- Per-timeframe labels
    h4_regime               VARCHAR(20),
    h4_confidence           DOUBLE PRECISION,
    h4_hmm_probs            JSONB,          -- {TREND_UP: 0.x, ...}

    h1_regime               VARCHAR(20),
    h1_confidence           DOUBLE PRECISION,
    h1_hmm_probs            JSONB,

    m15_regime              VARCHAR(20),
    m15_confidence          DOUBLE PRECISION,
    m15_hmm_probs           JSONB,

    -- MTF alignment
    alignment_score         DOUBLE PRECISION,
    active_strategy         VARCHAR(30),
    sizing_multiplier       DOUBLE PRECISION,

    -- Universal macro state
    global_risk_state       VARCHAR(20),
    global_multiplier       DOUBLE PRECISION,

    -- Regime age
    regime_age_candles      INTEGER,
    regime_maturity_flag    VARCHAR(20),    -- young / mature / extended / very_extended

    inserted_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_regime_states
    ON regime_states (symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_regime_states_ts
    ON regime_states (symbol, timeframe, timestamp DESC);

-- =============================================================================
-- SIGNALS — Every signal generated (including those rejected by risk engine)
-- =============================================================================
CREATE TABLE IF NOT EXISTS signals (
    id                      BIGSERIAL       PRIMARY KEY,
    signal_uuid             UUID            NOT NULL DEFAULT uuid_generate_v4(),
    magic_number            BIGINT          NOT NULL,

    symbol                  VARCHAR(20)     NOT NULL,
    timeframe               VARCHAR(5)      NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Signal direction and module
    direction               VARCHAR(10)     NOT NULL,   -- LONG / SHORT / NO_TRADE
    module                  VARCHAR(30)     NOT NULL,   -- MOMENTUM / MEAN_REVERSION / BREAKOUT
    confidence              DOUBLE PRECISION NOT NULL,

    -- Proposed prices
    entry_price             DOUBLE PRECISION,
    stop_loss               DOUBLE PRECISION,
    take_profit_1           DOUBLE PRECISION,
    take_profit_2           DOUBLE PRECISION,
    atr                     DOUBLE PRECISION,
    rr_ratio                DOUBLE PRECISION,

    -- Regime context at signal time
    regime_context          JSONB,

    -- Outcome
    was_executed            BOOLEAN         NOT NULL DEFAULT FALSE,
    rejection_reasons       TEXT[],         -- list of failed check reasons
    order_ticket            BIGINT,

    inserted_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
    ON signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_magic
    ON signals (magic_number);

-- =============================================================================
-- TRADES — Complete lifecycle record for every executed trade
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    id                      BIGSERIAL       PRIMARY KEY,
    trade_uuid              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    ticket                  BIGINT          NOT NULL UNIQUE,
    signal_id               BIGINT          REFERENCES signals(id),
    magic_number            BIGINT          NOT NULL,

    symbol                  VARCHAR(20)     NOT NULL,
    timeframe               VARCHAR(5)      NOT NULL,

    -- Entry
    direction               VARCHAR(10)     NOT NULL,   -- LONG / SHORT
    module                  VARCHAR(30)     NOT NULL,
    entry_time              TIMESTAMPTZ,
    entry_price             DOUBLE PRECISION,
    entry_price_requested   DOUBLE PRECISION,
    slippage_pips           DOUBLE PRECISION,
    lot_size                DOUBLE PRECISION NOT NULL,

    -- Stops and targets (as placed, not as managed)
    stop_loss_initial       DOUBLE PRECISION NOT NULL,
    take_profit_1           DOUBLE PRECISION NOT NULL,
    take_profit_2           DOUBLE PRECISION NOT NULL,
    stop_distance_pips      DOUBLE PRECISION,
    rr_ratio_planned        DOUBLE PRECISION,

    -- Sizing breakdown
    account_balance_at_entry DOUBLE PRECISION,
    base_risk_pct           DOUBLE PRECISION,
    kelly_size_multiplier   DOUBLE PRECISION,
    volatility_scalar       DOUBLE PRECISION,
    regime_age_multiplier   DOUBLE PRECISION,
    alignment_multiplier    DOUBLE PRECISION,
    correlation_multiplier  DOUBLE PRECISION,
    global_multiplier       DOUBLE PRECISION,
    final_risk_pct          DOUBLE PRECISION,
    risk_amount_currency    DOUBLE PRECISION,

    -- Regime at entry
    h4_regime_at_entry      VARCHAR(20),
    h1_regime_at_entry      VARCHAR(20),
    m15_regime_at_entry     VARCHAR(20),
    alignment_score_at_entry DOUBLE PRECISION,
    global_risk_state_at_entry VARCHAR(20),
    regime_age_at_entry     INTEGER,

    -- Exit
    exit_time               TIMESTAMPTZ,
    exit_price              DOUBLE PRECISION,
    exit_reason             VARCHAR(50),    -- stop_hit / tp1_hit / tp2_hit / regime_shift / time_expiry / manual / circuit_breaker
    partial_close_1_time    TIMESTAMPTZ,
    partial_close_1_price   DOUBLE PRECISION,
    partial_close_1_volume  DOUBLE PRECISION,

    -- Performance
    pnl_currency            DOUBLE PRECISION,
    pnl_percent             DOUBLE PRECISION,
    r_multiple              DOUBLE PRECISION,
    mae                     DOUBLE PRECISION,   -- max adverse excursion (pips)
    mfe                     DOUBLE PRECISION,   -- max favourable excursion (pips)
    hold_time_candles       INTEGER,
    commission              DOUBLE PRECISION,

    -- Regime at exit
    h4_regime_at_exit       VARCHAR(20),
    h1_regime_at_exit       VARCHAR(20),
    m15_regime_at_exit      VARCHAR(20),

    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'PENDING',  -- PENDING / OPEN / CLOSED / CANCELLED
    inserted_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry
    ON trades (symbol, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status
    ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time
    ON trades (exit_time DESC) WHERE exit_time IS NOT NULL;

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- POSITION EVENTS — Every position manager lifecycle event
-- =============================================================================
CREATE TABLE IF NOT EXISTS position_events (
    id                  BIGSERIAL       PRIMARY KEY,
    ticket              BIGINT          NOT NULL,
    event_type          VARCHAR(50)     NOT NULL,
    -- Types: trail_update / partial_close / stop_to_breakeven /
    --        tp1_hit / tp2_hit / stop_hit / regime_invalidation /
    --        time_expiry / circuit_breaker / emergency_close
    timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    current_price       DOUBLE PRECISION,
    new_stop_loss       DOUBLE PRECISION,
    old_stop_loss       DOUBLE PRECISION,
    volume_closed       DOUBLE PRECISION,
    close_price         DOUBLE PRECISION,
    atr_at_event        DOUBLE PRECISION,
    regime_at_event     VARCHAR(20),
    details             JSONB,
    inserted_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_position_events_ticket
    ON position_events (ticket, timestamp DESC);

-- =============================================================================
-- MODEL PERFORMANCE — Rolling metrics per model version per asset
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id                      BIGSERIAL       PRIMARY KEY,
    model_type              VARCHAR(30)     NOT NULL,   -- HMM / XGB_REGIME / XGB_MOMENTUM / LSTM / XGB_MR / XGB_BREAKOUT
    model_version           VARCHAR(50)     NOT NULL,
    symbol                  VARCHAR(20)     NOT NULL,
    timeframe               VARCHAR(5),
    evaluation_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    window_trades           INTEGER,

    -- Classification metrics (regime/signal models)
    accuracy                DOUBLE PRECISION,
    f1_macro                DOUBLE PRECISION,
    log_loss                DOUBLE PRECISION,

    -- Trading performance metrics (signal models)
    win_rate                DOUBLE PRECISION,
    avg_r_multiple          DOUBLE PRECISION,
    profit_factor           DOUBLE PRECISION,
    sharpe_ratio            DOUBLE PRECISION,
    max_drawdown            DOUBLE PRECISION,
    expectancy              DOUBLE PRECISION,

    -- Drift indicators
    psi_max                 DOUBLE PRECISION,
    feature_drift_flags     TEXT[],

    -- MLflow reference
    mlflow_run_id           VARCHAR(100),
    mlflow_model_uri        TEXT,
    is_champion             BOOLEAN         NOT NULL DEFAULT FALSE,

    inserted_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_perf_symbol_type
    ON model_performance (symbol, model_type, evaluation_date DESC);

-- =============================================================================
-- SYSTEM EVENTS — Circuit breaker triggers, retraining, connections, regime shifts
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_events (
    id              BIGSERIAL       PRIMARY KEY,
    event_type      VARCHAR(50)     NOT NULL,
    -- Types: circuit_breaker_trigger / circuit_breaker_reset /
    --        model_retrained / model_promoted / model_rollback /
    --        mt5_connection_lost / mt5_connection_restored /
    --        regime_shift / data_quality_issue /
    --        startup / shutdown / emergency_close
    severity        VARCHAR(10)     NOT NULL DEFAULT 'INFO',    -- DEBUG / INFO / WARNING / ERROR / CRITICAL
    symbol          VARCHAR(20),
    message         TEXT            NOT NULL,
    details         JSONB,
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_events_ts
    ON system_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_type
    ON system_events (event_type, timestamp DESC);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- Daily performance summary
CREATE OR REPLACE VIEW v_daily_performance AS
SELECT
    DATE(entry_time AT TIME ZONE 'UTC') AS trade_date,
    symbol,
    module,
    COUNT(*)                            AS num_trades,
    SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) AS wins,
    ROUND(AVG(r_multiple)::NUMERIC, 3)  AS avg_r,
    ROUND(SUM(pnl_percent)::NUMERIC, 4) AS total_pnl_pct,
    ROUND(SUM(pnl_currency)::NUMERIC, 2) AS total_pnl_usd
FROM trades
WHERE status = 'CLOSED'
  AND entry_time IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1 DESC, 2, 3;

-- Open positions snapshot
CREATE OR REPLACE VIEW v_open_positions AS
SELECT
    ticket, symbol, direction, module,
    entry_time, entry_price, lot_size,
    stop_loss_initial, take_profit_1, take_profit_2,
    final_risk_pct, risk_amount_currency,
    h4_regime_at_entry, m15_regime_at_entry,
    inserted_at
FROM trades
WHERE status = 'OPEN'
ORDER BY entry_time DESC;

-- Signal acceptance rate
CREATE OR REPLACE VIEW v_signal_stats AS
SELECT
    symbol,
    module,
    DATE_TRUNC('day', timestamp) AS day,
    COUNT(*)                     AS total_signals,
    SUM(was_executed::INT)       AS executed,
    ROUND(SUM(was_executed::INT)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 1) AS acceptance_pct
FROM signals
GROUP BY 1, 2, 3
ORDER BY 3 DESC, 1, 2;
