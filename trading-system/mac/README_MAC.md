# Mac Trading System — Full Setup Guide

The Mac is where everything lives: strategy code, models, backtesting, live trading, and monitoring. The Windows machine runs only the MT5 bridge (see [windows/README_WINDOWS.md](../windows/README_WINDOWS.md)).

---

## Prerequisites

| Requirement | Version |
|---|---|
| macOS | 12 Monterey or later (Apple Silicon supported) |
| Python | 3.11 (`brew install python@3.11`) |
| Docker Desktop | Latest (for Postgres, Redis, Grafana, MLflow) |
| Git | Any recent version |

---

## Initial setup (first time only)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git trading-system
cd trading-system
```

### 2. Create a Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key packages (from `requirements.txt`):

```
pandas>=2.2
pandas_ta>=0.3.14b
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
hmmlearn>=0.3
torch>=2.2                 # for LSTM (optional — CPU is fine)
httpx[http2]>=0.27
tenacity>=8.2
fastapi>=0.110             # used by bridge only
uvicorn>=0.29
sqlalchemy>=2.0
psycopg2-binary>=2.9
redis>=5.0
mlflow>=2.13
optuna>=3.6
matplotlib>=3.8
pytest>=8.1
pytest-cov>=5.0
python-dotenv>=1.0
```

### 4. Create the environment file

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```bash
# MT5 Bridge connection
MT5_BRIDGE_URL=http://192.168.1.50:8000    # Windows machine LAN IP
MT5_API_KEY=your-long-random-secret-key-here

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=your-db-password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Mode
PROP_FIRM_MODE=false     # set to true when trading FTMO/funded account
```

### 5. Start infrastructure with Docker

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL 15** on port 5432
- **Redis 7** on port 6379
- **MLflow 2.13** on port 5000
- **Grafana 10** on port 3000 (admin/admin)

### 6. Initialise the database

```bash
psql -h localhost -U trading_user -d trading -f database/schema.sql
```

Or if you have Docker Postgres:

```bash
docker exec -i trading-postgres psql -U trading_user -d trading < database/schema.sql
```

### 7. Verify MT5 bridge is reachable

```bash
curl -H "X-API-Key: your-api-key" http://192.168.1.50:8000/health
```

Expected:
```json
{"status": "ok", "mt5_connected": true}
```

---

## Backtesting

Run a walk-forward backtest for a single symbol:

```bash
python mac/run_backtest.py \
  --symbol XAUUSD \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --folds 4 \
  --balance 100000 \
  --output reports/
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--symbol` | required | Instrument symbol (XAUUSD, EURUSD, etc.) |
| `--start` | required | Backtest start date YYYY-MM-DD |
| `--end` | required | Backtest end date YYYY-MM-DD |
| `--folds` | 4 | Number of walk-forward folds |
| `--balance` | 100000 | Starting account balance |
| `--output` | reports | Output directory for charts and CSV |
| `--no-plots` | false | Skip chart generation |

Reports are written to `--output/`:
- `backtest_SYMBOL_TIMESTAMP.json` — full metrics
- `backtest_SYMBOL_TIMESTAMP.csv` — trade log
- `equity_curve.png`, `drawdown.png`, `r_distribution.png`, `regime_performance.png`

---

## Model training

Before running live, train models with historical data. The monitoring daemon can do this automatically, or you can trigger it manually:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from mac.run_monitor import _full_retrain_pipeline
# ... (see run_monitor.py for the full pipeline call signature)
"
```

Or run the monitor daemon for one cycle:

```bash
python mac/run_monitor.py
```

The daemon will train models on startup if no saved models are found.

---

## Live trading

```bash
# Ensure Docker infrastructure is running
docker-compose up -d

# Ensure Windows MT5 bridge is running (see windows/README_WINDOWS.md)

# Start live trading
python mac/run_live.py
```

Startup sequence:
1. Loads config from `configs/`
2. Connects to PostgreSQL
3. Verifies MT5 bridge (3 retries, 5 s apart)
4. Loads saved champion models
5. Fetches 500-bar warmup data per asset
6. Starts PositionManager background thread
7. Registers M15 candle callbacks for all assets
8. Spins waiting for candle events

Stop cleanly with `Ctrl+C` — positions are not automatically closed. Open positions continue to be managed until you stop the process.

---

## Monitoring daemon

```bash
python mac/run_monitor.py
```

The daemon runs these scheduled jobs:

| Job | Schedule | Purpose |
|---|---|---|
| Candle monitoring | Every M15 close | Regime export, Grafana update |
| Drift check | Every 500 candles | PSI feature drift detection |
| Session close | 22:00 UTC daily | Performance review and alerts |
| Weekly retrain | Sunday 01:00 UTC | Full 10-step retraining pipeline |

You can run the monitor and the live trading process simultaneously (they share read-only model access).

---

## Pushing code to Windows

The workflow is: **edit on Mac → push to GitHub → Windows pulls**.

```bash
# On Mac
git add -A
git commit -m "your message"
git push

# On Windows (in cmd)
git pull
pip install -r requirements_windows.txt
# restart bridge
```

Windows never edits code. It only pulls and runs `mt5_bridge.py`.

---

## Running tests

```bash
cd trading-system
pytest tests/ -v --cov=core --cov=backtesting --cov-report=term-missing
```

Minimum coverage target: **80%**.

Run a specific test file:
```bash
pytest tests/test_signals.py -v
```

---

## Grafana dashboards

Open [http://localhost:3000](http://localhost:3000) (admin / admin).

The system exports these Redis keys (auto-imported by the provisioned datasource):

| Key pattern | Content | TTL |
|---|---|---|
| `trading:regime:{symbol}` | Current regime state JSON | 120 s |
| `trading:signal:{symbol}` | Latest signal | 60 s |
| `trading:account` | Balance, equity, daily P&L | 30 s |
| `trading:circuit_breaker` | CB level | 30 s |
| `trading:open_positions` | Open position count | 30 s |

---

## Project layout

```
trading-system/
├── core/
│   ├── data/           data_pipeline.py, feature_engineer.py, db_manager.py
│   ├── regime/         hmm_model.py, xgb_classifier.py, mtf_alignment.py, regime_detector.py
│   ├── signals/        momentum_module.py, mean_reversion_module.py, breakout_module.py, signal_router.py
│   ├── risk/           __init__.py (RiskEngine), kelly_sizer.py, circuit_breakers.py, prop_firm_compliance.py
│   ├── execution/      mt5_client.py, order_manager.py, position_manager.py, fill_monitor.py, trade_journal.py
│   ├── monitoring/     performance_monitor.py, drift_detector.py, mlflow_tracker.py, grafana_exporter.py
│   └── utils/          config.py, logger.py, helpers.py
├── backtesting/        simulation_engine.py, performance_metrics.py, walk_forward.py, results_analyzer.py
├── configs/            assets.yaml, risk_params.yaml, regime_params.yaml, signal_params.yaml, prop_firm.yaml
├── database/           schema.sql (PostgreSQL)
├── mac/                run_live.py, run_backtest.py, run_monitor.py
├── windows/            mt5_bridge.py (runs on Windows only)
├── tests/              conftest.py, test_regime.py, test_signals.py, test_risk.py, test_execution.py, test_backtesting.py
├── models/             xgboost/ (saved .pkl bundles)
├── reports/            backtest output
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Configuration reference

All parameters are in `configs/`. No values are hardcoded in Python.

### `configs/assets.yaml`
Defines tradeable instruments: symbol, pip size, session filter, HMM state count.

### `configs/risk_params.yaml`
Kelly fraction, base/max risk per trade, portfolio heat caps, lot step.

### `configs/regime_params.yaml`
HMM states and transition thresholds; MTF weights (H4=50%, H1=30%, M15=20%); regime persistence gates.

### `configs/signal_params.yaml`
XGBoost hyperparameters for all three signal modules; entry confidence thresholds; z-score / oscillator parameters.

### `configs/prop_firm.yaml`
FTMO-style daily loss (4% internal buffer), total drawdown (8% internal buffer); 4-level circuit breaker thresholds.

---

## Risk warning

This software is for educational and research purposes only. Live trading of financial instruments carries significant risk of loss. Always complete full walk-forward backtesting and at minimum one month of shadow/demo trading before deploying on a live account. The authors accept no responsibility for trading losses.
