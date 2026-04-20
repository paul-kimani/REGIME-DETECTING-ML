# Trading System

Hybrid ML automated trading system for Forex, Metals, and Indices.
Timeframes: M15–H4. Execution via MetaTrader 5 on Windows.

---

## Architecture

```
Windows machine
─────────────────────────────────────────────────────────
MetaTrader 5 terminal  (must be open and logged in)
        │  MT5 Python API — direct in-process call
core/execution/mt5_connector.py   ← ONLY file importing MetaTrader5
        │
run_live.py  /  run_monitor.py
  ├── RegimeDetector   (HMM discovery → XGBoost live)
  ├── SignalRouter     (Momentum / Mean-Reversion / Breakout)
  ├── RiskEngine       (Kelly + ATR sizing, circuit breakers)
  └── PositionManager  (Python-managed stops — sl=0 tp=0 in MT5)
─────────────────────────────────────────────────────────
Docker Desktop
  ├── PostgreSQL  (port 5432)
  ├── Redis       (port 6379)
  ├── MLflow      (port 5000)
  └── Grafana     (port 3000)
```

There is **no HTTP bridge, no network hop, no FastAPI server**.  
MetaTrader5 Python library runs natively on the same Windows machine as the strategy engine.

---

## Quick Start (Windows)

See **[SETUP_WINDOWS.md](SETUP_WINDOWS.md)** for the full guide.

```cmd
:: 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git trading-system
cd trading-system
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

:: 2. Configure
copy .env.example .env
notepad .env          :: fill in MT5 credentials and DB password

:: 3. Start infrastructure
docker-compose up -d

:: 4. Go live
start.bat             :: opens Monitor + Trading windows
```

---

## Development (Mac)

See **[SETUP_MAC.md](SETUP_MAC.md)**.  
Mac is code + test only — MetaTrader5 does not run on Mac. All tests mock `MT5Connector`.

```bash
pip install -r requirements.txt   # MetaTrader5 install will fail — expected
pytest tests/ -v
```

---

## Backtesting

```cmd
python run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2024-01-01 --folds 4
```

Reports are written to `reports/`.

---

## Configuration

All parameters live in `configs/` — never hardcoded:

| File | Purpose |
|---|---|
| `assets.yaml` | Tradeable instruments, pip sizes, session filters |
| `risk_params.yaml` | Kelly sizing, volatility scalars, portfolio heat limits |
| `regime_params.yaml` | HMM states, persistence gates, MTF weights |
| `signal_params.yaml` | Model hyperparameters, entry logic per strategy |
| `prop_firm.yaml` | FTMO compliance rules, circuit breakers |

Copy `.env.example` → `.env` and fill in your values before running.

---

## Project Structure

```
trading-system/
├── run_live.py              ← live trading entry point
├── run_monitor.py           ← monitoring + retraining daemon
├── run_backtest.py          ← walk-forward backtesting CLI
├── start.bat                ← one-click launcher (Windows)
├── .env.example             ← environment template
├── requirements.txt         ← all Python dependencies
├── docker-compose.yml       ← PostgreSQL / Redis / MLflow / Grafana
├── core/
│   ├── execution/
│   │   ├── mt5_connector.py ← ONLY file that imports MetaTrader5
│   │   ├── order_manager.py
│   │   ├── position_manager.py
│   │   ├── fill_monitor.py
│   │   └── trade_journal.py
│   ├── regime/              ← HMM + XGBoost regime detection, MTF alignment
│   ├── signals/             ← Momentum, mean-reversion, breakout modules
│   ├── risk/                ← Kelly sizer, circuit breakers, prop-firm compliance
│   ├── monitoring/          ← Drift detector, MLflow tracker, Grafana exporter
│   ├── data/                ← Data pipeline, feature engineer, DB manager
│   └── utils/               ← Logger, config loader, helpers
├── backtesting/             ← Walk-forward validator, simulation engine, metrics
├── configs/                 ← YAML configuration files
├── database/                ← PostgreSQL schema
├── models/                  ← Trained model artifacts
├── tests/                   ← pytest test suite
├── reports/                 ← backtest output
└── SETUP_WINDOWS.md / SETUP_MAC.md
```

---

## Risk Warning

This software is for educational and research purposes only. Trading financial instruments
carries significant risk of loss. Always test thoroughly on a demo account before live deployment.
