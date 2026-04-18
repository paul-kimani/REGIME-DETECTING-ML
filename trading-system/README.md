# Trading System

Hybrid ML automated trading system for Forex, Metals, and Indices.
Timeframes: 15m–4H. Execution via MT5.

## System Design

- **Regime Detection**: HMM (discovery) → XGBoost (live prediction) → MTF alignment
- **Signal Engine**: Momentum / Mean-Reversion / Breakout modules (XGBoost + LSTM)
- **Risk Engine**: Kelly + ATR volatility + confidence + regime sizing, 4-level circuit breakers
- **Execution**: Python-managed positions (no broker-side stops), MT5 bridge over LAN
- **Monitoring**: MLflow + Grafana + automated retraining pipeline

## Architecture

```
Mac (main system)          Windows (MT5 terminal)
───────────────────        ──────────────────────
run_live.py                mt5_bridge.py (FastAPI)
  │                          │
  ├─ RegimeDetector           └─ MetaTrader5 Python API
  ├─ SignalRouter                  └─ Broker
  ├─ RiskEngine
  └─ ExecutionEngine ──HTTP──► MT5 Bridge
       └─ PositionManager
```

## Quick Start

### Windows (MT5 Bridge)
See [windows/README_WINDOWS.md](windows/README_WINDOWS.md)

### Mac (Main System)
See [mac/README_MAC.md](mac/README_MAC.md)

## Configuration

All parameters live in `configs/` — never hardcoded:

| File | Purpose |
|---|---|
| `assets.yaml` | Tradeable instruments, pip sizes, session filters |
| `risk_params.yaml` | Kelly sizing, volatility scalars, portfolio heat limits |
| `regime_params.yaml` | HMM states, persistence gates, MTF weights |
| `signal_params.yaml` | Model hyperparameters, entry logic per strategy |
| `prop_firm.yaml` | FTMO compliance rules, circuit breakers |

Copy `.env.example` to `.env` and fill in your values before running.

## Backtesting

```bash
python mac/run_backtest.py --symbol XAUUSD --start 2020-01-01 --end 2024-01-01 --folds 4
```

## Live Trading

```bash
# Start infrastructure
docker-compose up -d

# Start live system
python mac/run_live.py
```

## Monitoring

| Service | URL |
|---|---|
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5000 |

## Project Structure

```
trading-system/
├── core/
│   ├── regime/       # HMM + XGBoost regime detection, MTF alignment
│   ├── signals/      # Momentum, mean-reversion, breakout modules
│   ├── risk/         # Kelly sizer, circuit breakers, prop-firm compliance
│   ├── execution/    # MT5 client, order/position manager, trade journal
│   ├── monitoring/   # Performance monitor, drift detector, MLflow tracker
│   ├── data/         # Data pipeline, feature engineer, DB manager
│   └── utils/        # Logger, config loader, helpers
├── backtesting/      # Walk-forward validator, simulation engine, metrics
├── windows/          # MT5 FastAPI bridge (runs on Windows)
├── mac/              # Entry points (live, backtest, monitor)
├── configs/          # All YAML configuration
├── database/         # PostgreSQL schema
├── models/           # Trained model artifacts
├── tests/            # pytest test suite
└── docker/           # Dockerfile + Grafana provisioning
```

## Risk Warning

This software is for educational and research purposes. Trading financial instruments
carries significant risk. Always test thoroughly on a demo account before live deployment.
