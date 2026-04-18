# Windows — Full Trading System Setup

This guide sets up the **complete** trading system on a Windows machine that also has MetaTrader 5 installed. Both the MT5 bridge and the full strategy engine run on the same machine.

---

## How it works on Windows

```
Windows machine
─────────────────────────────────────────
MetaTrader 5 terminal
        │  (MT5 Python API — local)
windows\mt5_bridge.py   (localhost:8000)
        │  (HTTP — loopback)
mac\run_live.py          ← same Python code as Mac
        ├── RegimeDetector
        ├── SignalRouter
        ├── RiskEngine
        └── PositionManager
─────────────────────────────────────────
Docker Desktop
  ├── PostgreSQL (port 5432)
  ├── Redis      (port 6379)
  ├── MLflow     (port 5000)
  └── Grafana    (port 3000)
```

The bridge and the trading engine are on the same machine, so the HTTP connection is `localhost:8000` — no LAN networking required.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Windows 10 / 11 | 64-bit | |
| Python | 3.11 64-bit | `winget install Python.Python.3.11` |
| MetaTrader 5 | Any recent build | Must be running and logged in |
| Docker Desktop | Latest | For Postgres / Redis / MLflow / Grafana |
| Git | Any | `winget install Git.Git` |

---

## Initial setup (run once)

### 1. Clone the repository

Open **Command Prompt** or **Windows Terminal**:

```cmd
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git trading-system
cd trading-system
```

### 2. Create a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install all dependencies

```cmd
pip install --upgrade pip
pip install -r requirements_windows.txt
```

> **PyTorch note:** The default installs the CPU build. For GPU (NVIDIA), replace the torch line with:
> ```cmd
> pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Create the environment file

```cmd
copy windows\.env.windows.example .env
notepad .env
```

Fill in at minimum:

```
MT5_BRIDGE_API_KEY=your-long-random-secret-key-here
DB_PASSWORD=your-db-password
```

Leave `MT5_BRIDGE_URL=http://localhost:8000` and `MT5_ALLOWED_IP=` (empty) — the bridge and trading system are on the same machine.

### 5. Start Docker Desktop and launch infrastructure

Open Docker Desktop, wait for it to be ready, then:

```cmd
docker-compose up -d
```

This starts PostgreSQL, Redis, MLflow, and Grafana in the background. Verify:

```cmd
docker ps
```

You should see four containers running.

### 6. Initialise the database

```cmd
docker exec -i trading-postgres psql -U trading_user -d trading < database\schema.sql
```

### 7. Verify Docker services

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5000 | none |
| PostgreSQL | localhost:5432 | trading_user / your-password |
| Redis | localhost:6379 | none |

---

## Updating the code

```cmd
cd trading-system
git pull
pip install -r requirements_windows.txt
```

That's all. No rebuilding, no recompiling.

---

## Daily usage

### Option A — Start everything with one double-click

Double-click `windows\start_all.bat`. It opens two Command Prompt windows:
- **Window 1** — MT5 Bridge (keep open)
- **Window 2** — Live Trading System (keep open)

Close either window to stop that process. Positions already open are not automatically closed.

### Option B — Start components separately

**Step 1: Start the MT5 bridge**

```cmd
windows\start_bridge.bat
```

You should see:
```
INFO: MT5 connected: YourBroker-Demo (account 12345678)
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Step 2: Start live trading** (in a separate window)

```cmd
windows\start_live.bat
```

**Step 3 (optional): Start the monitoring daemon** (in a separate window)

```cmd
windows\start_monitor.bat
```

---

## Backtesting

```cmd
windows\run_backtest.bat XAUUSD 2022-01-01 2024-01-01
```

Or with all options:

```cmd
windows\run_backtest.bat XAUUSD 2022-01-01 2024-01-01 4 100000
```

Arguments: `SYMBOL  START  END  [FOLDS=4]  [BALANCE=100000]`

Reports are written to `reports\`.

Or call Python directly for full control:

```cmd
python mac\run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2024-01-01 --folds 4 --output reports
```

---

## Running tests

```cmd
.venv\Scripts\activate
pytest tests\ -v --cov=core --cov=backtesting --cov-report=term-missing
```

---

## Stopping the system

- Press `Ctrl+C` in the trading or monitor windows to stop cleanly.
- Open positions are **not** automatically closed on shutdown. They continue to be managed only while `run_live.py` is running.
- To stop Docker services: `docker-compose down`

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MT5 not connected` on bridge startup | Open MetaTrader 5 terminal and log in before starting the bridge |
| `401 Unauthorized` or `403 Forbidden` when trading system calls bridge | `MT5_BRIDGE_API_KEY` in `.env` does not match; both bridge and trading system read the same `.env` in the project root |
| `psycopg2.OperationalError: could not connect` | Docker is not running or `DB_PASSWORD` in `.env` is wrong |
| `pip install MetaTrader5` fails | Must use Python 3.11 **64-bit** on Windows |
| `pip install torch` hangs | Use the `+cpu` wheel for a fast install; see step 3 |
| `ModuleNotFoundError: No module named 'core'` | Activate the virtual environment: `.venv\Scripts\activate` |
| Bridge starts but MT5 returns -10004 | MT5 terminal is logged in to a read-only account; check account permissions |

---

## File structure on Windows

```
trading-system\
├── windows\
│   ├── mt5_bridge.py          ← bridge server (Windows only)
│   ├── start_bridge.bat       ← start the bridge
│   ├── start_live.bat         ← start live trading
│   ├── start_monitor.bat      ← start monitoring daemon
│   ├── start_all.bat          ← start bridge + live together
│   ├── run_backtest.bat       ← backtesting CLI wrapper
│   └── .env.windows.example   ← copy to project root as .env
├── mac\
│   ├── run_live.py            ← live trading entry point
│   ├── run_monitor.py         ← monitoring daemon
│   └── run_backtest.py        ← backtesting CLI
├── core\                      ← all strategy logic
├── backtesting\               ← walk-forward validator
├── configs\                   ← YAML configuration
├── database\
│   └── schema.sql             ← PostgreSQL schema
├── models\                    ← saved XGBoost model bundles
├── reports\                   ← backtest output
├── .env                       ← your environment variables
├── requirements_windows.txt   ← Python packages for Windows
└── docker-compose.yml         ← infrastructure
```

---

## Notes on the `mac\` folder name

The entry point scripts live in `mac\` for historical reasons (they were developed on Mac). They are 100% cross-platform Python — no Mac-only code. On Windows you call them exactly as shown above (`python mac\run_live.py`).

---

## Risk warning

This software is for educational and research purposes only. Trading financial instruments carries significant risk of loss. Always complete walk-forward backtesting and demo trading before deploying with real money.
