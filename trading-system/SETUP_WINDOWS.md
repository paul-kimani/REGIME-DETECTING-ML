# Windows Setup — Full Trading System

This guide sets up the complete trading system on a Windows machine that also has MetaTrader 5 installed. MetaTrader5 Python library connects **directly** to the running MT5 terminal — no HTTP bridge, no network hop.

---

## Architecture

```
Windows machine
─────────────────────────────────────────────────────────
MetaTrader 5 terminal  (must be open and logged in)
        │  MT5 Python API — direct in-process call
core/execution/mt5_connector.py
        │
run_live.py  /  run_monitor.py
  ├── RegimeDetector (HMM + XGBoost)
  ├── SignalRouter   (Momentum / MeanReversion / Breakout)
  ├── RiskEngine     (Kelly sizing, prop-firm compliance)
  └── PositionManager (Python-managed stops, no SL/TP in MT5)
─────────────────────────────────────────────────────────
Docker Desktop
  ├── PostgreSQL  (port 5432)  — trade history & features
  ├── Redis       (port 6379)  — real-time state cache
  ├── MLflow      (port 5000)  — experiment tracking
  └── Grafana     (port 3000)  — dashboards
```

**Important:** SL and TP are **never** set in MT5 (`sl=0, tp=0` always). All stop management runs in Python via `PositionManager`.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Windows 10 / 11 | 64-bit | |
| Python | 3.11 64-bit | `winget install Python.Python.3.11` |
| MetaTrader 5 | Any recent build | Must be running and logged in before `run_live.py` |
| Docker Desktop | Latest | For Postgres / Redis / MLflow / Grafana |
| Git | Any | `winget install Git.Git` |

---

## Initial Setup (run once)

### 1. Clone the repository

```cmd
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git trading-system
cd trading-system
```

### 2. Create a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install TA-Lib (pre-built wheel required on Windows)

Download the wheel for Python 3.11 64-bit from:
> https://github.com/cgohlke/talib-binary/releases

Look for a file named something like `TA_Lib-0.4.32-cp311-cp311-win_amd64.whl`, then:

```cmd
pip install TA_Lib-0.4.32-cp311-cp311-win_amd64.whl
```

### 4. Install all other dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch GPU note:** The default installs the CPU build. For GPU (NVIDIA) replace the torch line with:
> ```cmd
> pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

### 5. Create the environment file

```cmd
copy .env.example .env
notepad .env
```

Fill in at minimum:

```
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
DB_PASSWORD=your-db-password
```

Leave `MT5_TERMINAL_PATH` blank unless MT5 is installed in a non-standard location.

### 6. Start Docker Desktop and launch infrastructure

Open Docker Desktop, wait for it to be ready, then:

```cmd
docker-compose up -d
```

Verify all four containers are running:

```cmd
docker ps
```

### 7. Initialise the database

```cmd
docker exec -i trading-postgres psql -U trading_user -d trading < database\schema.sql
```

### 8. Verify Docker services

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5000 | none |
| PostgreSQL | localhost:5432 | trading_user / your-password |
| Redis | localhost:6379 | none |

### 9. Run a first backtest to validate the setup

```cmd
python run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2023-01-01 --folds 2
```

This does **not** require MT5 to be running if PostgreSQL has data. If MT5 is running, it will fetch data directly.

---

## Daily Usage

### One-click start (recommended)

Double-click `start.bat`. It opens two Command Prompt windows:
- **Monitor** — drift detection and model retraining daemon
- **Trading** — live signal generation and order execution

Close either window to stop that process. Open positions are **not** automatically closed on shutdown.

### Manual start (separate windows)

**Window 1 — Monitor daemon:**
```cmd
.venv\Scripts\activate
python run_monitor.py
```

**Window 2 — Live trading:**
```cmd
.venv\Scripts\activate
python run_live.py
```

**Optional — Backtest:**
```cmd
python run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2024-01-01 --folds 4
```

---

## Auto-start with Windows Task Scheduler

To start the trading system automatically at Windows logon:

1. Open **Task Scheduler** → Create Basic Task
2. Name: `Trading System`
3. Trigger: At log on
4. Action: Start a program
   - Program: `C:\path\to\trading-system\start.bat`
   - Start in: `C:\path\to\trading-system\`
5. Finish

Make sure Docker Desktop is also configured to start at logon (Settings → General → Start Docker Desktop when you log in).

---

## Backtesting

```cmd
python run_backtest.py --symbol XAUUSD --start 2022-01-01 --end 2024-01-01 --folds 4 --balance 100000
```

Arguments: `--symbol SYMBOL --start YYYY-MM-DD --end YYYY-MM-DD [--folds N] [--balance N] [--no-plots]`

Reports are written to `reports\`.

---

## Running Tests

```cmd
.venv\Scripts\activate
pytest tests\ -v --cov=core --cov=backtesting --cov-report=term-missing
```

---

## Stopping the System

- Press `Ctrl+C` in the trading or monitor windows to stop cleanly.
- Open positions are **not** automatically closed. They continue to be managed only while `run_live.py` is running.
- To stop Docker services: `docker-compose down`

---

## Updating the Code

```cmd
cd trading-system
git pull
pip install -r requirements.txt
```

No recompiling, no rebuilding Docker images (unless `docker-compose.yml` changed).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MetaTrader5 not available` on startup | Ensure MetaTrader5 package installed: `pip install MetaTrader5` |
| `MT5 initialize() failed` | Open MetaTrader 5 terminal and log in before starting `run_live.py` |
| `psycopg2.OperationalError` | Docker is not running or `DB_PASSWORD` in `.env` is wrong |
| `pip install MetaTrader5` fails | Must use Python **3.11 64-bit** on Windows (not 3.12+, not 32-bit) |
| `TA_Lib ImportError` | Install the `.whl` manually (step 3 above) before `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'core'` | Activate the virtual environment: `.venv\Scripts\activate` |
| MT5 returns retcode -10004 | MT5 terminal is in read-only mode; check account permissions in MT5 |

---

## File Structure

```
trading-system\
├── run_live.py            ← live trading entry point
├── run_monitor.py         ← monitoring and retraining daemon
├── run_backtest.py        ← backtesting CLI
├── start.bat              ← one-click launcher
├── .env                   ← your environment variables (gitignored)
├── .env.example           ← template — copy to .env
├── requirements.txt       ← all Python dependencies
├── docker-compose.yml     ← infrastructure
├── core\                  ← all strategy and execution logic
│   ├── execution\
│   │   └── mt5_connector.py  ← ONLY file that imports MetaTrader5
│   ├── data\
│   ├── regime\
│   ├── signals\
│   ├── risk\
│   └── monitoring\
├── backtesting\           ← walk-forward validation
├── configs\               ← YAML configuration
├── database\
│   └── schema.sql
├── models\                ← saved XGBoost model bundles
├── reports\               ← backtest output
└── tests\                 ← pytest test suite
```

---

## Risk Warning

This software is for educational and research purposes only. Trading financial instruments carries significant risk of loss. Always complete walk-forward backtesting and demo trading before deploying with real money.
