# Mac Setup — Development Environment

The Mac is a **development and testing environment only**. MetaTrader5 does not run on Mac, so live trading and real data fetching are not possible. You can write code, run the full test suite, and push changes that Windows will pull.

---

## What works on Mac

| Feature | Mac |
|---|---|
| Edit source code | Yes |
| Run pytest test suite | Yes |
| Run backtests (with CSV data) | Yes (no MT5) |
| Live trading | No — MetaTrader5 Windows only |
| MT5 data fetching | No — MetaTrader5 Windows only |

---

## Prerequisites

| Requirement | Install |
|---|---|
| Python 3.11 | `pyenv install 3.11.9` |
| pyenv | `brew install pyenv` |
| TA-Lib (optional) | `brew install ta-lib` |
| Git | Built in or `brew install git` |

---

## Setup

### 1. Install pyenv and Python 3.11

```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9   # sets .python-version in project root
```

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git trading-system
cd trading-system
```

### 3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install TA-Lib (optional but recommended)

```bash
brew install ta-lib
pip install TA-Lib
```

If you skip this step, the code falls back to `pandas-ta` automatically.

### 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`MetaTrader5` will **fail to install** on Mac — that is expected and safe to ignore. The error looks like:

```
ERROR: Could not find a version that satisfies the requirement MetaTrader5==5.0.45
```

The codebase guards the import:

```python
try:
    import MetaTrader5 as _mt5_lib
    _MT5_AVAILABLE = True
except ImportError:
    _mt5_lib = None
    _MT5_AVAILABLE = False
```

All tests mock `MT5Connector` so they run without the real library.

### 6. Create a minimal .env (for config loading)

```bash
cp .env.example .env
```

You do not need to fill in real values for development — defaults are used when fields are blank.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v --cov=core --cov=backtesting --cov-report=term-missing
```

The `conftest.py` provides `mock_mt5_client` as a `MagicMock` fixture, so all execution tests work without MetaTrader5.

To run a specific test file:

```bash
pytest tests/test_regime.py -v
pytest tests/test_risk.py -v
pytest tests/test_execution.py -v
```

---

## Development Workflow

1. Make changes on Mac and test with `pytest`.
2. Push to your remote branch: `git push origin main`.
3. On Windows, pull the changes: `git pull`.
4. No restarts required for Docker services unless `docker-compose.yml` changed.

---

## Note on the `mac/` folder

The `mac/` folder contains the previous entry points (`run_live.py`, `run_backtest.py`, `run_monitor.py`). These are superseded by the root-level `run_live.py`, `run_backtest.py`, and `run_monitor.py`. The `mac/` folder is kept for reference but the root versions should be used.

---

## Project Structure

```
trading-system/
├── run_live.py            ← live trading entry point (run on Windows)
├── run_monitor.py         ← monitoring daemon (run on Windows)
├── run_backtest.py        ← backtesting CLI (run on Windows or Mac with CSV data)
├── core/
│   └── execution/
│       └── mt5_connector.py  ← ONLY file that imports MetaTrader5
├── tests/
│   ├── conftest.py           ← shared fixtures including mock_mt5_client
│   ├── test_regime.py
│   ├── test_signals.py
│   ├── test_risk.py
│   ├── test_execution.py
│   └── test_backtesting.py
├── requirements.txt
└── .env.example
```
