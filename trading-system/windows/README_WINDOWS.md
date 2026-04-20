# DEPRECATED — windows/ folder

This folder contained the MT5 FastAPI bridge server and supporting batch scripts.
The bridge architecture has been removed.

## What replaced it

MetaTrader5 is now used **directly** on Windows via:

```
core/execution/mt5_connector.py
```

This is the only file in the codebase that imports MetaTrader5.
No HTTP server, no network hop, no separate bridge process.

## New entry points (project root)

| Old | New |
|---|---|
| `windows\start_all.bat` | `start.bat` (project root) |
| `windows\start_live.bat` | `python run_live.py` |
| `windows\start_monitor.bat` | `python run_monitor.py` |
| `windows\run_backtest.bat` | `python run_backtest.py` |

## Setup guide

See **SETUP_WINDOWS.md** in the project root.
