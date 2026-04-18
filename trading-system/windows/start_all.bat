@echo off
title Trading System Launcher
echo ============================================
echo  Trading System — Full Startup
echo ============================================
echo.
echo  This will open TWO windows:
echo    Window 1: MT5 Bridge (port 8000)
echo    Window 2: Live Trading System
echo.
echo  Make sure MetaTrader 5 is open and logged in.
echo  Make sure Docker Desktop is running (for Postgres/Redis).
echo.
pause

cd /d "%~dp0.."

echo Starting MT5 Bridge ...
start "MT5 Bridge" cmd /k "cd /d %CD% && python windows\mt5_bridge.py"

echo Waiting 5 seconds for bridge to initialise ...
timeout /t 5 /nobreak >nul

echo Starting Live Trading System ...
start "Live Trading" cmd /k "cd /d %CD% && python mac\run_live.py"

echo.
echo Both processes are running in separate windows.
echo Close those windows (or press Ctrl+C in each) to stop.
