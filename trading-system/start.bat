@echo off
title Trading System Launcher
echo ============================================
echo  Trading System — Full Startup
echo ============================================
echo.
echo  Make sure MetaTrader 5 is open and logged in.
echo  Make sure Docker Desktop is running.
echo.
pause

cd /d "%~dp0"

echo [1] Starting Docker services (PostgreSQL, Grafana, MLflow, Redis) ...
docker-compose up -d
timeout /t 5 /nobreak >nul

echo [2] Starting Monitoring Daemon ...
start "Monitor" cmd /k "cd /d %CD% && python run_monitor.py"

echo [3] Starting Live Trading System ...
start "Trading" cmd /k "cd /d %CD% && python run_live.py"

echo.
echo System running.
echo Grafana:  http://localhost:3000
echo MLflow:   http://localhost:5000
echo.
echo Close the Monitor or Trading windows to stop those processes.
echo To stop Docker: docker-compose down
echo.
pause
