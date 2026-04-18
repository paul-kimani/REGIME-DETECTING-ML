@echo off
title Backtest
echo ============================================
echo  Backtest CLI
echo  Usage: run_backtest.bat SYMBOL START END
echo  Example: run_backtest.bat XAUUSD 2022-01-01 2024-01-01
echo ============================================
cd /d "%~dp0.."

if "%~1"=="" (
    echo ERROR: Missing required arguments.
    echo Usage: run_backtest.bat SYMBOL START END [FOLDS] [BALANCE]
    echo Example: run_backtest.bat XAUUSD 2022-01-01 2024-01-01 4 100000
    pause
    exit /b 1
)

set SYMBOL=%~1
set START=%~2
set END=%~3
set FOLDS=%~4
set BALANCE=%~5

if "%FOLDS%"=="" set FOLDS=4
if "%BALANCE%"=="" set BALANCE=100000

python mac\run_backtest.py --symbol %SYMBOL% --start %START% --end %END% --folds %FOLDS% --balance %BALANCE% --output reports
pause
