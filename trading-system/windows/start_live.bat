@echo off
title Live Trading
echo ============================================
echo  Starting Live Trading System ...
echo  Make sure the MT5 bridge is running first.
echo ============================================
cd /d "%~dp0.."
python mac\run_live.py
pause
