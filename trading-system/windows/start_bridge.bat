@echo off
title MT5 Bridge
echo ============================================
echo  Starting MT5 Bridge on port 8000 ...
echo ============================================
cd /d "%~dp0.."
python windows\mt5_bridge.py
pause
