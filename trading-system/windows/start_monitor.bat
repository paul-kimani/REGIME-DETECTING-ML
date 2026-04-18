@echo off
title Monitor Daemon
echo ============================================
echo  Starting Monitoring and Retraining Daemon
echo ============================================
cd /d "%~dp0.."
python mac\run_monitor.py
pause
