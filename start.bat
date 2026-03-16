@echo off
title Pulse - Nova's Heartbeat
echo ================================================
echo   Pulse - Starting Nova...
echo ================================================
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0pulse.py" %*
echo.
echo Pulse has stopped.
pause
