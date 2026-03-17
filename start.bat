@echo off
title Pulse - AI Companion Heartbeat
echo ================================================
echo   Pulse - Starting companion...
echo ================================================
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0pulse.py" %*
echo.
echo Pulse has stopped.
pause
