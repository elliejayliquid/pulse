@echo off
title Pulse Engine
"%~dp0.venv\Scripts\python.exe" "%~dp0pulse_gui.py" %*
if errorlevel 1 pause
