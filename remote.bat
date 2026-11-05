@echo off
setlocal EnableExtensions
rem Remote companions on the home server (designs/remote_host.md).
rem   remote.bat                     menu: who's up, start / stop / restart / log
rem   remote.bat who                 who's running on the box
rem   remote.bat start kai           (also: stop, restart, status, log kai 200, ...)
rem Anything else goes straight to scripts\remote.py.
title Pulse - remote companions
set "PY=%~dp0.venv\Scripts\python.exe"
set "REMOTE=%~dp0scripts\remote.py"

if "%~1"=="" goto menu
"%PY%" "%REMOTE%" %*
set "CODE=%ERRORLEVEL%"
if /i "%~1"=="start" goto after_control
if /i "%~1"=="stop" goto after_control
if /i "%~1"=="restart" goto after_control
exit /b %CODE%

:after_control
if not "%CODE%"=="0" exit /b %CODE%
call :settle %1
exit /b 0

:menu
cls
echo ================================================
echo   Pulse - companions on the box
echo ================================================
echo.
"%PY%" "%REMOTE%" who
echo.
echo   [S]tart  s[T]op  [R]estart  [L]og  re[F]resh  [Q]uit
echo.
choice /c STRLFQ /n /m "  What now? "
if errorlevel 6 exit /b 0
if errorlevel 5 goto menu
if errorlevel 4 goto do_log
if errorlevel 3 (set "VERB=restart" & goto ask_name)
if errorlevel 2 (set "VERB=stop" & goto ask_name)
set "VERB=start"

:ask_name
echo.
set "WHO="
set /p "WHO=  %VERB% who? (folder name, e.g. kai; empty = back) "
if "%WHO%"=="" goto menu
"%PY%" "%REMOTE%" %VERB% %WHO% >nul
if errorlevel 1 (
    echo.
    "%PY%" "%REMOTE%" %VERB% %WHO%
    pause
    goto menu
)
call :settle %VERB%
pause
goto menu

:do_log
echo.
set "WHO="
set /p "WHO=  Log of who? (empty = back) "
if "%WHO%"=="" goto menu
"%PY%" "%REMOTE%" log %WHO% 60
echo.
pause
goto menu

rem Give systemd a moment, then show the new state. A graceful stop can wait
rem for a reply in flight (up to ~4 min), so "stopping" right away is normal.
:settle
echo.
echo   %~1 requested. Checking in a few seconds...
timeout /t 8 /nobreak >nul
"%PY%" "%REMOTE%" who
exit /b 0
