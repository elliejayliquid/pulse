@echo off
setlocal EnableExtensions
rem Remote companions on the home server (designs/remote_host.md).
rem   remote.bat                     menu: who's up, then type e.g. "start kai"
rem   remote.bat who                 who's running on the box
rem   remote.bat start kai           (also: stop, restart, status, log kai 200, ...)
rem   remote.bat push kai            send PC config.yaml/persona.yaml edits to the box
rem Anything else goes straight to scripts\remote.py.
title Pulse - remote companions
set "PY=%~dp0.venv\Scripts\python.exe"
set "REMOTE=%~dp0scripts\remote.py"
set "OUT=%TEMP%\pulse_remote_out.txt"

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
"%PY%" "%REMOTE%" who <nul
:help
echo.
echo   Type a command and press Enter, for example:
echo     start kai     stop kai     restart kai     log kai
echo     diff kai      push kai     (send config/persona edits made on the PC)
echo   (either order works: "kai start" too; s / t / r / l for short)
echo   Enter on its own refreshes the list, q quits.

:prompt
echo.
set "LINE="
set /p "LINE=  > "
if not defined LINE goto menu
set "LINE=%LINE:"=%"
set "A=" & set "B="
for /f "tokens=1,2" %%a in ("%LINE%") do (set "A=%%a" & set "B=%%b")
call :verb "%A%"
if defined VERB (set "WHO=%B%") else (call :verb "%B%" & set "WHO=%A%")
if not defined VERB (
    echo   Not sure what to do with "%LINE%".
    if not "%A%"=="" if "%B%"=="" echo   Try: start %A%   stop %A%   log %A%
    goto prompt
)
if "%VERB%"=="quit" exit /b 0
if "%VERB%"=="who" goto menu
if "%VERB%"=="help" goto help
if not defined WHO set /p "WHO=  %VERB% who? (e.g. kai; empty = back) "
if not defined WHO goto prompt
if "%VERB%"=="log" goto do_log
if "%VERB%"=="push" goto do_sync
if "%VERB%"=="diff" goto do_sync

"%PY%" "%REMOTE%" %VERB% %WHO% <nul >"%OUT%" 2>&1
if errorlevel 1 (
    echo.
    type "%OUT%"
    goto prompt
)
call :settle %VERB%
goto prompt

:do_sync
echo.
"%PY%" "%REMOTE%" %VERB% %WHO%
goto prompt

:do_log
echo.
"%PY%" "%REMOTE%" log %WHO% 60 <nul
goto prompt

rem Sets VERB from a typed word, or clears it when the word isn't a command.
:verb
set "VERB="
for %%v in (start s) do if /i "%~1"=="%%v" set "VERB=start"
for %%v in (stop t x) do if /i "%~1"=="%%v" set "VERB=stop"
for %%v in (restart r) do if /i "%~1"=="%%v" set "VERB=restart"
for %%v in (log logs l) do if /i "%~1"=="%%v" set "VERB=log"
for %%v in (who refresh f) do if /i "%~1"=="%%v" set "VERB=who"
for %%v in (push p) do if /i "%~1"=="%%v" set "VERB=push"
for %%v in (diff d) do if /i "%~1"=="%%v" set "VERB=diff"
for %%v in (help h) do if /i "%~1"=="%%v" set "VERB=help"
for %%v in (quit exit q) do if /i "%~1"=="%%v" set "VERB=quit"
exit /b 0

rem Give systemd a moment, then show the new state. A graceful stop can wait
rem for a reply in flight (up to ~4 min), so "stopping" right away is normal.
:settle
echo.
echo   %~1 requested. Checking in a few seconds...
timeout /t 8 /nobreak >nul
"%PY%" "%REMOTE%" who <nul
exit /b 0
