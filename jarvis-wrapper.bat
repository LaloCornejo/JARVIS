@echo off
:: JARVIS Server Wrapper Script - Auto-restart on Telegram /restart command
:: Usage: jarvis-wrapper.bat [options]

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"
set "RESTART_CODE=42"
set "MAX_RESTARTS=10"
set "RESTART_COUNT=0"
set "RESTART_DELAY=3"

echo ========================================
echo JARVIS Server Wrapper
echo Auto-restart enabled (exit code %RESTART_CODE%)
echo ========================================
echo.

cd /d "%PROJECT_DIR%"

:loop
set /a RESTART_COUNT+=1

if %RESTART_COUNT% gtr %MAX_RESTARTS% (
    echo ⚠️  Maximum restart attempts (%MAX_RESTARTS%) reached. Giving up.
    exit /b 1
)

if %RESTART_COUNT% gtr 1 (
    echo 🔄 Restart attempt %RESTART_COUNT%/%MAX_RESTARTS%
    echo ⏳ Waiting %RESTART_DELAY%s before starting...
    timeout /t %RESTART_DELAY% /nobreak >nul
)

echo 🚀 Starting JARVIS server...
echo.

:: Run the server and capture exit code
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "from core.websocket_server import run_server; run_server()" %*
) else (
    python -c "from core.websocket_server import run_server; run_server()" %*
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.

if %EXIT_CODE% equ %RESTART_CODE% (
    echo 🔄 Restart signal received from Telegram (exit code %EXIT_CODE%)
    echo    Server will restart automatically...
    goto loop
) else if %EXIT_CODE% equ 0 (
    echo ✅ Server exited normally (code %EXIT_CODE%)
    exit /b 0
) else (
    echo ❌ Server crashed with exit code %EXIT_CODE%
    echo    Restarting in %RESTART_DELAY%s...
    timeout /t %RESTART_DELAY% /nobreak >nul
    goto loop
)

endlocal
