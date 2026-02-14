@echo off
chcp 65001 >nul
title JARVIS + WhatsApp
cd /d "%~dp0"

echo =========================================
echo    JARVIS with WhatsApp Integration
echo =========================================
echo.

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Install deps if needed
if not exist "services\whatsapp-bailey\node_modules" (
    echo [SETUP] Installing WhatsApp dependencies...
    cd services\whatsapp-bailey
    call npm install
    cd ..\..
    echo [SETUP] Done
echo.
)

:: Kill any existing node processes
REM taskkill /F /IM node.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo [START] Launching WhatsApp service...
echo [INFO] QR code will appear below - scan with WhatsApp app
echo.
echo =========================================

:: Create a temp script to run WhatsApp and output to this console
echo @echo off > %TEMP%\whatsapp_runner.bat
echo cd /d "%~dp0\services\whatsapp-bailey" >> %TEMP%\whatsapp_runner.bat
echo node server.js >> %TEMP%\whatsapp_runner.bat

:: Start WhatsApp in same console window
cd services\whatsapp-bailey
start /B node server.js

:: Go back to root
cd ..\..

:: Wait for WhatsApp to show QR
timeout /t 5 /nobreak >nul

:: Show instructions
echo.
echo ^>^>^> WHATSAPP SETUP ^<^<^<
echo 1. Check above for QR code (black/white squares)
echo 2. Open WhatsApp on your phone
echo 3. Settings ^> Linked Devices ^> Link a Device
echo 4. Scan the QR code
echo 5. Wait for "Connected" message
echo.
pause
echo.

:: Now start JARVIS
echo [START] Launching JARVIS...
echo =========================================
echo.

python jarvis_wrapper.py

:: Cleanup
echo.
echo [SHUTDOWN] Stopping WhatsApp service...
taskkill /F /IM node.exe >nul 2>&1
echo [SHUTDOWN] Done.
timeout /t 2 /nobreak >nul
