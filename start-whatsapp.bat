@echo off
chcp 65001 >nul
title JARVIS WhatsApp Bridge
cd /d "%~dp0"

echo [JARVIS] Starting WhatsApp Baileys Service...
echo.

if not exist "services\whatsapp-bailey\node_modules" (
    echo [JARVIS] Installing dependencies...
    cd services\whatsapp-bailey
    call npm install
    cd ..\..
)

start "WhatsApp Baileys Service" cmd /k "cd services\whatsapp-bailey && node server.js"

timeout /t 3 /nobreak >nul

echo.
echo [JARVIS] WhatsApp service started on http://localhost:3001
echo [JARVIS] Scan the QR code that appears in the other window
echo.
pause
