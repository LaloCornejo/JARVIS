@echo off

title JARVIS Server
cd /d "%~dp0"

echo Starting JARVIS Server...
echo.

uv run python -c "from core.websocket_server import run_server; run_server()"

pause