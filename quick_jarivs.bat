@echo off

title JARVIS Client
cd /d "%~dp0"

echo Starting JARVIS Client...
echo Make sure the server is running first!
echo.

uv run python -m jarvis -d

pause
