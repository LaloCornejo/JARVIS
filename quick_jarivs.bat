@echo off

title JARVIS Start
cd /d "%~dp0"

echo Starting JARVIS...
echo.

uv run python -m jarvis -d 

pause
