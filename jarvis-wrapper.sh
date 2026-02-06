#!/bin/bash
# JARVIS Server Wrapper Script - Auto-restart on Telegram /restart command
# Usage: ./jarvis-wrapper.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
RESTART_CODE=42
MAX_RESTARTS=10
RESTART_COUNT=0
RESTART_DELAY=3

echo "========================================"
echo "JARVIS Server Wrapper"
echo "Auto-restart enabled (exit code $RESTART_CODE)"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    if [ $RESTART_COUNT -gt $MAX_RESTARTS ]; then
        echo "⚠️  Maximum restart attempts ($MAX_RESTARTS) reached. Giving up."
        exit 1
    fi
    
    if [ $RESTART_COUNT -gt 1 ]; then
        echo "🔄 Restart attempt $RESTART_COUNT/$MAX_RESTARTS"
        echo "⏳ Waiting ${RESTART_DELAY}s before starting..."
        sleep $RESTART_DELAY
    fi
    
    echo "🚀 Starting JARVIS server..."
    echo ""
    
    # Run the server
    set +e
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python -c "from core.websocket_server import run_server; run_server()" "$@"
    elif command -v uv &> /dev/null; then
        uv run python -c "from core.websocket_server import run_server; run_server()" "$@"
    else
        python -c "from core.websocket_server import run_server; run_server()" "$@"
    fi
    EXIT_CODE=$?
    set -e
    
    echo ""
    
    if [ $EXIT_CODE -eq $RESTART_CODE ]; then
        echo "🔄 Restart signal received from Telegram (exit code $EXIT_CODE)"
        echo "   Server will restart automatically..."
        continue
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Server exited normally (code $EXIT_CODE)"
        exit 0
    else
        echo "❌ Server crashed with exit code $EXIT_CODE"
        echo "   Restarting in ${RESTART_DELAY}s..."
        sleep $RESTART_DELAY
        continue
    fi
done
