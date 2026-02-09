"""
JARVIS Server Wrapper - Handles automatic restarts from Telegram

Usage:
    python jarvis_wrapper.py [options]

This wrapper will:
1. Start the JARVIS server
2. Monitor for restart signals (exit code 42)
3. Automatically restart when triggered via Telegram /restart command
4. Handle crashes gracefully with exponential backoff
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

RESTART_CODE = 42
MAX_RESTARTS = 10
RESTART_DELAY = 3


def get_python_executable() -> str:
    """Get the appropriate Python executable."""
    # Check for virtual environment
    venv_python = Path(".venv/bin/python")
    if sys.platform == "win32":
        venv_python = Path(".venv/Scripts/python.exe")

    if venv_python.exists():
        return str(venv_python)

    # Fall back to sys.executable
    return sys.executable


def run_server(args: list[str]) -> int:
    """Run the JARVIS server and return exit code."""
    python = get_python_executable()

    cmd = [python, "-m", "core.websocket_server", *args]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error running server: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="JARVIS Server Wrapper with auto-restart support")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram bot")

    # Parse only known args to pass rest to server
    args, remaining = parser.parse_known_args()

    print("=" * 48)
    print("JARVIS Server Wrapper")
    print(f"Auto-restart enabled (exit code {RESTART_CODE})")
    print("=" * 48)
    print()

    restart_count = 0

    while True:
        restart_count += 1

        if restart_count > MAX_RESTARTS:
            print(f"! Maximum restart attempts ({MAX_RESTARTS}) reached. Giving up.")
            sys.exit(1)

        if restart_count > 1:
            print(f"* Restart attempt {restart_count}/{MAX_RESTARTS}")
            print(f"> Waiting {RESTART_DELAY}s before starting...")
            time.sleep(RESTART_DELAY)

        print(">>> Starting JARVIS server...")
        print()

        # Build server args
        server_args = [
            f"--host={args.host}",
            f"--port={args.port}",
        ]
        if args.no_telegram:
            server_args.append("--no-telegram")
        server_args.extend(remaining)

        # Run server
        exit_code = run_server(server_args)

        print()

        if exit_code == RESTART_CODE:
            print(f"🔄 Restart signal received from Telegram (exit code {exit_code})")
            print("   Server will restart automatically...")
            continue
        elif exit_code == 0:
            print(f"✅ Server exited normally (code {exit_code})")
            sys.exit(0)
        else:
            print(f"❌ Server crashed with exit code {exit_code}")
            print(f"   Restarting in {RESTART_DELAY}s...")
            time.sleep(RESTART_DELAY)
            continue


if __name__ == "__main__":
    main()
