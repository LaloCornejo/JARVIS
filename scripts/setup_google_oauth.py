#!/usr/bin/env python3
"""
Google OAuth Setup Script for JARVIS

This script guides you through authenticating with Google APIs (Gmail & Calendar).

Prerequisites:
1. Create a Google Cloud project at https://console.cloud.google.com/
2. Enable Gmail API and Google Calendar API
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download the credentials or note the Client ID and Client Secret

Usage:
    python scripts/setup_google_oauth.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
GMAIL_TOKEN_PATH = DATA_DIR / "gmail_token.json"
CALENDAR_TOKEN_PATH = DATA_DIR / "google_calendar_token.json"

REDIRECT_URI = "http://localhost:8888/callback"
TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]

auth_code: str | None = None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        query = parse_qs(urlparse(self.path).query)

        if "code" in query:
            auth_code = query["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: green;">Authentication Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        elif "error" in query:
            auth_code = None
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            error = query.get("error", ["Unknown error"])[0]
            self.wfile.write(
                f"""
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: red;">Authentication Failed</h1>
                    <p>Error: {error}</p>
                </body>
                </html>
            """.encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def get_auth_url(client_id: str, scopes: list[str]) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_token(code: str, client_id: str, client_secret: str) -> dict | None:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
        )
        if response.status_code == 200:
            return response.json()
        print(f"Error exchanging code: {response.status_code}")
        print(response.text)
        return None


def save_token(token_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
            }
        )
    )
    print(f"Token saved to {path}")


def wait_for_callback(timeout: int = 120) -> str | None:
    global auth_code
    auth_code = None

    server = HTTPServer(("localhost", 8888), OAuthCallbackHandler)
    server.timeout = timeout

    def serve():
        while auth_code is None:
            server.handle_request()

    thread = Thread(target=serve, daemon=True)
    thread.start()

    print("\nWaiting for authentication (timeout: 2 minutes)...")
    thread.join(timeout=timeout)

    server.server_close()
    return auth_code


async def authenticate_service(
    service_name: str,
    scopes: list[str],
    token_path: Path,
    client_id: str,
    client_secret: str,
) -> bool:
    global auth_code
    auth_code = None

    print(f"\n{'=' * 50}")
    print(f"Setting up {service_name}")
    print("=" * 50)

    if token_path.exists():
        response = input(f"Token already exists at {token_path}. Re-authenticate? (y/N): ")
        if response.lower() != "y":
            print(f"Skipping {service_name}")
            return True

    auth_url = get_auth_url(client_id, scopes)

    print(f"\nOpening browser for {service_name} authentication...")
    print("If browser doesn't open, visit this URL manually:")
    print(f"\n{auth_url}\n")

    webbrowser.open(auth_url)

    code = wait_for_callback()

    if not code:
        print(f"Failed to get authorization code for {service_name}")
        return False

    print("Got authorization code, exchanging for token...")

    token_data = await exchange_code_for_token(code, client_id, client_secret)

    if not token_data:
        print(f"Failed to get token for {service_name}")
        return False

    save_token(token_data, token_path)
    print(f"{service_name} authentication successful!")
    return True


async def verify_gmail_token() -> bool:
    if not GMAIL_TOKEN_PATH.exists():
        return False

    token_data = json.loads(GMAIL_TOKEN_PATH.read_text())
    access_token = token_data.get("access_token")

    if not access_token:
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/profile",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Gmail: Connected as {data.get('emailAddress', 'unknown')}")
            return True
        return False


async def verify_calendar_token() -> bool:
    if not CALENDAR_TOKEN_PATH.exists():
        return False

    token_data = json.loads(CALENDAR_TOKEN_PATH.read_text())
    access_token = token_data.get("access_token")

    if not access_token:
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Calendar: Connected to {data.get('summary', 'primary calendar')}")
            return True
        return False


async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           JARVIS - Google OAuth Setup                        ║
╠══════════════════════════════════════════════════════════════╣
║  This script will help you authenticate with:                ║
║    • Gmail API (read, send, modify emails)                   ║
║    • Google Calendar API (read, create, delete events)       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("Prerequisites:")
    print("  1. Create a Google Cloud project")
    print("  2. Enable Gmail API and Google Calendar API")
    print("  3. Create OAuth 2.0 credentials (Desktop app)")
    print("  4. Have your Client ID and Client Secret ready")
    print()

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    if client_id and client_secret:
        print("Found credentials in environment variables.")
        use_env = input("Use environment credentials? (Y/n): ")
        if use_env.lower() == "n":
            client_id = ""
            client_secret = ""

    if not client_id:
        client_id = input("Enter your Google Client ID: ").strip()
    if not client_secret:
        client_secret = input("Enter your Google Client Secret: ").strip()

    if not client_id or not client_secret:
        print("Error: Client ID and Client Secret are required.")
        sys.exit(1)

    print("\nWhich services do you want to authenticate?")
    print("  1. Gmail only")
    print("  2. Calendar only")
    print("  3. Both Gmail and Calendar")

    choice = input("\nEnter choice (1/2/3) [3]: ").strip() or "3"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if choice in ("1", "3"):
        await authenticate_service(
            "Gmail",
            GMAIL_SCOPES,
            GMAIL_TOKEN_PATH,
            client_id,
            client_secret,
        )

    if choice in ("2", "3"):
        await authenticate_service(
            "Google Calendar",
            CALENDAR_SCOPES,
            CALENDAR_TOKEN_PATH,
            client_id,
            client_secret,
        )

    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    gmail_ok = await verify_gmail_token()
    calendar_ok = await verify_calendar_token()

    if not gmail_ok and GMAIL_TOKEN_PATH.exists():
        print("  Gmail: Token exists but may need refresh")
    elif not gmail_ok:
        print("  Gmail: Not configured")

    if not calendar_ok and CALENDAR_TOKEN_PATH.exists():
        print("  Calendar: Token exists but may need refresh")
    elif not calendar_ok:
        print("  Calendar: Not configured")

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)

    if not os.environ.get("GOOGLE_CLIENT_ID"):
        print("\nTip: Add these to your environment to avoid re-entering:")
        print(f"  set GOOGLE_CLIENT_ID={client_id}")
        print(f"  set GOOGLE_CLIENT_SECRET={client_secret}")
        print("\nOr add to a .env file in the project root.")


if __name__ == "__main__":
    asyncio.run(main())
