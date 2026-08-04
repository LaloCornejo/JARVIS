#!/usr/bin/env python3
"""
JARVIS Integration Setup Script

Sets up authentication for all external integrations:
- Spotify (OAuth 2.0)
- GitHub (Personal Access Token)
- Discord (Bot Token)
- YouTube (API Key)

For Google (Gmail/Calendar), use: python scripts/setup_google_oauth.py

Usage:
    python scripts/setup_integrations.py
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
from urllib.parse import parse_qs, urlencode, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
ENV_FILE = Path(__file__).parent.parent / ".env"
SPOTIFY_TOKEN_PATH = DATA_DIR / "spotify_token.json"

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
REDIRECT_URI = "http://localhost:8888/callback"

SPOTIFY_SCOPES = [
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "playlist-modify-public",
    "playlist-modify-private",
]

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


def load_env() -> dict[str, str]:
    env_vars = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def save_env(env_vars: dict[str, str]) -> None:
    existing_lines = []
    existing_keys = set()

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.partition("=")[0].strip()
                if key in env_vars:
                    existing_lines.append(f"{key}={env_vars[key]}")
                    existing_keys.add(key)
                else:
                    existing_lines.append(line)
            else:
                existing_lines.append(line)

    for key, value in env_vars.items():
        if key not in existing_keys:
            existing_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(existing_lines) + "\n")
    print(f"Saved to {ENV_FILE}")


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


async def setup_spotify() -> bool:
    global auth_code
    auth_code = None

    print("\n" + "=" * 50)
    print("Spotify Setup")
    print("=" * 50)
    print("\nPrerequisites:")
    print("  1. Go to https://developer.spotify.com/dashboard")
    print("  2. Create an app (or use existing)")
    print("  3. Add http://localhost:8888/callback to Redirect URIs")
    print("  4. Get your Client ID and Client Secret")
    print()

    if SPOTIFY_TOKEN_PATH.exists():
        response = input("Spotify token already exists. Re-authenticate? (y/N): ")
        if response.lower() != "y":
            print("Skipping Spotify")
            return True

    env_vars = load_env()
    client_id = os.environ.get("SPOTIFY_CLIENT_ID") or env_vars.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET") or env_vars.get(
        "SPOTIFY_CLIENT_SECRET", ""
    )

    if client_id and client_secret:
        print(f"Found Spotify credentials (Client ID: {client_id[:8]}...)")
        use_existing = input("Use existing credentials? (Y/n): ")
        if use_existing.lower() == "n":
            client_id = ""
            client_secret = ""

    if not client_id:
        client_id = input("Enter Spotify Client ID: ").strip()
    if not client_secret:
        client_secret = input("Enter Spotify Client Secret: ").strip()

    if not client_id or not client_secret:
        print("Error: Client ID and Secret are required")
        return False

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SPOTIFY_SCOPES),
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

    print("\nOpening browser for Spotify authentication...")
    print("If browser doesn't open, visit this URL:")
    print(f"\n{auth_url}\n")

    webbrowser.open(auth_url)
    code = wait_for_callback()

    if not code:
        print("Failed to get authorization code")
        return False

    print("Got authorization code, exchanging for token...")

    from base64 import b64encode

    auth_header = b64encode(f"{client_id}:{client_secret}".encode()).decode()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Authorization": f"Basic {auth_header}"},
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False

        token_data = response.json()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPOTIFY_TOKEN_PATH.write_text(
        json.dumps(
            {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
            },
            indent=2,
        )
    )

    print(f"Token saved to {SPOTIFY_TOKEN_PATH}")

    save_to_env = input("\nSave Client ID/Secret to .env file? (Y/n): ")
    if save_to_env.lower() != "n":
        env_vars["SPOTIFY_CLIENT_ID"] = client_id
        env_vars["SPOTIFY_CLIENT_SECRET"] = client_secret
        save_env(env_vars)

    print("Spotify setup complete!")
    return True


async def verify_spotify() -> bool:
    if not SPOTIFY_TOKEN_PATH.exists():
        return False

    token_data = json.loads(SPOTIFY_TOKEN_PATH.read_text())
    access_token = token_data.get("access_token")

    if not access_token:
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Spotify: Connected as {data.get('display_name', data.get('id', 'unknown'))}")
            return True
        return False


def setup_github() -> bool:
    print("\n" + "=" * 50)
    print("GitHub Setup")
    print("=" * 50)
    print("\nPrerequisites:")
    print("  1. Go to https://github.com/settings/tokens")
    print("  2. Generate new token (classic) with required scopes:")
    print("     - repo (for private repos)")
    print("     - read:user, user:email (for user info)")
    print()

    env_vars = load_env()
    existing_token = os.environ.get("GITHUB_TOKEN") or env_vars.get("GITHUB_TOKEN", "")

    if existing_token:
        print(f"Found existing GitHub token: {existing_token[:8]}...")
        update = input("Update token? (y/N): ")
        if update.lower() != "y":
            print("Keeping existing token")
            return True

    token = input("Enter GitHub Personal Access Token: ").strip()

    if not token:
        print("No token provided, skipping")
        return False

    env_vars["GITHUB_TOKEN"] = token
    save_env(env_vars)

    print("GitHub token saved!")
    return True


async def verify_github() -> bool:
    env_vars = load_env()
    token = os.environ.get("GITHUB_TOKEN") or env_vars.get("GITHUB_TOKEN", "")

    if not token:
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}"},
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  GitHub: Connected as {data.get('login', 'unknown')}")
            return True
        return False


def setup_discord() -> bool:
    print("\n" + "=" * 50)
    print("Discord Bot Setup")
    print("=" * 50)
    print("\nPrerequisites:")
    print("  1. Go to https://discord.com/developers/applications")
    print("  2. Create application and add a Bot")
    print("  3. Copy the Bot Token")
    print("  4. Invite bot to your server with required permissions")
    print()

    env_vars = load_env()
    existing_token = os.environ.get("DISCORD_TOKEN") or env_vars.get("DISCORD_TOKEN", "")

    if existing_token:
        print(f"Found existing Discord token: {existing_token[:8]}...")
        update = input("Update token? (y/N): ")
        if update.lower() != "y":
            print("Keeping existing token")
            return True

    token = input("Enter Discord Bot Token: ").strip()

    if not token:
        print("No token provided, skipping")
        return False

    env_vars["DISCORD_TOKEN"] = token
    save_env(env_vars)

    print("Discord token saved!")
    return True


def setup_youtube() -> bool:
    print("\n" + "=" * 50)
    print("YouTube API Setup")
    print("=" * 50)
    print("\nPrerequisites:")
    print("  1. Go to https://console.cloud.google.com/apis/credentials")
    print("  2. Create an API Key")
    print("  3. (Optional) Restrict key to YouTube Data API v3")
    print()

    env_vars = load_env()
    existing_key = os.environ.get("YOUTUBE_API_KEY") or env_vars.get("YOUTUBE_API_KEY", "")

    if existing_key:
        print(f"Found existing YouTube API key: {existing_key[:8]}...")
        update = input("Update key? (y/N): ")
        if update.lower() != "y":
            print("Keeping existing key")
            return True

    key = input("Enter YouTube API Key: ").strip()

    if not key:
        print("No key provided, skipping")
        return False

    env_vars["YOUTUBE_API_KEY"] = key
    save_env(env_vars)

    print("YouTube API key saved!")
    return True


async def verify_youtube() -> bool:
    env_vars = load_env()
    api_key = os.environ.get("YOUTUBE_API_KEY") or env_vars.get("YOUTUBE_API_KEY", "")

    if not api_key:
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "id", "chart": "mostPopular", "maxResults": 1, "key": api_key},
        )
        if response.status_code == 200:
            print("  YouTube: API key valid")
            return True
        return False


def print_menu():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           JARVIS - Integration Setup                         ║
╠══════════════════════════════════════════════════════════════╣
║  Configure authentication for external services:             ║
║                                                              ║
║    1. Spotify      (OAuth 2.0 - music control)              ║
║    2. GitHub       (PAT - repo access)                      ║
║    3. Discord      (Bot token - messaging)                  ║
║    4. YouTube      (API key - video search)                 ║
║    5. All of the above                                      ║
║    6. Verify existing integrations                          ║
║    0. Exit                                                  ║
║                                                              ║
║  For Google (Gmail/Calendar):                               ║
║    Run: python scripts/setup_google_oauth.py                ║
╚══════════════════════════════════════════════════════════════╝
    """)


async def verify_all():
    print("\n" + "=" * 50)
    print("Integration Status")
    print("=" * 50)

    await verify_spotify() or print("  Spotify: Not configured")
    await verify_github() or print("  GitHub: Not configured")
    await verify_youtube() or print("  YouTube: Not configured")

    env_vars = load_env()
    discord_token = os.environ.get("DISCORD_TOKEN") or env_vars.get("DISCORD_TOKEN", "")
    if discord_token:
        print(f"  Discord: Token configured ({discord_token[:8]}...)")
    else:
        print("  Discord: Not configured")

    gmail_path = DATA_DIR / "gmail_token.json"
    calendar_path = DATA_DIR / "google_calendar_token.json"
    if gmail_path.exists():
        print("  Gmail: Token exists")
    else:
        print("  Gmail: Not configured (run setup_google_oauth.py)")
    if calendar_path.exists():
        print("  Calendar: Token exists")
    else:
        print("  Calendar: Not configured (run setup_google_oauth.py)")


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print_menu()
        choice = input("Enter choice: ").strip()

        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            await setup_spotify()
        elif choice == "2":
            setup_github()
        elif choice == "3":
            setup_discord()
        elif choice == "4":
            setup_youtube()
        elif choice == "5":
            await setup_spotify()
            setup_github()
            setup_discord()
            setup_youtube()
        elif choice == "6":
            await verify_all()
        else:
            print("Invalid choice")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
