#!/usr/bin/env python3
"""
Test script to check different YouTube API endpoints
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

import asyncio
import httpx


async def test_api_endpoints():
    print("=== Testing Different YouTube API Endpoints ===")

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not found in environment variables")
        return

    print(f"API Key: {api_key[:8]}...")

    endpoints = [
        (
            "Search API",
            "https://www.googleapis.com/youtube/v3/search",
            {"q": "test", "part": "snippet", "maxResults": 1},
        ),
        (
            "Videos API",
            "https://www.googleapis.com/youtube/v3/videos",
            {"id": "dQw4w9WgXcQ", "part": "snippet"},
        ),
        (
            "Channels API",
            "https://www.googleapis.com/youtube/v3/channels",
            {"forUsername": "google", "part": "snippet"},
        ),
    ]

    async with httpx.AsyncClient() as client:
        for name, url, params in endpoints:
            print(f"\n--- Testing {name} ---")
            params["key"] = api_key

            try:
                response = await client.get(url, params=params, timeout=30.0)
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"Success! Items: {len(data.get('items', []))}")
                else:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "Unknown error")
                    print(f"Error: {error_message}")
            except Exception as e:
                print(f"Exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_api_endpoints())
