#!/usr/bin/env python3
"""
Comprehensive test script for YouTube API functionality
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
from tools.integrations.youtube import YouTubeSearchTool, get_youtube_client


async def test_comprehensive():
    print("=== Comprehensive YouTube API Test ===")

    # Check if API key is loaded
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not found in environment variables")
        return

    print(f"API Key found: {api_key[:8]}...")

    # Test 1: Direct HTTP request
    print("\n--- Test 1: Direct HTTP Request ---")
    async with httpx.AsyncClient() as client:
        params = {
            "key": api_key,
            "q": "test",
            "part": "snippet",
            "maxResults": 1,
            "type": "video",
        }
        try:
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/search", params=params, timeout=30.0
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Found {len(data.get('items', []))} items")
            else:
                print(f"Error Response: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

    # Test 2: YouTube Client
    print("\n--- Test 2: YouTube Client ---")
    youtube_client = get_youtube_client()
    try:
        results = await youtube_client.search("test", max_results=1)
        print(f"Client search returned {len(results)} results")
    except Exception as e:
        print(f"Client search failed: {e}")

    # Test 3: YouTube Tool
    print("\n--- Test 3: YouTube Tool ---")
    tool = YouTubeSearchTool()
    try:
        result = await tool.execute("test", limit=1)
        print(f"Tool success: {result.success}")
        if result.success:
            print(f"Found {len(result.data)} results")
        else:
            print(f"Tool error: {result.error}")
    except Exception as e:
        print(f"Tool exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_comprehensive())
