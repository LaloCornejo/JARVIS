#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from tools.integrations.youtube import YouTubeSearchTool


async def test_search():
    tool = YouTubeSearchTool()
    result = await tool.execute("한갱", limit=5)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    if result.success and result.data:
        print(f"\nFound {len(result.data)} videos:")
        for i, video in enumerate(result.data):
            print(f"{i + 1}. {video['url']}")


if __name__ == "__main__":
    asyncio.run(test_search())
