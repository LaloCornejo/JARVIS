#!/usr/bin/env python3
"""
Simple test script for volume control tools
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.system.volume_control import VolumeUpTool, VolumeDownTool, MuteToggleTool, SetVolumeTool


async def test_volume_tools():
    """Test the volume control tools directly"""
    print("Testing volume control tools directly...")
    
    # Test VolumeUpTool
    print("\n1. Testing VolumeUpTool:")
    volume_up_tool = VolumeUpTool()
    result = await volume_up_tool.execute()
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Data: {result.data}")
    if result.error:
        print(f"   Error: {result.error}")
    
    # Test VolumeDownTool
    print("\n2. Testing VolumeDownTool:")
    volume_down_tool = VolumeDownTool()
    result = await volume_down_tool.execute()
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Data: {result.data}")
    if result.error:
        print(f"   Error: {result.error}")
    
    # Test MuteToggleTool
    print("\n3. Testing MuteToggleTool:")
    mute_toggle_tool = MuteToggleTool()
    result = await mute_toggle_tool.execute()
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Data: {result.data}")
    if result.error:
        print(f"   Error: {result.error}")
    
    # Test SetVolumeTool
    print("\n4. Testing SetVolumeTool:")
    set_volume_tool = SetVolumeTool()
    result = await set_volume_tool.execute(level=75)
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Data: {result.data}")
    if result.error:
        print(f"   Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_volume_tools())