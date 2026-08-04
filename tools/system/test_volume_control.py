#!/usr/bin/env python3
"""
Test script for volume control tools
"""

import asyncio

from tools.system.volume_control import (
    LowerAppVolumesTool,
    MuteToggleTool,
    RestoreAppVolumesTool,
    SetVolumeTool,
    VolumeDownTool,
    VolumeUpTool,
)


async def test_volume_tools():
    """Test the volume control tools"""
    print("Testing volume control tools...")

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

    # Test LowerAppVolumesTool
    print("\n5. Testing LowerAppVolumesTool:")
    lower_tool = LowerAppVolumesTool()
    result = await lower_tool.execute()
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Data: {result.data}")
        # Extract the actual JSON from the stdout
        stdout_data = result.data.get("stdout", "{}")
        if '"stdout":' in stdout_data:
            # Parse the nested JSON structure
            import json

            try:
                parsed = json.loads(stdout_data)
                if isinstance(parsed, dict) and "stdout" in parsed:
                    volume_data = parsed["stdout"]
                    print(f"   Extracted volume data: {volume_data}")
                else:
                    volume_data = "{}"
            except:
                volume_data = "{}"
        else:
            volume_data = stdout_data
    if result.error:
        print(f"   Error: {result.error}")
        volume_data = "{}"

    # Test RestoreAppVolumesTool
    print("\n6. Testing RestoreAppVolumesTool:")
    if volume_data != "{}":
        restore_tool = RestoreAppVolumesTool()
        result = await restore_tool.execute(volume_data)
        print(f"   Success: {result.success}")
        if result.data:
            print(f"   Data: {result.data}")
        if result.error:
            print(f"   Error: {result.error}")
    else:
        print("   Skipping restore test (no volume data to restore)")


if __name__ == "__main__":
    asyncio.run(test_volume_tools())
