#!/usr/bin/env python3
"""
Simple test to verify volume control works
"""

import asyncio
import sys
import os

# Add the tools directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from tools.system.volume_control import LowerAppVolumesTool, RestoreAppVolumesTool


async def test_volume_control():
    """Test volume control functionality"""
    print("Testing volume control...")

    # Test lowering volumes
    print("Lowering app volumes...")
    lower_tool = LowerAppVolumesTool()
    result = await lower_tool.execute()
    print(f"Lower result: {result.success}")
    if result.data:
        stdout_data = result.data.get("stdout", "{}")
        print(f"Lower stdout: {stdout_data}")

        # Extract volume data
        import json

        try:
            parsed = json.loads(stdout_data)
            if isinstance(parsed, dict) and "stdout" in parsed:
                volume_data = parsed["stdout"]
                print(f"Extracted volume data: {volume_data}")

                # Test restoring volumes
                print("Restoring app volumes...")
                restore_tool = RestoreAppVolumesTool()
                restore_result = await restore_tool.execute(volume_data)
                print(f"Restore result: {restore_result.success}")
            else:
                print("Could not extract volume data")
        except Exception as e:
            print(f"Error parsing JSON: {e}")


if __name__ == "__main__":
    asyncio.run(test_volume_control())
