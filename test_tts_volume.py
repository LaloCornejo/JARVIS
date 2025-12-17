#!/usr/bin/env python3
"""
Test script to verify TTS volume management
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.voice.tts import TextToSpeech


async def test_tts_volume():
    """Test TTS with volume management"""
    print("Testing TTS volume management...")

    # Create TTS instance
    tts = TextToSpeech()

    # Test short message to see if volume changes
    test_text = "This is a test of TTS volume management."

    print("Playing TTS - listen for volume changes...")
    await tts.play_stream(test_text)

    print("TTS playback completed.")


if __name__ == "__main__":
    asyncio.run(test_tts_volume())
