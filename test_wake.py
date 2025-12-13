#!/usr/bin/env python3
"""
Standalone wake word test script.
Runs wake word detection for "hey jarvis" without starting full JARVIS.
Press Ctrl+C to stop.
"""

import sys
import time
sys.path.append('.')

from core.voice.wake_word import WakeWordDetector
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def wake_callback():
    print("\n*** WAKE WORD DETECTED: hey jarvis ***\n")

def main():
    print("Starting wake word test...")
    print("Say 'hey jarvis' into the microphone.")
    print("Press Ctrl+C to stop.")

    # Use default device or specify
    detector = WakeWordDetector(threshold=0.3)
    detector.start(wake_callback)

    try:
        input("Press Enter to stop...\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        detector.stop()
        print("Wake word test stopped.")

if __name__ == "__main__":
    main()