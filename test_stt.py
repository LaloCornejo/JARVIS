import sys
import time
sys.path.append('.')

from core.voice.stt import SpeechToText
import numpy as np
import sounddevice as sd

def test_stt():
    print("Testing STT...")
    stt = SpeechToText()
    print("STT initialized")

    # Record real audio
    device = 12  # Realtek DirectSound
    duration = 3  # seconds
    sample_rate = 16000

    print(f"Recording {duration} seconds of audio from device {device}...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        rms = np.sqrt(np.mean(audio**2))
        print(f"Recorded audio: {len(audio)} samples, RMS: {rms:.4f}")

        if rms > 0.01:
            print("Transcribing...")
            result = stt.transcribe(audio)
            print(f"Transcription result: '{result}'")
            if result and len(result.strip()) > 0:
                print("STT worked!")
            else:
                print("STT returned empty or no speech detected")
        else:
            print("Audio too quiet, check microphone")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_stt()