import sounddevice as sd
import numpy as np
import time

def record_and_playback(device=1, duration=3, samplerate=44100):
    print(f"Recording from device {device} for {duration} seconds at {samplerate} Hz...")
    
    # Record
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, device=device)
    sd.wait()
    
    rms = np.sqrt(np.mean(recording**2))
    print(f"Recorded RMS: {rms:.4f}")
    
    if rms > 0.01:
        print("Playing back...")
        sd.play(recording, samplerate=samplerate)
        sd.wait()
    else:
        print("Audio too quiet to playback.")

if __name__ == "__main__":
    record_and_playback()