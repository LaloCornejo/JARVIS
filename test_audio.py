import sounddevice as sd
import numpy as np
import time


def test_audio(device=12, duration=5):
    print(f"Testing audio input on device {device} for {duration} seconds...")
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    with sd.InputStream(device=device, channels=1, samplerate=16000, callback=callback):
        time.sleep(duration)

    if audio_data:
        full_audio = np.concatenate(audio_data)
        rms = np.sqrt(np.mean(full_audio**2))
        print(f"Audio captured: {len(full_audio)} samples, RMS: {rms:.4f}")
        if rms > 0.01:
            print("Audio input seems working.")
        else:
            print("Very low audio level - check microphone.")
    else:
        print("No audio data captured.")


if __name__ == "__main__":
    test_audio()

