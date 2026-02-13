from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("jarvis.discord_voice.opus")


def _setup_opus_dll():
    """Setup Opus DLL path on Windows."""
    if sys.platform == "win32":
        project_root = Path(__file__).parent.parent.parent
        dll_path = project_root / "opus.dll"

        log.debug(f"Looking for opus.dll at: {dll_path}")

        if dll_path.exists():
            log.debug(f"Found opus.dll: {dll_path.stat().st_size} bytes")
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(project_root))
                    log.debug(f"Added {project_root} to DLL search path")
            except Exception as e:
                log.warning(f"Could not add DLL directory: {e}")

            try:
                ctypes.CDLL(str(dll_path))
                log.info(f"Successfully loaded opus.dll from {dll_path}")
                return True
            except Exception as e:
                log.error(f"Failed to load opus.dll: {e}")
                return False
        else:
            log.warning(f"opus.dll not found at {dll_path}")
            return False
    return True


# Setup DLL path before importing opuslib
_setup_opus_dll()


class OpusEncoder:
    """Encodes PCM audio to Opus format for Discord voice channels.

    Discord requires:
    - Sample rate: 48000 Hz
    - Channels: Mono (1)
    - Frame size: 960 samples (20ms)
    - Format: Opus encoded frames
    """

    DISCORD_SAMPLE_RATE = 48000
    DISCORD_FRAME_SIZE = 960
    DISCORD_CHANNELS = 1

    def __init__(self):
        self._encoder: Optional[object] = None
        self._available = False
        self._init_encoder()

    def is_available(self) -> bool:
        """Check if Opus encoder is available."""
        return self._available

    def _init_encoder(self) -> None:
        """Initialize the Opus encoder."""
        try:
            import opuslib

            self._encoder = opuslib.Encoder(
                self.DISCORD_SAMPLE_RATE, self.DISCORD_CHANNELS, opuslib.APPLICATION_AUDIO
            )
            self._available = True
            log.info(
                "Opus encoder initialized: %d Hz, %d channel(s)",
                self.DISCORD_SAMPLE_RATE,
                self.DISCORD_CHANNELS,
            )
        except ImportError:
            log.warning("opuslib not installed. Run: pip install opuslib-next")
            self._available = False
        except Exception as e:
            log.error(f"Failed to initialize Opus encoder: {e}")
            self._available = False

    def encode_pcm(self, pcm_data: np.ndarray) -> List[bytes]:
        """Encode numpy PCM array to Opus frames.

        Args:
            pcm_data: Audio data as numpy array (int16 or float32)

        Returns:
            List of Opus-encoded frames
        """
        if self._encoder is None:
            raise RuntimeError("Encoder not initialized")

        # Convert to int16 if needed
        if pcm_data.dtype == np.float32:
            pcm_data = (pcm_data * 32767).astype(np.int16)
        elif pcm_data.dtype != np.int16:
            pcm_data = pcm_data.astype(np.int16)

        # Ensure mono
        if len(pcm_data.shape) > 1:
            pcm_data = pcm_data.mean(axis=1).astype(np.int16)

        # Encode in 20ms frames
        frames = []
        frame_samples = self.DISCORD_FRAME_SIZE

        for i in range(0, len(pcm_data), frame_samples):
            chunk = pcm_data[i : i + frame_samples]

            # Pad last frame if needed
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)))

            # Encode
            opus_frame = self._encoder.encode(chunk.tobytes(), frame_samples)
            frames.append(opus_frame)

        return frames

    def encode_bytes(self, pcm_bytes: bytes) -> List[bytes]:
        """Encode raw PCM bytes to Opus frames.

        Args:
            pcm_bytes: Raw PCM data (16-bit, little-endian)

        Returns:
            List of Opus-encoded frames
        """
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        return self.encode_pcm(pcm_array)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    try:
        from scipy import signal

        # Calculate new length
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        # Resample
        resampled = signal.resample(audio, new_length)
        return resampled.astype(audio.dtype)
    except ImportError:
        log.warning("scipy not available for resampling, using linear interpolation")
        # Fallback to simple linear interpolation
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)
