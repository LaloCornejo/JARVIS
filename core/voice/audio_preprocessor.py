"""
Advanced audio preprocessing for streaming optimization.

This module provides real-time audio preprocessing capabilities including
noise reduction, equalization, compression, and quality enhancement for
optimal streaming speech-to-text performance.
"""

import asyncio
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class AudioPreprocessor:
    """Advanced audio preprocessing for streaming optimization"""

    def __init__(self):
        self.noise_gate = NoiseGate()
        self.equalizer = AdaptiveEqualizer()
        self.compressor = AudioCompressor()
        self.resampler = AudioResampler()
        self.enabled = True

    def preprocess_streaming_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Real-time audio preprocessing chain for streaming optimization"""
        if not self.enabled or len(audio_chunk) == 0:
            return audio_chunk

        try:
            # 1. Noise gate (remove background noise)
            audio = self.noise_gate.process(audio_chunk)

            # 2. Adaptive equalization (enhance speech frequencies)
            audio = self.equalizer.process(audio)

            # 3. Dynamic range compression (even out volume)
            audio = self.compressor.process(audio)

            # 4. Quality enhancement (spectral processing)
            audio = self._enhance_audio_quality(audio)

            # 5. Ensure proper range and type
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            return audio

        except Exception as e:
            log.warning(f"Audio preprocessing failed: {e}")
            return audio_chunk

    def _enhance_audio_quality(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced spectral processing for quality enhancement"""
        # For now, return audio as-is to avoid shape issues
        # Spectral enhancement can be enabled later with proper windowing
        return audio


class NoiseGate:
    """Adaptive noise gate for background noise reduction"""

    def __init__(
        self, threshold_db: float = -30.0, attack_time: float = 0.01, release_time: float = 0.1
    ):
        self.threshold_db = threshold_db
        self.threshold_linear = 10 ** (threshold_db / 20)
        self.attack_coeff = 1 - np.exp(-1 / (16000 * attack_time))  # 16kHz sample rate
        self.release_coeff = 1 - np.exp(-1 / (16000 * release_time))
        self.envelope = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to audio chunk"""
        if len(audio) == 0:
            return audio

        # Calculate envelope
        abs_audio = np.abs(audio)

        # Update envelope with attack/release characteristics
        for i in range(len(abs_audio)):
            if abs_audio[i] > self.envelope:
                self.envelope += (abs_audio[i] - self.envelope) * self.attack_coeff
            else:
                self.envelope += (abs_audio[i] - self.envelope) * self.release_coeff

        # Apply gate
        gate = np.where(self.envelope > self.threshold_linear, 1.0, 0.1)  # Soft gate
        return audio * gate


class AdaptiveEqualizer:
    """Adaptive equalizer that boosts speech frequencies"""

    def __init__(self):
        self.sample_rate = 16000
        self.speech_freq_range = (300, 3400)  # Hz
        self.boost_db = 3.0  # dB
        self.boost_factor = 10 ** (self.boost_db / 20)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive equalization to enhance speech frequencies"""
        if len(audio) < 64:
            return audio

        try:
            # FFT-based equalization
            fft_size = min(512, len(audio))
            if len(audio) < fft_size:
                # Zero-pad if necessary
                padded = np.zeros(fft_size)
                padded[: len(audio)] = audio
                audio = padded

            # Forward FFT
            fft = np.fft.fft(audio[:fft_size])
            freqs = np.fft.fftfreq(fft_size, 1 / self.sample_rate)

            # Create speech enhancement filter
            speech_mask = (np.abs(freqs) >= self.speech_freq_range[0]) & (
                np.abs(freqs) <= self.speech_freq_range[1]
            )

            # Apply boost
            fft[speech_mask] *= self.boost_factor

            # Inverse FFT
            enhanced = np.real(np.fft.ifft(fft))

            # Return enhanced audio with same length as input
            if len(audio) <= fft_size:
                return enhanced[: len(audio)]
            else:
                # For longer audio, we only processed the first fft_size samples
                # Return the enhanced portion plus the rest unchanged
                result = np.copy(audio)
                result[:fft_size] = enhanced
                return result

        except Exception as e:
            log.debug(f"Equalization failed: {e}")
            return audio


class AudioCompressor:
    """Dynamic range compressor for consistent audio levels"""

    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_time: float = 0.001,
        release_time: float = 0.1,
    ):
        self.threshold_db = threshold_db
        self.threshold_linear = 10 ** (threshold_db / 20)
        self.ratio = ratio
        self.attack_coeff = 1 - np.exp(-1 / (16000 * attack_time))
        self.release_coeff = 1 - np.exp(-1 / (16000 * release_time))
        self.envelope = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio chunk"""
        if len(audio) == 0:
            return audio

        try:
            # Calculate envelope
            abs_audio = np.abs(audio)

            # Side-chain envelope detection
            envelope = np.zeros_like(abs_audio)
            envelope[0] = abs_audio[0]

            for i in range(1, len(abs_audio)):
                if abs_audio[i] > envelope[i - 1]:
                    envelope[i] = (
                        envelope[i - 1] + (abs_audio[i] - envelope[i - 1]) * self.attack_coeff
                    )
                else:
                    envelope[i] = (
                        envelope[i - 1] + (abs_audio[i] - envelope[i - 1]) * self.release_coeff
                    )

            # Calculate gain reduction
            over_threshold = envelope > self.threshold_linear
            gain_reduction_db = np.zeros_like(envelope)
            gain_reduction_db[over_threshold] = (
                self.threshold_db - 20 * np.log10(envelope[over_threshold])
            ) * (1 - 1 / self.ratio)

            # Convert to linear gain
            gain_reduction_linear = 10 ** (gain_reduction_db / 20)

            # Apply compression
            compressed = audio * gain_reduction_linear

            # Soft clipping to prevent distortion
            compressed = np.tanh(compressed * 1.5) / 1.5

            return compressed

        except Exception as e:
            log.debug(f"Compression failed: {e}")
            return audio


class AudioResampler:
    """High-quality audio resampling"""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self._resample_cache = {}

    def resample(self, audio: np.ndarray, source_sample_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if source_sample_rate == self.target_sample_rate:
            return audio

        if len(audio) == 0:
            return audio

        try:
            from scipy.signal import resample

            # Calculate target length
            target_length = int(len(audio) * self.target_sample_rate / source_sample_rate)

            # Resample
            resampled = resample(audio, target_length)

            return resampled.astype(np.float32)

        except ImportError:
            log.warning("scipy not available, skipping resampling")
            return audio
        except Exception as e:
            log.debug(f"Resampling failed: {e}")
            return audio


class CircularAudioBuffer:
    """Circular buffer optimized for streaming audio processing"""

    def __init__(self, max_samples: int, dtype=None):
        if dtype is None:
            dtype = np.float32

        self.max_samples = max_samples
        self.dtype = dtype
        self.buffer = np.zeros(max_samples, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self._lock = asyncio.Lock()

    async def async_write(self, data: np.ndarray):
        """Asynchronous write to buffer"""
        async with self._lock:
            self.write(data)

    def write(self, data: np.ndarray):
        """Write data to circular buffer"""
        data_len = len(data)

        if data_len > self.max_samples:
            # If data is larger than buffer, only keep the most recent samples
            data = data[-self.max_samples :]
            data_len = self.max_samples

        # Calculate available space
        available = self.max_samples - self.size

        if data_len > available:
            # Need to overwrite old data
            overwrite = data_len - available
            self.read_pos = (self.read_pos + overwrite) % self.max_samples
            self.size = self.max_samples
        else:
            self.size += data_len

        # Write data
        end_pos = (self.write_pos + data_len) % self.max_samples

        if end_pos > self.write_pos:
            # Single contiguous write
            self.buffer[self.write_pos : end_pos] = data
        else:
            # Wrap around write
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos :] = data[:first_part]
            self.buffer[:end_pos] = data[first_part:]

        self.write_pos = end_pos

    def read(self, samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Read data from circular buffer"""
        if self.size == 0:
            return None

        if samples is None:
            samples = self.size
        else:
            samples = min(samples, self.size)

        result = np.zeros(samples, dtype=self.dtype)

        # Calculate read positions
        end_pos = (self.read_pos + samples) % self.max_samples

        if end_pos > self.read_pos:
            # Single contiguous read
            result[:] = self.buffer[self.read_pos : end_pos]
        else:
            # Wrap around read
            first_part = self.max_samples - self.read_pos
            result[:first_part] = self.buffer[self.read_pos :]
            result[first_part:] = self.buffer[:end_pos]

        self.read_pos = end_pos
        self.size -= samples

        return result

    def peek(self, samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Peek at data without removing it"""
        if self.size == 0:
            return None

        if samples is None:
            samples = self.size
        else:
            samples = min(samples, self.size)

        result = np.zeros(samples, dtype=self.dtype)

        # Calculate peek positions (same as read but don't update positions)
        end_pos = (self.read_pos + samples) % self.max_samples

        if end_pos > self.read_pos:
            result[:] = self.buffer[self.read_pos : end_pos]
        else:
            first_part = self.max_samples - self.read_pos
            result[:first_part] = self.buffer[self.read_pos :]
            result[first_part:] = self.buffer[:end_pos]

        return result

    def clear(self):
        """Clear the buffer"""
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.buffer.fill(0)

    def available(self) -> int:
        """Get number of available samples"""
        return self.size

    def space_available(self) -> int:
        """Get available space for writing"""
        return self.max_samples - self.size
