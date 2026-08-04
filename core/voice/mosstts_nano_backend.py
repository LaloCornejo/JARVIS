from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from .tts_base import TTSBackend

log = logging.getLogger("jarvis.tts.mosstts_nano")

NANO_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 24000


def _resample_48k_to_24k(audio: np.ndarray) -> np.ndarray:
    orig_len = audio.shape[-1]
    new_len = int(orig_len * TARGET_SAMPLE_RATE / NANO_SAMPLE_RATE)
    orig_indices = np.arange(orig_len)
    new_indices = np.linspace(0, orig_len - 1, new_len)
    return np.interp(new_indices, orig_indices, audio.squeeze()).astype(np.float32)


def _load_audio_soundfile(path: str, target_sr: int, target_channels: int) -> np.ndarray:
    """Load audio with soundfile (no torchaudio/torchcodec dependency)."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Reference audio not found: {resolved}")
    import soundfile as sf

    data, sr = sf.read(str(resolved), always_2d=True)
    # data shape: (samples, channels), float64 or float32
    data = data.astype(np.float32)

    if data.shape[1] > target_channels:
        data = data.mean(axis=1, keepdims=True)
    elif data.shape[1] < target_channels:
        data = np.repeat(data, target_channels, axis=1)

    if sr != target_sr:
        orig_len = data.shape[0]
        new_len = int(orig_len * target_sr / sr)
        orig_indices = np.arange(orig_len)
        new_indices = np.linspace(0, orig_len - 1, new_len)
        resampled = np.zeros((new_len, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            resampled[:, ch] = np.interp(new_indices, orig_indices, data[:, ch])
        data = resampled

    # shape to (1, channels, samples) — matches torchaudio output format
    return data.T[np.newaxis, :].astype(np.float32, copy=False)


class MossttsNanoEngine(TTSBackend):
    sample_rate: int = TARGET_SAMPLE_RATE

    def __init__(
        self,
        voice: str = "Ava",
        prompt_audio_path: str | None = None,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
    ):
        self.voice_name = voice
        self.prompt_audio_path = prompt_audio_path
        self.audio_temperature = audio_temperature
        self.audio_top_p = audio_top_p
        self.audio_top_k = audio_top_k
        self.audio_repetition_penalty = audio_repetition_penalty
        self.max_new_frames = max_new_frames
        self.voice_clone_max_text_tokens = voice_clone_max_text_tokens
        self._runtime: Any = None

    async def _ensure_loaded(self) -> bool:
        if self._runtime is not None:
            return True

        try:
            from onnx_tts_runtime import OnnxTtsRuntime
        except ImportError:
            log.error(
                "MOSS-TTS-Nano ONNX runtime not installed. "
                "Clone from: git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git && "
                "cd MOSS-TTS-Nano && pip install -e ."
            )
            return False

        loop = asyncio.get_event_loop()
        try:
            self._runtime = await loop.run_in_executor(
                None,
                lambda: self._create_runtime(OnnxTtsRuntime),
            )
            log.info(
                "MOSS-TTS-Nano ONNX runtime loaded (voice=%s, models=%s)",
                self.voice_name,
                self._runtime.model_dir,
            )
            return True
        except Exception as e:
            log.error("Failed to load MOSS-TTS-Nano ONNX runtime: %s", e)
            return False

    def _create_runtime(self, RuntimeClass: type) -> Any:
        runtime = RuntimeClass(
            model_dir=None,
            max_new_frames=self.max_new_frames,
            do_sample=True,
        )
        # Monkey-patch _load_reference_audio to use soundfile instead of torchaudio
        runtime._load_reference_audio = self._patched_load_reference_audio.__get__(runtime, RuntimeClass)
        return runtime

    def _patched_load_reference_audio(self, reference_audio_path: str | os.PathLike) -> np.ndarray:
        target_sr = int(self.codec_meta["codec_config"]["sample_rate"])  # type: ignore[attr-defined]
        target_channels = int(self.codec_meta["codec_config"]["channels"])  # type: ignore[attr-defined]
        return _load_audio_soundfile(str(Path(reference_audio_path).expanduser().resolve()), target_sr, target_channels)

    def _waveform_to_int16_bytes(self, waveform: np.ndarray) -> bytes:
        if waveform.ndim == 2 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)
        elif waveform.ndim == 2:
            waveform = waveform.ravel()
        resampled = _resample_48k_to_24k(waveform)
        return (resampled * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

    async def speak_stream(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[bytes]:
        if not await self._ensure_loaded():
            return

        loop = asyncio.get_event_loop()
        try:
            # Offload ENTIRE synthesis + post-processing to the executor so
            # CPU-bound work (inference, resampling, int16 conversion) never
            # blocks the event loop.
            audio_bytes = await loop.run_in_executor(
                None,
                lambda: self._synthesize_and_encode(text),
            )
        except Exception as e:
            log.error("MOSS-TTS-Nano synthesis failed: %s", e)
            return

        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    def _synthesize_and_encode(self, text: str) -> bytes:
        """Run synthesis + waveform-to-bytes in one shot (called from executor)."""
        result = self._runtime.synthesize(
            text=text,
            voice=self.voice_name if not self.prompt_audio_path else None,
            prompt_audio_path=self.prompt_audio_path,
            do_sample=True,
            streaming=False,
            max_new_frames=self.max_new_frames,
            voice_clone_max_text_tokens=self.voice_clone_max_text_tokens,
            enable_wetext=False,
            enable_normalize_tts_text=True,
        )
        waveform = result["waveform"]
        return self._waveform_to_int16_bytes(waveform)

    async def speak(self, text: str, language: str | None = None) -> bytes:
        chunks = b""
        async for chunk in self.speak_stream(text, language):
            chunks += chunk
        return chunks

    async def speak_to_audio(self, text: str, language: str | None = None) -> bytes:
        return await self.speak(text, language)

    async def health_check(self) -> bool:
        return await self._ensure_loaded()

    async def get_speakers(self) -> list[dict[str, Any]]:
        if not await self._ensure_loaded():
            return []
        try:
            voices = self._runtime.list_builtin_voices()
            return [{"name": v["voice"], "description": v.get("description", "")} for v in voices]
        except Exception as e:
            log.error("Failed to list Nano voices: %s", e)
            return []

    async def close(self) -> None:
        self._runtime = None
