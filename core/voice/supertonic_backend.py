"""
Supertonic TTS backend — in-process ONNX-based TTS engine.

Install: pip install supertonic
Docs: https://supertone-inc.github.io/supertonic-py/
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from .tts_base import TTSBackend

log = logging.getLogger("jarvis.tts.supertonic")

SUPPORTED_LANGUAGES = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
]

SAMPLE_RATE_SUPERTONIC = 44100
TARGET_SAMPLE_RATE = 24000


def _resample_44k_to_24k(audio: np.ndarray) -> np.ndarray:
    """Resample float32 array from 44.1kHz to 24kHz and convert to int16.

    Uses numpy linear interpolation — no extra dependencies.
    """
    orig_len = audio.shape[-1]
    new_len = int(orig_len * TARGET_SAMPLE_RATE / SAMPLE_RATE_SUPERTONIC)
    orig_indices = np.arange(orig_len)
    new_indices = np.linspace(0, orig_len - 1, new_len)
    resampled = np.interp(new_indices, orig_indices, audio.squeeze())
    return (resampled * 32767).astype(np.int16)


class SupertonicEngine(TTSBackend):
    """TTS backend using supertonic (in-process ONNX inference).

    Parameters
    ----------
    voice : str, optional
        Built-in voice name: M1-M5 or F1-F5. Default "M1".
    custom_voice_path : str, optional
        Path to a Voice Builder JSON export. Overrides ``voice`` when set.
    speed : float, optional
        Speech rate: 0.7 (slow) to 2.0 (fast). Default 1.05.
    steps : int, optional
        Inference steps: 5 (low) to 12 (high). Default 8.
    language : str, optional
        Default language code. Default "en".
    """

    sample_rate: int = TARGET_SAMPLE_RATE

    def __init__(
        self,
        voice: str = "M1",
        custom_voice_path: str | None = None,
        speed: float = 1.05,
        steps: int = 8,
        language: str = "en",
    ):
        self.voice_name = voice
        self.custom_voice_path = custom_voice_path
        self.speed = speed
        self.steps = steps
        self._language = language
        self._tts: Any = None
        self._style: Any = None

    async def _ensure_loaded(self) -> bool:
        """Lazy-init supertonic TTS on first use (model downloads on first run)."""
        if self._tts is not None:
            return True
        try:
            from supertonic import TTS

            self._tts = await asyncio.get_event_loop().run_in_executor(
                None, lambda: TTS(auto_download=True)
            )
            self._load_voice()
            log.info(
                "Supertonic loaded (voice=%s, custom=%s)",
                self.voice_name,
                self.custom_voice_path or "(none)",
            )
            return True
        except Exception as e:
            log.error("Failed to load supertonic: %s", e)
            return False

    def _load_voice(self):
        if self._tts is None:
            return
        if self.custom_voice_path:
            expanded = os.path.expanduser(self.custom_voice_path)
            if os.path.isfile(expanded):
                self._style = self._tts.get_voice_style_from_path(expanded)
                log.info("Loaded custom voice from %s", expanded)
                return
            log.warning("Custom voice path not found: %s, falling back to %s", expanded, self.voice_name)
        self._style = self._tts.get_voice_style(voice_name=self.voice_name)

    async def speak_stream(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[bytes]:
        if not await self._ensure_loaded():
            return

        loop = asyncio.get_event_loop()
        lang = language or self._language

        try:
            wav, _ = await loop.run_in_executor(
                None,
                lambda: self._tts.synthesize(
                    text=text,
                    voice_style=self._style,
                    lang=lang,
                    speed=self.speed,
                    total_steps=self.steps,
                ),
            )
        except Exception as e:
            log.error("Supertonic synthesis failed: %s", e)
            return

        audio_24k = _resample_44k_to_24k(wav)

        chunk_size = 4096
        for i in range(0, len(audio_24k), chunk_size):
            yield audio_24k[i : i + chunk_size].tobytes()

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
        return [{"name": f"M{i}", "gender": "male"} for i in range(1, 6)] + [
            {"name": f"F{i}", "gender": "female"} for i in range(1, 6)
        ]

    async def close(self) -> None:
        self._tts = None
        self._style = None
