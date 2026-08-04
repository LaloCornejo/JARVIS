"""
Abstract base class for TTS backends.

Defines the interface all TTS engines must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger("jarvis.tts")


class TTSBackend(ABC):
    """Abstract base for TTS engine implementations.

    Each backend handles audio generation from text. Playback
    (sounddevice) is handled by TextToSpeech in tts.py.
    """

    sample_rate: int = 24000

    @abstractmethod
    def speak_stream(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    async def speak(self, text: str, language: str | None = None) -> bytes:
        ...

    @abstractmethod
    async def speak_to_audio(self, text: str, language: str | None = None) -> bytes:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @abstractmethod
    async def get_speakers(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
