from __future__ import annotations

import logging
from collections.abc import Callable
from typing import AsyncIterator

import httpx
import numpy as np

log = logging.getLogger("jarvis.tts")


class TextToSpeech:
    def __init__(
        self,
        base_url: str = "http://localhost:8020",
        speaker: str = "duckie",
        language: str = "en",
        sample_rate: int = 24000,
    ):
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self.language = language
        self.sample_rate = sample_rate
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except RuntimeError:
                pass
            self._client = None

    async def speak_stream(self, text: str, language: str | None = None) -> AsyncIterator[bytes]:
        log.info("TTS streaming: %s", text[:50] + "..." if len(text) > 50 else text)
        client = await self._get_client()
        params = {
            "text": text,
            "speaker_wav": self.speaker,
            "language": language or self.language,
        }
        async with client.stream(
            "GET",
            f"{self.base_url}/tts_stream",
            params=params,
        ) as response:
            response.raise_for_status()
            chunk_count = 0
            async for chunk in response.aiter_bytes(chunk_size=4096):
                chunk_count += 1
                yield chunk
            log.debug("TTS stream complete: %d chunks", chunk_count)

    async def speak(self, text: str, language: str | None = None) -> bytes:
        chunks = []
        async for chunk in self.speak_stream(text, language):
            chunks.append(chunk)
        return b"".join(chunks)

    async def speak_to_audio(self, text: str, language: str | None = None) -> bytes:
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/tts_to_audio/",
            json={
                "text": text,
                "speaker_wav": self.speaker,
                "language": language or self.language,
            },
        )
        response.raise_for_status()
        return response.content

    async def play_stream(self, text: str, language: str | None = None) -> None:
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
        )
        stream.start()

        try:
            async for chunk in self.speak_stream(text, language):
                audio = np.frombuffer(chunk, dtype=np.int16)
                if len(audio) > 0:
                    stream.write(audio)
        finally:
            stream.stop()
            stream.close()

    async def play_stream_interruptible(
        self,
        text: str,
        should_stop: Callable[[], bool],
        language: str | None = None,
    ) -> None:
        import sounddevice as sd

        log.info("Starting interruptible playback")
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
        )
        stream.start()

        try:
            async for chunk in self.speak_stream(text, language):
                if should_stop():
                    log.info("Playback interrupted")
                    break
                audio = np.frombuffer(chunk, dtype=np.int16)
                if len(audio) > 0:
                    stream.write(audio)
            log.info("Playback finished")
        finally:
            stream.stop()
            stream.close()

    async def get_speakers(self) -> list[dict]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/speakers")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/speakers_list", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
