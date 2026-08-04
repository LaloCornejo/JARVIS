"""
Text-to-speech module with pluggable backends.

Provides:
- ``XTTSEngine`` — HTTP-based XTTS backend (original)
- ``TextToSpeech`` — playback facade (sounddevice) wrapping any backend
- ``create_tts_backend`` — factory that picks the engine from config
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import numpy as np

from .tts_base import TTSBackend

log = logging.getLogger("jarvis.tts")

# ---------------------------------------------------------------------------
# XTTS backend (HTTP API)
# ---------------------------------------------------------------------------

XTTS_SPEAKERS_CACHE: list[dict[str, Any]] | None = None


class XTTSEngine(TTSBackend):
    """TTS backend for XTTS via HTTP API (e.g. xtts-api-server)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8020",
        speaker: str = "duckie",
        language: str = "en",
        sample_rate: int = 24000,
    ):
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self._language = language
        self.sample_rate = sample_rate
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def speak_stream(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[bytes]:
        log.info("XTTS streaming: %s", text[:50] + ("..." if len(text) > 50 else ""))
        client = await self._get_client()
        params = {
            "text": text,
            "speaker_wav": self.speaker,
            "language": language or self._language,
        }
        async with client.stream(
            "GET",
            f"{self.base_url}/tts_stream",
            params=params,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                yield chunk

    async def speak(self, text: str, language: str | None = None) -> bytes:
        chunks: list[bytes] = []
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
                "language": language or self._language,
            },
        )
        response.raise_for_status()
        return response.content

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/speakers_list", timeout=1.0)
            return response.status_code == 200
        except Exception as e:
            log.debug("XTTS health check skipped: %s", e)
            return False

    async def get_speakers(self) -> list[dict[str, Any]]:
        global XTTS_SPEAKERS_CACHE
        if XTTS_SPEAKERS_CACHE is not None:
            return XTTS_SPEAKERS_CACHE
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/speakers")
            response.raise_for_status()
            speakers = response.json()
            XTTS_SPEAKERS_CACHE = speakers
            return speakers
        except Exception:
            return []

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except RuntimeError:
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Playback facade
# ---------------------------------------------------------------------------

class TextToSpeech:
    """Unified TTS interface wrapping a ``TTSBackend``.

    Handles audio playback via ``sounddevice``. Can be instantiated
    directly with XTTS params (backward-compatible) or with a custom
    backend.

    Parameters
    ----------
    backend : TTSBackend, optional
        Backend to use. If omitted an ``XTTSEngine`` is created from the
        other keyword arguments.
    base_url, speaker, language, sample_rate :
        XTTS-specific params (used only when ``backend`` is ``None``).
    """

    def __init__(
        self,
        backend: TTSBackend | None = None,
        base_url: str = "http://localhost:8020",
        speaker: str = "duckie",
        language: str = "en",
        sample_rate: int = 24000,
    ):
        if backend is not None:
            self.backend = backend
        else:
            self.backend = XTTSEngine(
                base_url=base_url,
                speaker=speaker,
                language=language,
                sample_rate=sample_rate,
            )
        self._stop_playback: threading.Event | None = None

    # --- generation (delegated to backend) ---

    async def speak_stream(self, text: str, language: str | None = None) -> AsyncIterator[bytes]:
        async for chunk in self.backend.speak_stream(text, language):
            yield chunk

    async def speak(self, text: str, language: str | None = None) -> bytes:
        return await self.backend.speak(text, language)

    async def speak_to_audio(self, text: str, language: str | None = None) -> bytes:
        return await self.backend.speak_to_audio(text, language)

    async def health_check(self) -> bool:
        return await self.backend.health_check()

    async def get_speakers(self) -> list[dict[str, Any]]:
        return await self.backend.get_speakers()

    async def close(self) -> None:
        self.interrupt_playback()
        await self.backend.close()

    def interrupt_playback(self) -> None:
        """Signal any ongoing playback thread to stop immediately."""
        if self._stop_playback:
            self._stop_playback.set()


    async def play_stream(
        self,
        text: str,
        language: str | None = None,
        speed: float = 1.5,
    ) -> None:
        """Stream TTS audio and play in a background thread.

        Any previous playback is interrupted first. Audio is resampled
        for *speed* (1.0 = normal) before writing to the sounddevice
        output.
        """
        self.interrupt_playback()
        stop = threading.Event()
        self._stop_playback = stop

        chunks: list[bytes] = []
        async for chunk in self.speak_stream(text, language):
            if stop.is_set():
                return
            chunks.append(chunk)

        if not chunks or stop.is_set():
            return

        sr = self.backend.sample_rate

        def _worker() -> None:
            import sounddevice as sd

            stream = sd.OutputStream(samplerate=sr, channels=1, dtype=np.int16)
            stream.start()
            try:
                for chunk in chunks:
                    if stop.is_set():
                        break
                    audio = np.frombuffer(chunk, dtype=np.int16)
                    if len(audio) > 0:
                        if speed != 1.0:
                            audio = self._resample(audio, speed)
                        stream.write(audio)
            finally:
                stream.stop()
                stream.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


    async def play_stream_interruptible(
        self,
        text: str,
        should_stop: Callable[[], bool],
        language: str | None = None,
        speed: float = 1.5,
    ) -> None:
        import sounddevice as sd

        log.info("Starting interruptible playback")

        stream = sd.OutputStream(
            samplerate=self.backend.sample_rate,
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
                    if speed != 1.0:
                        audio = self._resample(audio, speed)
                    stream.write(audio)
            log.info("Playback finished")
        finally:
            stream.stop()
            stream.close()

    # --- helpers ---

    @staticmethod
    def _resample(audio: np.ndarray, speed: float) -> np.ndarray:
        """Linear interpolation resampling for playback speed change.

        Values >1 speed up (fewer samples), <1 slow down (more samples).
        Pitch shifts proportionally (chipmunk effect with speed).
        """
        if speed <= 0 or speed == 1.0:
            return audio
        n = len(audio)
        target_len = int(n / speed)
        indices = np.linspace(0, n - 1, target_len)
        x_floor = np.floor(indices).astype(np.int32)
        x_ceil = np.minimum(x_floor + 1, n - 1)
        frac = (indices - x_floor).astype(audio.dtype)
        resampled = audio[x_floor] * (1 - frac) + audio[x_ceil] * frac
        return resampled.astype(audio.dtype)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tts_backend(config: Any) -> TTSBackend:
    """Create a TTS backend from a Config instance.

    Reads ``tts.engine`` from config and returns the matching backend.

    Parameters
    ----------
    config : Config
        Application config object with ``tts_*`` properties.

    Returns
    -------
    TTSBackend
        The selected backend (``XTTSEngine``, ``SupertonicEngine``, or ``MossttsNanoEngine``).
    """
    engine = config.tts_engine

    if engine == "supertonic":
        from .supertonic_backend import SupertonicEngine

        return SupertonicEngine(
            voice=config.supertonic_voice,
            custom_voice_path=config.supertonic_custom_voice_path,
            speed=config.supertonic_speed,
            steps=config.supertonic_steps,
            language=config.tts_language,
        )

    if engine == "mosstts_nano":
        from .mosstts_nano_backend import MossttsNanoEngine

        return MossttsNanoEngine(
            voice=config.mosstts_nano_voice,
            prompt_audio_path=config.mosstts_nano_prompt_audio_path,
            audio_temperature=config.mosstts_nano_audio_temperature,
            audio_top_p=config.mosstts_nano_audio_top_p,
            audio_top_k=config.mosstts_nano_audio_top_k,
            audio_repetition_penalty=config.mosstts_nano_audio_repetition_penalty,
            max_new_frames=config.mosstts_nano_max_new_frames,
            voice_clone_max_text_tokens=config.mosstts_nano_voice_clone_max_text_tokens,
        )

    return XTTSEngine(
        base_url=config.tts_base_url,
        speaker=config.tts_speaker,
        language=config.tts_language,
        sample_rate=config.tts_sample_rate,
    )
