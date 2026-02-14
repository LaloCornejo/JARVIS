from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Callable

import numpy as np

from .opus_encoder import OpusDecoder, OpusEncoder, resample_audio
from .udp_connection import UDPVoiceConnection

log = logging.getLogger("jarvis.discord_voice.tts")


class DiscordTTSPlayer:
    def __init__(self, tts_service):
        self.tts = tts_service
        self.opus_encoder = OpusEncoder()
        self.opus_decoder = OpusDecoder()
        self.voice_connections: Dict[str, UDPVoiceConnection] = {}
        self._speaking: Dict[str, bool] = {}
        self._tts_sample_rate = 24000
        self._discord_sample_rate = 48000
        self._on_transcription: Optional[Callable] = None
        self._audio_buffers: Dict[str, np.ndarray] = {}

        if not self.opus_encoder.is_available():
            log.error("Opus encoder not available. Voice playback will not work.")
            log.error(
                "Windows: Download opus.dll from https://github.com/discord/opus-builds/releases"
            )
            log.error("Linux: sudo apt-get install libopus0")
            log.error("Mac: brew install opus")

    def set_transcription_callback(self, callback: Callable) -> None:
        self._on_transcription = callback

    async def connect_to_voice(
        self,
        guild_id: str,
        endpoint: str,
        token: str,
        session_id: str,
        user_id: str,
        secret_key: bytes,
        ssrc: int = 0,
    ) -> bool:
        """Connect to Discord voice server for a guild.

        Args:
            guild_id: Discord guild ID
            endpoint: Voice server endpoint
            token: Voice token from Gateway
            session_id: Session ID from Gateway
            user_id: Bot user ID
            secret_key: Encryption key from Gateway
            ssrc: Synchronization source ID from voice server

        Returns:
            True if connected
        """
        if guild_id in self.voice_connections:
            await self.voice_connections[guild_id].disconnect()
            del self.voice_connections[guild_id]

        connection = UDPVoiceConnection()
        success = await connection.connect(endpoint, token, session_id, user_id, secret_key)

        if success:
            connection.set_ssrc(ssrc)
            self.voice_connections[guild_id] = connection
            self._speaking[guild_id] = False
            log.info("Connected to voice for guild %s", guild_id)
            return True
        else:
            log.error("Failed to connect to voice for guild %s", guild_id)
            return False

    async def disconnect_from_voice(self, guild_id: str) -> None:
        """Disconnect from voice channel for a guild."""
        if guild_id in self.voice_connections:
            await self.voice_connections[guild_id].disconnect()
            del self.voice_connections[guild_id]
            log.info("Disconnected from voice for guild %s", guild_id)

        if guild_id in self._speaking:
            del self._speaking[guild_id]

    async def speak(self, guild_id: str, text: str) -> bool:
        """Play TTS audio to voice channel.

        Args:
            guild_id: Discord guild ID
            text: Text to speak

        Returns:
            True if spoken successfully
        """
        if not self.opus_encoder.is_available():
            log.error("Cannot speak - Opus encoder not available. Install Opus library.")
            return False

        if guild_id not in self.voice_connections:
            log.warning("Not connected to voice for guild %s", guild_id)
            return False

        connection = self.voice_connections[guild_id]
        if not connection.is_connected():
            log.warning("Voice connection not active for guild %s", guild_id)
            return False

        try:
            self._speaking[guild_id] = True

            log.info("Generating TTS for guild %s: %s...", guild_id, text[:50])

            # Get TTS audio from service (24kHz, PCM)
            audio_bytes = await self.tts.speak_to_audio(text)

            if not audio_bytes:
                log.warning("TTS generated empty audio for guild %s", guild_id)
                return False

            # Convert bytes to numpy array (int16)
            import numpy as np

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Resample from 24kHz to 48kHz
            if self._tts_sample_rate != self._discord_sample_rate:
                log.debug(
                    "Resampling audio %dHz -> %dHz",
                    self._tts_sample_rate,
                    self._discord_sample_rate,
                )
                audio_array = resample_audio(
                    audio_array.astype(np.float32), self._tts_sample_rate, self._discord_sample_rate
                )
                audio_array = (audio_array * 32767).astype(np.int16)

            # Encode to Opus
            log.debug("Encoding to Opus...")
            opus_frames = self.opus_encoder.encode_pcm(audio_array)

            # Send to voice channel
            log.info("Sending %d Opus frames to voice channel", len(opus_frames))
            await connection.send_audio(opus_frames)

            log.info("TTS playback complete for guild %s", guild_id)
            return True

        except Exception as e:
            log.error("TTS playback failed for guild %s: %s", guild_id, e)
            return False
        finally:
            self._speaking[guild_id] = False

    async def stop(self, guild_id: str) -> None:
        """Stop speaking (not fully implemented - would need packet queue)."""
        if guild_id in self._speaking:
            self._speaking[guild_id] = False

    def is_speaking(self, guild_id: str) -> bool:
        """Check if currently speaking in a guild."""
        return self._speaking.get(guild_id, False)

    def is_connected(self, guild_id: str) -> bool:
        """Check if connected to voice in a guild."""
        if guild_id not in self.voice_connections:
            return False
        return self.voice_connections[guild_id].is_connected()

    async def close(self) -> None:
        for guild_id in list(self.voice_connections.keys()):
            await self.disconnect_from_voice(guild_id)

    async def start_voice_receive(self, guild_id: str) -> bool:
        if guild_id not in self.voice_connections:
            log.warning(f"Cannot start voice receive: not connected to guild {guild_id}")
            return False

        connection = self.voice_connections[guild_id]
        if not connection.is_connected():
            log.warning(f"Cannot start voice receive: not connected")
            return False

        self._audio_buffers[guild_id] = np.array([], dtype=np.int16)

        async def on_audio(opus_data: bytes, user_id: int, timestamp: int) -> None:
            if not self.opus_decoder.is_available():
                return

            pcm_data = self.opus_decoder.decode(opus_data)
            if pcm_data is not None and len(pcm_data) > 0:
                if guild_id in self._audio_buffers:
                    self._audio_buffers[guild_id] = np.concatenate(
                        [self._audio_buffers[guild_id], pcm_data]
                    )

        async def on_speaking(user_id: int, speaking: bool) -> None:
            if not speaking and user_id != 0:
                await self._process_audio(guild_id, user_id)

        await connection.start_receiving(on_audio=on_audio, on_speaking=on_speaking)
        log.info(f"Started voice receive for guild {guild_id}")
        return True

    async def _process_audio(self, guild_id: str, user_id: int) -> None:
        if guild_id not in self._audio_buffers:
            return

        audio = self._audio_buffers[guild_id]
        if len(audio) < 1600:
            self._audio_buffers[guild_id] = np.array([], dtype=np.int16)
            return

        audio_float = audio.astype(np.float32) / 32767.0

        audio_16k = resample_audio(audio_float, 48000, 16000)
        audio_16k = (audio_16k * 32767).astype(np.int16)

        self._audio_buffers[guild_id] = np.array([], dtype=np.int16)

        if self._on_transcription:
            try:
                await self._on_transcription(audio_16k, user_id, guild_id)
            except Exception as e:
                log.error(f"Error in transcription callback: {e}")

    async def stop_voice_receive(self, guild_id: str) -> None:
        if guild_id in self.voice_connections:
            connection = self.voice_connections[guild_id]
            connection.stop_receiving()

        if guild_id in self._audio_buffers:
            del self._audio_buffers[guild_id]

        log.info(f"Stopped voice receive for guild {guild_id}")
