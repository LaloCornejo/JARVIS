"""
Discord Voice Integration for JARVIS

This module handles voice channel interactions for Discord:
- TTS playback through voice channels
- Opus audio encoding
- UDP voice connections

Usage:
    from core.discord_voice import DiscordTTSPlayer
    player = DiscordTTSPlayer(tts_service)
    await player.speak(guild_id, "Hello!")
"""

from __future__ import annotations

__all__ = [
    "DiscordTTSPlayer",
    "OpusEncoder",
    "UDPVoiceConnection",
    "VoiceWebSocket",
]

# Import these only when available
try:
    from .opus_encoder import OpusEncoder
    from .udp_connection import UDPVoiceConnection
    from .tts_player import DiscordTTSPlayer
    from .voice_websocket import VoiceWebSocket

    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create dummy classes if voice dependencies not installed
    class OpusEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Voice dependencies not installed: {IMPORT_ERROR}")

    class UDPVoiceConnection:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Voice dependencies not installed: {IMPORT_ERROR}")

    class DiscordTTSPlayer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Voice dependencies not installed: {IMPORT_ERROR}")
