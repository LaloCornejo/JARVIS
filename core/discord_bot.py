"""Discord Bot Handler for JARVIS - Active two-way communication.
This module provides a background service that:
- Connects to Discord Gateway for real-time events
- Receives and processes messages from Discord channels
- Routes messages to JARVIS for processing
- Sends responses back to Discord channels
- Maintains conversation context per channel/user
Usage:
    from core.discord_bot import discord_bot_handler
    await discord_bot_handler.start()
    # ... JARVIS running ...
    await discord_bot_handler.stop()
Setup:
    1. Create a Discord application at https://discord.com/developers/applications
    2. Create a Bot user and get the token
    3. Enable "Message Content Intent" in Bot settings
    4. Invite bot to your server with appropriate permissions
    5. Set DISCORD_BOT_TOKEN environment variable
"""

import asyncio
import json
import logging
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import httpx
import websockets

# Import voice receive functionality
try:
    from discord.ext import voice_recv

    HAS_VOICE_RECV = True
except ImportError:
    HAS_VOICE_RECV = False
    voice_recv = None
# Import STT functionality
from core.voice.stt import OptimizedSpeechToText

try:
    from core.discord_voice import VOICE_AVAILABLE, DiscordTTSPlayer, VoiceWebSocket
except ImportError as _discord_voice_import_error:
    _import_error_msg = str(_discord_voice_import_error)
    VOICE_AVAILABLE = False
    DiscordTTSPlayer = None
    VoiceWebSocket = None
# Reduce websocket logging verbosity
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
if TYPE_CHECKING:
    from jarvis.server import JarvisServer

log = logging.getLogger(__name__)

if not VOICE_AVAILABLE:
    log.warning(f"[DISCORD] Voice module not available: voice dependencies may not be installed")


@dataclass
class DiscordSession:
    """Maintains conversation state for a Discord channel or DM."""

    channel_id: str
    channel_name: str | None = None
    guild_id: str | None = None
    guild_name: str | None = None
    user_id: str | None = None
    username: str | None = None
    messages: list[dict] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    max_messages: int = 25
    voice_client: Any | None = None
    voice_sink: Any | None = None

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation history."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
        )
        self.last_activity = datetime.now()
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_context(self) -> list[dict]:
        """Get conversation context for LLM."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
            "user_id": self.user_id,
            "username": self.username,
            "message_count": len(self.messages),
            "last_activity": self.last_activity.isoformat(),
        }


class DiscordBotHandler:
    """Handles active two-way Discord communication with JARVIS."""

    GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
    COMPRESSED_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json&compress=zlib-stream"
    API_URL = "https://discord.com/api/v10"

    def __init__(
        self,
        jarvis_server: "JarvisServer | None" = None,
        poll_interval: float = 1.0,
        allowed_channel_ids: list[str] | None = None,
        allowed_guild_ids: list[str] | None = None,
        dm_only: bool = False,
        mention_only: bool = False,
    ):
        self.token = ""
        self.jarvis: "JarvisServer | None" = jarvis_server
        self.poll_interval = poll_interval
        self.dm_only = dm_only
        self.mention_only = mention_only
        if allowed_channel_ids is not None:
            self.allowed_channel_ids = set(allowed_channel_ids)
        else:
            env_ids = os.environ.get("DISCORD_ALLOWED_CHANNEL_IDS", "")
            self.allowed_channel_ids = (
                set(id.strip() for id in env_ids.split(",") if id.strip()) or None
            )
        if allowed_guild_ids is not None:
            self.allowed_guild_ids = set(allowed_guild_ids)
        else:
            env_ids = os.environ.get("DISCORD_ALLOWED_GUILD_IDS", "")
            self.allowed_guild_ids = (
                set(id.strip() for id in env_ids.split(",") if id.strip()) or None
            )
        self._running = False
        self._ws_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._session_id: str | None = None
        self._sequence_number: int | None = None
        self._resume_gateway_url: str | None = None
        self._heartbeat_interval: float | None = None
        self._last_heartbeat_ack: bool = True
        self._zlib_suffix = b"\x00\x00\xff\xff"
        self._zlib_inflater: zlib._Decompress | None = None
        self._bot_info: dict | None = None
        self._bot_user_id: str | None = None
        self._bot_username: str | None = None
        self._sessions: dict[str, DiscordSession] = {}
        self._voice_clients: dict[str, Any] = {}
        self._voice_player: DiscordTTSPlayer | None = None
        self._voice_states: dict[str, dict] = {}
        self._voice_websockets: dict[str, VoiceWebSocket] = {}
        self._voice_session_ids: dict[str, str] = {}
        self._pending_voice_server_updates: dict[str, dict] = {}
        self._stt = None
        self._http_client: httpx.AsyncClient | None = None
        self._command_handlers: dict[str, Callable] = {
            "!jarvis": self._handle_help,
            "!help": self._handle_help,
            "!status": self._handle_status,
            "!clear": self._handle_clear,
            "!join": self._handle_join_voice,
            "!leave": self._handle_leave_voice,
            "!listen": self._handle_listen_voice,
            "!stoplisten": self._handle_stop_listen_voice,
            "!test-voice": self._handle_test_voice,
        }

    async def start(self, jarvis_server: "JarvisServer | None" = None) -> bool:
        """Start the Discord bot handler."""
        if self._running:
            log.warning("[DISCORD] Handler already running")
            return True
        if not self.token:
            self.token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not self.token:
            log.error("[DISCORD] No bot token configured. Set DISCORD_BOT_TOKEN")
            return False
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bot {self.token}"},
            timeout=30.0,
        )
        try:
            response = await self._http_client.get(f"{self.API_URL}/users/@me")
            if response.status_code == 200:
                self._bot_info = response.json()
                self._bot_user_id = self._bot_info.get("id")
                self._bot_username = self._bot_info.get("username")
                log.info(f"[DISCORD] Bot connected: {self._bot_username}")
            else:
                log.error(f"[DISCORD] Failed to authenticate: {response.status_code}")
                return False
        except Exception as e:
            log.error(f"[DISCORD] Error connecting to Discord: {e}")
            return False
        if self._stt is None:
            try:
                self._stt = OptimizedSpeechToText(
                    model_size="base.en",
                    device="cpu",
                    compute_type="int8",
                    preload=True,
                )
                log.info("[DISCORD] STT initialized successfully")
            except Exception as e:
                log.error(f"[DISCORD] Failed to initialize STT: {e}")
                self._stt = None
        self.jarvis = jarvis_server
        self._running = True
        self._ws_task = asyncio.create_task(self._websocket_loop())
        log.info(
            f"[DISCORD] Voice initialization - VOICE_AVAILABLE: {VOICE_AVAILABLE}, DiscordTTSPlayer: {DiscordTTSPlayer is not None}, jarvis: {self.jarvis is not None}"
        )
        if VOICE_AVAILABLE and DiscordTTSPlayer and self.jarvis and hasattr(self.jarvis, "tts"):
            try:
                self._voice_player = DiscordTTSPlayer(self.jarvis.tts)
                if self._stt:
                    self._voice_player.set_transcription_callback(self._handle_voice_transcription)
                log.info("[DISCORD] Voice player initialized successfully")
            except Exception as e:
                log.error(f"[DISCORD] Failed to initialize voice player: {e}", exc_info=True)
                self._voice_player = None
        else:
            log.warning(
                f"[DISCORD] Voice player not initialized - VOICE_AVAILABLE={VOICE_AVAILABLE}, DiscordTTSPlayer={DiscordTTSPlayer is not None}, jarvis={self.jarvis is not None}, has_tts={hasattr(self.jarvis, 'tts') if self.jarvis else False}"
            )
        log.info("[DISCORD] Handler started")
        return True

    async def stop(self) -> None:
        """Stop the Discord bot handler."""
        if not self._running:
            return
        log.info("[DISCORD] Stopping handler...")
        self._running = False
        await self._cleanup_voice_resources()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        log.info("[DISCORD] Handler stopped")

    async def _cleanup_voice_resources(self) -> None:
        if self._voice_player:
            try:
                for guild_id in list(self._voice_player._connections.keys()):
                    await self._voice_player.disconnect_from_voice(guild_id)
            except Exception as e:
                log.warning(f"[DISCORD] Error disconnecting voice player: {e}")
            self._voice_player = None

        for guild_id, ws in list(self._voice_websockets.items()):
            try:
                await ws.disconnect()
            except Exception as e:
                log.warning(f"[DISCORD] Error closing voice WebSocket for guild {guild_id}: {e}")
        self._voice_websockets.clear()
        self._voice_states.clear()
        self._voice_session_ids.clear()
        self._pending_voice_server_updates.clear()

    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "running": self._running,
            "bot_username": self._bot_username,
            "bot_user_id": self._bot_user_id,
            "active_sessions": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
            "allowed_channel_ids": list(self.allowed_channel_ids)
            if self.allowed_channel_ids
            else None,
            "allowed_guild_ids": list(self.allowed_guild_ids) if self.allowed_guild_ids else None,
        }

    async def _websocket_loop(self) -> None:
        """Main WebSocket connection loop with reconnection support."""
        reconnect_delay = 1.0
        max_reconnect_delay = 60.0
        while self._running:
            try:
                gateway_url = self._resume_gateway_url or self.GATEWAY_URL
                log.info(f"[DISCORD] Connecting to Gateway: {gateway_url}")
                async with websockets.connect(gateway_url) as websocket:
                    self._websocket = websocket
                    log.info("[DISCORD] WebSocket connected")
                    reconnect_delay = 1.0
                    self._zlib_inflater = zlib.decompressobj()
                    async for message in websocket:
                        if not self._running:
                            break
                        try:
                            await self._handle_gateway_message(message)
                        except Exception as e:
                            log.error(f"[DISCORD] Error handling message: {e}")
            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"[DISCORD] WebSocket closed: {e}")
            except Exception as e:
                log.error(f"[DISCORD] WebSocket error: {e}")
            if self._running:
                log.info(f"[DISCORD] Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _handle_gateway_message(self, message: bytes | str) -> None:
        """Handle a message from the Discord Gateway."""
        if isinstance(message, bytes):
            try:
                if message.endswith(self._zlib_suffix):
                    message = self._zlib_inflater.decompress(message)
                    message = message.decode("utf-8")
                else:
                    message = self._zlib_inflater.decompress(message)
                    message = message.decode("utf-8")
            except Exception as e:
                log.error(f"[DISCORD] Decompression error: {e}")
                return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            log.error("[DISCORD] Failed to parse JSON message")
            return
        sequence = payload.get("s")
        if sequence is not None:
            self._sequence_number = sequence
        op_code = payload.get("op")
        event_type = payload.get("t")
        event_data = payload.get("d", {})
        if op_code == 10:
            await self._handle_hello(event_data)
        elif op_code == 11:
            self._last_heartbeat_ack = True
        elif op_code == 0:
            await self._handle_dispatch(event_type, event_data)
        elif op_code == 7:
            log.info("[DISCORD] Received reconnect request")
            self._resume_gateway_url = None
        elif op_code == 9:
            log.warning("[DISCORD] Invalid session, re-identifying")
            self._session_id = None
            self._sequence_number = None
            await asyncio.sleep(5)
            await self._send_identify()

    async def _handle_hello(self, data: dict) -> None:
        """Handle Gateway Hello event."""
        self._heartbeat_interval = data.get("heartbeat_interval", 45000) / 1000
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if self._session_id and self._sequence_number:
            await self._send_resume()
        else:
            await self._send_identify()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._running and self._websocket:
            try:
                await asyncio.sleep(self._heartbeat_interval or 45)
                if not self._last_heartbeat_ack:
                    log.warning("[DISCORD] Heartbeat not acknowledged, reconnecting")
                    await self._websocket.close()
                    return
                self._last_heartbeat_ack = False
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[DISCORD] Heartbeat error: {e}")

    async def _send_heartbeat(self) -> None:
        """Send heartbeat payload."""
        if self._websocket:
            payload = {"op": 1, "d": self._sequence_number}
            await self._websocket.send(json.dumps(payload))

    async def _send_identify(self) -> None:
        """Send identify payload to authenticate."""
        if not self._websocket:
            return
        payload = {
            "op": 2,
            "d": {
                "token": self.token,
                "intents": 33409,
                "properties": {
                    "os": "linux",
                    "browser": "JARVIS",
                    "device": "JARVIS",
                },
            },
        }
        await self._websocket.send(json.dumps(payload))
        log.info("[DISCORD] Sent identify payload")

    async def _send_resume(self) -> None:
        """Send resume payload to continue previous session."""
        if not self._websocket:
            return
        payload = {
            "op": 6,
            "d": {
                "token": self.token,
                "session_id": self._session_id,
                "seq": self._sequence_number,
            },
        }
        await self._websocket.send(json.dumps(payload))
        log.info("[DISCORD] Sent resume payload")

    async def _handle_dispatch(self, event_type: str | None, data: dict) -> None:
        """Handle Discord dispatch events."""
        if not event_type:
            return
        if event_type == "READY":
            self._session_id = data.get("session_id")
            self._resume_gateway_url = data.get("resume_gateway_url")
            user = data.get("user", {})
            log.info(f"[DISCORD] Ready as {user.get('username')}#{user.get('discriminator', '0')}")
        elif event_type == "MESSAGE_CREATE":
            await self._handle_message_create(data)
        elif event_type == "RESUMED":
            log.info("[DISCORD] Session resumed")
        elif event_type == "VOICE_STATE_UPDATE":
            await self._handle_voice_state_update(data)
        elif event_type == "VOICE_SERVER_UPDATE":
            await self._handle_voice_server_update(data)

    async def _handle_voice_state_update(self, data: dict) -> None:
        """Handle voice state update events."""
        user_id = data.get("user_id")
        guild_id = data.get("guild_id")
        channel_id = data.get("channel_id")
        session_id = data.get("session_id")

        log.debug(
            f"[DISCORD] Voice state update - User: {user_id}, Guild: {guild_id}, Channel: {channel_id}, Session: {session_id}"
        )
        log.debug(f"[DISCORD] Bot user ID: {self._bot_user_id}")

        if user_id == self._bot_user_id:
            log.info(
                f"[DISCORD] Bot voice state updated - Guild: {guild_id}, Channel: {channel_id}"
            )
            if guild_id:
                if channel_id:
                    self._voice_states[guild_id] = data
                    log.debug(f"[DISCORD] Cached voice state for guild {guild_id}")
                    if session_id:
                        self._voice_session_ids[guild_id] = session_id
                        log.debug(f"[DISCORD] Stored session_id for guild {guild_id}")
                        if guild_id in self._pending_voice_server_updates:
                            pending = self._pending_voice_server_updates.pop(guild_id)
                            log.info(
                                f"[DISCORD] Processing pending voice server update for guild {guild_id}"
                            )
                            await self._do_voice_server_connect(guild_id, pending)
                else:
                    if guild_id in self._voice_states:
                        del self._voice_states[guild_id]
                        log.debug(f"[DISCORD] Removed cached voice state for guild {guild_id}")
                    if self._voice_player and self._voice_player.is_connected(guild_id):
                        await self._voice_player.disconnect_from_voice(guild_id)
                    if guild_id in self._voice_session_ids:
                        del self._voice_session_ids[guild_id]

                if channel_id:
                    log.info(f"[DISCORD] Bot joined voice channel {channel_id} in guild {guild_id}")
                else:
                    if guild_id in self._voice_clients:
                        del self._voice_clients[guild_id]
                    log.info(f"[DISCORD] Bot left voice channel in guild {guild_id}")

    async def _handle_voice_server_update(self, data: dict) -> None:
        """Handle voice server update events."""
        guild_id = data.get("guild_id")
        endpoint = data.get("endpoint")
        token = data.get("token")

        log.info(f"[DISCORD] Voice server update - Guild: {guild_id}, Endpoint: {endpoint}")
        log.debug(f"[DISCORD] Voice player available: {self._voice_player is not None}")

        if not self._voice_player or not endpoint or not guild_id:
            log.debug(
                f"[DISCORD] Skipping voice connection - player: {self._voice_player is not None}, endpoint: {endpoint}, guild: {guild_id}"
            )
            return

        if not self._bot_user_id:
            log.error("[DISCORD] Bot user ID not available")
            return

        session_id = self._voice_session_ids.get(guild_id)

        if not session_id:
            log.info(
                f"[DISCORD] VOICE_STATE_UPDATE not yet received for guild {guild_id}, caching server update"
            )
            data["_timestamp"] = datetime.now().timestamp()
            self._pending_voice_server_updates[guild_id] = data
            self._cleanup_stale_pending_updates()
            return

        await self._do_voice_server_connect(guild_id, data)

    async def _do_voice_server_connect(self, guild_id: str, data: dict) -> None:
        endpoint = data.get("endpoint")
        token = data.get("token")

        if not endpoint or not token:
            log.error(f"[DISCORD] Missing endpoint or token for voice connection")
            return

        if not self._bot_user_id:
            log.error("[DISCORD] Bot user ID not available")
            return

        session_id = self._voice_session_ids.get(guild_id)
        if not session_id:
            log.error(f"[DISCORD] No session_id for guild {guild_id}")
            return

        log.info(f"[DISCORD] Connecting to voice server for guild {guild_id}")

        voice_ws = VoiceWebSocket()
        connection_ready = asyncio.Future()

        async def on_session_description(secret_key: bytes) -> None:
            log.info(
                f"[DISCORD] Voice session ready for guild {guild_id}, connecting TTS player..."
            )

            if self._voice_player:
                success = await self._voice_player.connect_to_voice(
                    guild_id=guild_id,
                    endpoint=endpoint,
                    token=token,
                    session_id=session_id,
                    user_id=self._bot_user_id,
                    secret_key=secret_key,
                    ssrc=voice_ws.ssrc,
                )
                if success:
                    log.info(f"[DISCORD] TTS player connected for guild {guild_id}")
                    if not connection_ready.done():
                        connection_ready.set_result(True)
                else:
                    log.error(f"[DISCORD] Failed to connect TTS player for guild {guild_id}")
                    if not connection_ready.done():
                        connection_ready.set_exception(Exception("TTS player connection failed"))
            else:
                log.error(f"[DISCORD] No voice player available for guild {guild_id}")
                if not connection_ready.done():
                    connection_ready.set_exception(Exception("No voice player"))

        voice_ws.on_session_description = on_session_description

        try:
            ws_connected = await voice_ws.connect(
                endpoint=endpoint,
                token=token,
                server_id=guild_id,
                user_id=self._bot_user_id,
                session_id=session_id,
            )

            if not ws_connected:
                log.error(f"[DISCORD] Failed to connect voice WebSocket for guild {guild_id}")
                return

            self._voice_websockets[guild_id] = voice_ws

            try:
                await asyncio.wait_for(connection_ready, timeout=10.0)
                log.info(f"[DISCORD] Voice connection established for guild {guild_id}")
            except asyncio.TimeoutError:
                log.error(f"[DISCORD] Voice connection timeout for guild {guild_id}")
                await voice_ws.disconnect()
                if guild_id in self._voice_websockets:
                    del self._voice_websockets[guild_id]

        except Exception as e:
            log.error(
                f"[DISCORD] Error establishing voice connection for guild {guild_id}: {e}",
                exc_info=True,
            )

    def _cleanup_stale_pending_updates(self) -> None:
        if not self._pending_voice_server_updates:
            return
        now = datetime.now().timestamp()
        stale = [
            gid
            for gid, data in self._pending_voice_server_updates.items()
            if data.get("_timestamp", 0) and now - data.get("_timestamp", 0) > 60
        ]
        for gid in stale:
            del self._pending_voice_server_updates[gid]
        if stale:
            log.debug(f"[DISCORD] Cleaned up {len(stale)} stale pending voice updates")

    async def _handle_message_create(self, data: dict) -> None:
        """Handle incoming message."""
        author = data.get("author", {})
        if author.get("id") == self._bot_user_id:
            return
        if author.get("bot"):
            return
        content = data.get("content", "")
        channel_id = data.get("channel_id")
        guild_id = data.get("guild_id")
        message_id = data.get("id")
        channel_data = data.get("channel", {})
        is_dm = guild_id is None
        log.info(
            f"[DISCORD] Received message - Channel: {channel_id}, DM: {is_dm}, "
            f"Author: {author.get('username')}, Content: {content[:100]}..."
        )
        if self.dm_only and not is_dm:
            log.debug("[DISCORD] Message filtered out - dm_only is True but this is not a DM")
            return
        if self.allowed_channel_ids and channel_id not in self.allowed_channel_ids:
            log.debug(f"[DISCORD] Message filtered out - Channel {channel_id} not in allowed list")
            return
        if self.allowed_guild_ids and guild_id and guild_id not in self.allowed_guild_ids:
            log.debug(f"[DISCORD] Message filtered out - Guild {guild_id} not in allowed list")
            return
        mentions = data.get("mentions", [])
        bot_mentioned = any(m.get("id") == self._bot_user_id for m in mentions)
        if not is_dm:
            if self.mention_only:
                if not bot_mentioned and not any(
                    content.startswith(cmd) for cmd in self._command_handlers
                ):
                    log.debug(
                        "[DISCORD] Message filtered out - mention_only is True but bot not mentioned"
                    )
                    return
        session = self._get_or_create_session(
            channel_id=channel_id,
            guild_id=guild_id,
            user_id=author.get("id"),
            username=author.get("username"),
        )
        log.info(
            f"[DISCORD] Processing message from {author.get('username')} in {channel_id}: {content[:100]}..."
        )
        for cmd, handler in self._command_handlers.items():
            if content.startswith(cmd):
                log.info(f"[DISCORD] Handling command: {cmd}")
                await handler(session, data, content)
                return
        log.info("[DISCORD] Sending message to JARVIS for processing")
        await self._process_message_through_jarvis(session, content, channel_id, message_id)

    async def _process_message_through_jarvis(
        self,
        session: DiscordSession,
        text: str,
        channel_id: str,
        message_id: str,
    ) -> None:
        """Send message to JARVIS and return response."""
        if not self.jarvis:
            log.error("[DISCORD] No JarvisServer configured")
            await self._send_message(channel_id, "JARVIS server not available")
            return
        session.add_message("user", text)
        await self._send_typing(channel_id)
        try:
            full_response = ""
            response_chunks = []

            async def capture_broadcast(message: dict) -> None:
                try:
                    nonlocal full_response
                    message_type = message.get("type")
                    log.debug(f"[DISCORD] Received broadcast message: {message_type}")
                    if message_type == "streaming_chunk":
                        content = message.get("content", "")
                        log.debug(f"[DISCORD] Received streaming chunk: {len(content)} chars")
                        response_chunks.append(content)
                    elif message_type == "message_complete":
                        full_response = message.get("full_response", "")
                        log.debug(
                            f"[DISCORD] Received message complete: {len(full_response)} chars"
                        )
                    else:
                        log.debug(f"[DISCORD] Received unknown message type: {message_type}")
                except Exception as e:
                    log.error(f"[DISCORD] Error in capture_broadcast: {e}", exc_info=True)

            log.info(f"[DISCORD] Sending message to JARVIS: {text[:100]}...")
            log.debug("[DISCORD] Starting process_message call")
            await self.jarvis.process_message(text, broadcast_func=capture_broadcast)
            log.debug("[DISCORD] Finished process_message call")
            log.debug(
                f"[DISCORD] Response chunks: {len(response_chunks)}, Full response: {len(full_response) if full_response else 0} chars"
            )
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)
                log.debug(
                    f"[DISCORD] Joined chunks to form full response: {len(full_response)} chars"
                )
            if full_response:
                log.info(f"[DISCORD] Got response from JARVIS: {len(full_response)} chars")
                session.add_message("assistant", full_response)
                tts_available = False
                if hasattr(self.jarvis, "tts") and self.jarvis.tts:
                    try:
                        tts_available = await self.jarvis.tts.health_check()
                        if tts_available:
                            log.info("[DISCORD] TTS service is available")
                        else:
                            log.debug("[DISCORD] TTS service is not available")
                    except Exception as e:
                        log.warning(f"[DISCORD] TTS health check failed: {e}")
                        tts_available = False
                if tts_available:
                    in_voice = await self._is_in_voice_channel(session.guild_id)
                    voice_player_connected = (
                        self._voice_player.is_connected(session.guild_id)
                        if self._voice_player
                        else False
                    )
                    log.debug(
                        f"[DISCORD] Voice check - in_voice: {in_voice}, voice_player: {self._voice_player is not None}, connected: {voice_player_connected}"
                    )
                    if in_voice and self._voice_player and voice_player_connected:
                        try:
                            await self._send_long_message(
                                channel_id, full_response, reply_to=message_id
                            )
                            await self._voice_player.speak(session.guild_id, full_response)
                            log.info("[DISCORD] Voice response sent successfully")
                        except Exception as e:
                            log.error(f"[DISCORD] Error speaking to voice: {e}")
                    else:
                        log.debug(f"[DISCORD] Not sending to voice - using audio attachment")
                        try:
                            audio_data = await self.jarvis.tts.speak_to_audio(full_response)
                            await self._send_audio_message(
                                channel_id,
                                audio_data,
                                filename="response.mp3",
                                text_content=full_response,
                                reply_to=message_id,
                            )
                            log.info("[DISCORD] Audio message sent successfully")
                        except Exception as e:
                            log.error(f"[DISCORD] Error generating/sending TTS: {e}")
                            await self._send_long_message(
                                channel_id, full_response, reply_to=message_id
                            )
                else:
                    await self._send_long_message(channel_id, full_response, reply_to=message_id)
            else:
                log.warning("[DISCORD] No response from JARVIS")
                await self._send_message(
                    channel_id, "I didn't get a response. Please try again.", reply_to=message_id
                )
        except Exception as e:
            log.error(f"[DISCORD] Error processing message: {e}", exc_info=True)
            await self._send_message(channel_id, f"Error: {str(e)[:200]}", reply_to=message_id)

    async def _send_message(
        self,
        channel_id: str,
        content: str,
        reply_to: str | None = None,
        embed: dict | None = None,
    ) -> dict | None:
        """Send a message to Discord."""
        if not self._http_client:
            return None
        payload: dict[str, Any] = {"content": content}
        if reply_to:
            payload["message_reference"] = {"message_id": reply_to}
        if embed:
            payload["embeds"] = [embed]
        try:
            response = await self._http_client.post(
                f"{self.API_URL}/channels/{channel_id}/messages",
                json=payload,
            )
            if response.status_code in (200, 201):
                return response.json()
            else:
                log.error(f"[DISCORD] Failed to send message: {response.status_code}")
                return None
        except Exception as e:
            log.error(f"[DISCORD] Error sending message: {e}")
            return None

    async def _send_audio_message(
        self,
        channel_id: str,
        audio_data: bytes,
        filename: str = "response.mp3",
        text_content: str | None = None,
        reply_to: str | None = None,
    ) -> dict | None:
        """Send an audio message to Discord with optional text content."""
        if not self._http_client:
            return None
        try:
            import io

            audio_file = io.BytesIO(audio_data)
            audio_file.seek(0)
            payload_data = {}
            if text_content:
                payload_data["content"] = text_content
            if reply_to:
                payload_data["message_reference"] = {"message_id": reply_to}
            files = {"file": (filename, audio_file, "audio/mpeg")}
            data = {"payload_json": json.dumps(payload_data)}
            response = await self._http_client.post(
                f"{self.API_URL}/channels/{channel_id}/messages",
                files=files,
                data=data,
            )
            if response.status_code in (200, 201):
                result = response.json()
                log.info(f"[DISCORD] Audio message sent successfully to channel {channel_id}")
                return result
            else:
                log.error(f"[DISCORD] Failed to send audio message: {response.status_code}")
                log.error(f"[DISCORD] Response: {response.text}")
                return None
        except Exception as e:
            log.error(f"[DISCORD] Error sending audio message: {e}")
            return None

    async def _send_long_message(
        self,
        channel_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> None:
        """Send a potentially long message, splitting if needed."""
        max_length = 1950
        if len(text) <= max_length:
            await self._send_message(channel_id, text, reply_to)
            return
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            split_point = text.rfind("\n", 0, max_length)
            if split_point == -1:
                split_point = text.rfind(" ", 0, max_length)
            if split_point == -1:
                split_point = max_length
            chunks.append(text[:split_point])
            text = text[split_point:].strip()
        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)}) " if len(chunks) > 1 else ""
            await self._send_message(channel_id, prefix + chunk, reply_to if i == 0 else None)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    async def _send_typing(self, channel_id: str) -> None:
        """Send typing indicator."""
        if not self._http_client:
            return
        try:
            await self._http_client.post(
                f"{self.API_URL}/channels/{channel_id}/typing",
            )
        except Exception:
            pass

    def _get_or_create_session(
        self,
        channel_id: str,
        guild_id: str | None = None,
        user_id: str | None = None,
        username: str | None = None,
    ) -> DiscordSession:
        """Get existing session or create new one."""
        if channel_id not in self._sessions:
            self._sessions[channel_id] = DiscordSession(
                channel_id=channel_id,
                guild_id=guild_id,
                user_id=user_id,
                username=username,
            )
            log.info(f"[DISCORD] New session created for channel: {channel_id}")
        return self._sessions[channel_id]

    async def _get_voice_state(self, guild_id: str) -> dict | None:
        """Get voice state for a guild."""
        if guild_id in self._voice_states:
            return self._voice_states[guild_id]
        if not self._http_client:
            log.error("[DISCORD] No HTTP client available for voice state retrieval")
            return None
        try:
            response = await self._http_client.get(
                f"{self.API_URL}/guilds/{guild_id}/voice-states/@me"
            )
            log.debug(f"[DISCORD] Voice state response status: {response.status_code}")
            if response.status_code == 200:
                voice_state = response.json()
                log.debug(f"[DISCORD] Voice state: {voice_state}")
                self._voice_states[guild_id] = voice_state
                return voice_state
            else:
                log.debug(f"[DISCORD] Failed to get voice state: {response.status_code}")
                log.debug(f"[DISCORD] Response text: {response.text}")
        except Exception as e:
            log.debug(f"[DISCORD] Exception getting voice state: {e}")
        return None

    async def _is_in_voice_channel(self, guild_id: str | None) -> bool:
        """Check if bot is currently in a voice channel."""
        if not guild_id:
            return False
        voice_state = await self._get_voice_state(guild_id)
        return voice_state is not None and voice_state.get("channel_id") is not None

    async def _update_voice_state(self, guild_id: str, channel_id: str | None) -> bool:
        """Update voice state for a guild through Gateway."""
        if not self._websocket:
            log.error("[DISCORD] No WebSocket connection available for voice state update")
            return False
        try:
            payload = {
                "op": 4,
                "d": {
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "self_mute": False,
                    "self_deaf": False,
                },
            }
            log.debug(f"[DISCORD] Sending voice state update: {payload}")
            await self._websocket.send(json.dumps(payload))
            log.debug("[DISCORD] Voice state update sent successfully")
            return True
        except Exception as e:
            log.error(f"[DISCORD] Failed to send voice state update: {e}")
            return False

    async def _handle_help(self, session: DiscordSession, message_data: dict, text: str) -> None:
        help_text = """**JARVIS Discord Bot Help**
Just send me a message and I'll respond! I can:
Search the web
Manage files and notes
Check email (if configured)
Manage calendar
Control music
Manage Docker containers
Take screenshots
Run system commands
And much more!
**Commands:**
`!jarvis` or `!help` - Show this help
`!status` - Check bot status
`!clear` - Clear conversation history
`!join [channel_id]` - Join a voice channel (requires channel ID - right-click channel - Copy ID)
`!leave` - Leave current voice channel
`!listen` - Start listening to voice channel
`!stoplisten` - Stop listening to voice channel
`!test-voice` - Test voice functionality
Your messages are processed through JARVIS AI."""
        await self._send_message(session.channel_id, help_text)

    async def _handle_status(self, session: DiscordSession, message_data: dict, text: str) -> None:
        """Handle status command."""
        stats = self.get_stats()
        status_text = f"""**JARVIS Discord Bot Status**
Bot: {stats.get("bot_username", "Unknown")}
Running: {"Yes" if stats.get("running") else "No"}
Active Sessions: {stats.get("active_sessions", 0)}
**Your Session:**
Channel: {session.channel_id}
User: {session.username or "Unknown"}
Messages: {len(session.messages)}
Last Activity: {session.last_activity.strftime("%H:%M:%S")}"""
        await self._send_message(session.channel_id, status_text)

    async def _handle_clear(self, session: DiscordSession, message_data: dict, text: str) -> None:
        """Handle clear command."""
        session.messages.clear()
        await self._send_message(session.channel_id, "Conversation history cleared!")

    async def _get_user_voice_state(self, guild_id: str, user_id: str) -> dict | None:
        """Get voice state for a specific user in a guild."""
        if not self._http_client:
            return None
        try:
            response = await self._http_client.get(
                f"{self.API_URL}/guilds/{guild_id}/voice-states/{user_id}"
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            log.debug(f"[DISCORD] Failed to get user voice state: {e}")
        return None

    async def _handle_join_voice(
        self, session: DiscordSession, message_data: dict, text: str
    ) -> None:
        """Handle join voice channel command."""
        guild_id = session.guild_id
        if not guild_id:
            await self._send_message(session.channel_id, "This command only works in guilds.")
            return

        parts = text.split()
        channel_id = None

        if len(parts) > 1:
            channel_id = parts[1]
            log.debug(f"[DISCORD] Channel ID provided in command: {channel_id}")
        else:
            author_id = message_data.get("author", {}).get("id")
            if author_id:
                user_voice_state = await self._get_user_voice_state(guild_id, author_id)
                if user_voice_state:
                    channel_id = user_voice_state.get("channel_id")
                    if channel_id:
                        log.info(f"[DISCORD] Found user in voice channel: {channel_id}")

        if not channel_id:
            await self._send_message(
                session.channel_id,
                "You're not in a voice channel. Either:\n"
                "1. Join a voice channel first, then use `!join`\n"
                "2. Use `!join [channel_id]` with a specific channel ID\n\n"
                "To get a channel ID: Enable Developer Mode, right-click voice channel, Copy ID",
            )
            return

        success = await self._update_voice_state(guild_id, channel_id)
        if success:
            await self._send_message(session.channel_id, f"Joined voice channel <#{channel_id}>")
        else:
            await self._send_message(
                session.channel_id,
                f"Failed to join voice channel <#{channel_id}>. Please check:\n"
                "- The channel ID is correct\n"
                "- The channel exists and is a voice channel\n"
                "- The bot has permission to join the channel\n"
                "- The channel is not full",
            )

    async def _handle_leave_voice(
        self, session: DiscordSession, message_data: dict, text: str
    ) -> None:
        """Handle leave voice channel command."""
        guild_id = session.guild_id
        if not guild_id:
            await self._send_message(session.channel_id, "This command only works in guilds.")
            return
        success = await self._update_voice_state(guild_id, None)
        if success:
            await self._send_message(session.channel_id, "Left voice channel")
        else:
            await self._send_message(
                session.channel_id,
                "Failed to leave voice channel. There may be a connection issue.",
            )

    async def _handle_listen_voice(
        self, session: DiscordSession, message_data: dict, text: str
    ) -> None:
        """Handle listen to voice channel command."""
        guild_id = session.guild_id
        if not guild_id:
            await self._send_message(session.channel_id, "This command only works in guilds.")
            return
        voice_state = await self._get_voice_state(guild_id)
        if not voice_state or not voice_state.get("channel_id"):
            await self._send_message(session.channel_id, "Not in a voice channel. Use !join first.")
            return
        if not self._voice_player:
            await self._send_message(session.channel_id, "Voice player not available")
            return

        if not self._stt:
            await self._send_message(session.channel_id, "STT not available")
            return

        success = await self._voice_player.start_voice_receive(guild_id)
        if success:
            await self._send_message(
                session.channel_id, "Listening to voice channel... Speak and I'll respond!"
            )
        else:
            await self._send_message(session.channel_id, "Failed to start voice listening")

    async def _handle_voice_transcription(
        self, audio: np.ndarray, user_id: int, guild_id: str
    ) -> None:
        if not self._stt or not self.jarvis:
            return

        try:
            log.info(f"[DISCORD] Processing voice from user {user_id} in guild {guild_id}")
            audio_float = audio.astype(np.float32) / 32767.0
            transcription = await self._stt.transcribe_async(audio_float)

            if transcription and len(transcription.strip()) > 2:
                log.info(f"[DISCORD] Transcription: {transcription}")

                session = None
                for s in self._sessions.values():
                    if s.guild_id == guild_id:
                        session = s
                        break

                if not session:
                    channel_id = None
                    for s in self._sessions.values():
                        if s.guild_id == guild_id:
                            channel_id = s.channel_id
                            break
                    if channel_id:
                        session = self._get_or_create_session(
                            channel_id=channel_id, guild_id=guild_id
                        )

                if session:
                    await self._process_message_through_jarvis(
                        session, transcription, session.channel_id, None
                    )
        except Exception as e:
            log.error(f"[DISCORD] Error in voice transcription: {e}", exc_info=True)

    async def _handle_stop_listen_voice(
        self, session: DiscordSession, message_data: dict, text: str
    ) -> None:
        guild_id = session.guild_id
        if not guild_id:
            await self._send_message(session.channel_id, "This command only works in guilds.")
            return

        if not self._voice_player:
            await self._send_message(session.channel_id, "Voice player not available")
            return

        await self._voice_player.stop_voice_receive(guild_id)
        await self._send_message(session.channel_id, "Stopped listening to voice channel")

    async def _handle_test_voice(
        self, session: DiscordSession, message_data: dict, text: str
    ) -> None:
        """Test voice functionality."""
        if self._stt is None:
            await self._send_message(session.channel_id, "STT is not available")
            return
        try:
            import numpy as np

            test_audio = np.zeros(16000, dtype=np.float32)
            transcription = await self._stt.transcribe_async(test_audio)
            if HAS_VOICE_RECV:
                status = "discord-ext-voice-recv is available"
            else:
                status = "discord-ext-voice-recv is not available (install with: pip install discord-ext-voice-recv)"
            response = f"""**Voice Functionality Test**
{status}
STT Model: Loaded
Test Transcription: "{transcription or "(silent)"}"
**Voice Commands:**
`!join [channel_id]` - Join a voice channel
`!leave` - Leave current voice channel
`!listen` - Listen to voice channel (voice commands)
**Current Status:**
Gateway WebSocket: {self._running}
Voice Player: {self._voice_player is not None}
STT: {self._stt is not None}"""
            await self._send_message(session.channel_id, response)
        except Exception as e:
            await self._send_message(session.channel_id, f"Voice test failed: {e}")


discord_bot_handler = DiscordBotHandler()
