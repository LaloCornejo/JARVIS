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
from typing import TYPE_CHECKING, Any, Callable, TYPE_CHECKING

import httpx
import websockets

# Reduce websocket logging verbosity
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from jarvis.server import JarvisServer

log = logging.getLogger(__name__)


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
        # Trim old messages
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
        mention_only: bool = False,  # Changed default to False for better user experience
    ):
        # Don't load token in __init__, load it when needed
        self.token = ""
        self.jarvis: "JarvisServer | None" = jarvis_server
        self.poll_interval = poll_interval
        self.dm_only = dm_only
        self.mention_only = mention_only

        # Load allowed IDs from parameters or env vars
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

        # Connection state
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

        # Bot info
        self._bot_info: dict | None = None
        self._bot_user_id: str | None = None
        self._bot_username: str | None = None

        # Sessions
        self._sessions: dict[str, DiscordSession] = {}

        # HTTP client
        self._http_client: httpx.AsyncClient | None = None

        # Command handlers
        self._command_handlers: dict[str, Callable] = {
            "!jarvis": self._handle_help,
            "!help": self._handle_help,
            "!status": self._handle_status,
            "!clear": self._handle_clear,
        }

    async def start(self, jarvis_server: "JarvisServer | None" = None) -> bool:
        """Start the Discord bot handler."""
        if self._running:
            log.warning("[DISCORD] Handler already running")
            return True

        if jarvis_server:
            self.jarvis = jarvis_server

        # Load token when needed
        if not self.token:
            self.token = os.environ.get("DISCORD_BOT_TOKEN", "")

        if not self.token:
            log.error("[DISCORD] No bot token configured. Set DISCORD_BOT_TOKEN")
            return False

        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bot {self.token}"},
            timeout=30.0,
        )

        # Get bot info
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

        # Start WebSocket connection
        self._running = True
        self._ws_task = asyncio.create_task(self._websocket_loop())
        log.info("[DISCORD] Handler started")
        return True

    async def stop(self) -> None:
        """Stop the Discord bot handler."""
        if not self._running:
            return

        log.info("[DISCORD] Stopping handler...")
        self._running = False

        # Cancel tasks
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

        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        log.info("[DISCORD] Handler stopped")

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
                # Use resume URL if available
                gateway_url = self._resume_gateway_url or self.GATEWAY_URL

                log.info(f"[DISCORD] Connecting to Gateway: {gateway_url}")
                async with websockets.connect(gateway_url) as websocket:
                    self._websocket = websocket
                    log.info("[DISCORD] WebSocket connected")

                    # Reset reconnection delay on successful connection
                    reconnect_delay = 1.0

                    # Initialize zlib inflater for compressed payloads
                    self._zlib_inflater = zlib.decompressobj()

                    # Handle messages
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

            # Attempt reconnection
            if self._running:
                log.info(f"[DISCORD] Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _handle_gateway_message(self, message: bytes | str) -> None:
        """Handle a message from the Discord Gateway."""
        # Decompress if needed
        if isinstance(message, bytes):
            try:
                # Check for zlib suffix
                if message.endswith(self._zlib_suffix):
                    message = self._zlib_inflater.decompress(message)
                    message = message.decode("utf-8")
                else:
                    # Partial message, decompress what we have
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

        # Update sequence number
        sequence = payload.get("s")
        if sequence is not None:
            self._sequence_number = sequence

        op_code = payload.get("op")
        event_type = payload.get("t")
        event_data = payload.get("d", {})

        # Handle opcode
        if op_code == 10:  # Hello
            await self._handle_hello(event_data)
        elif op_code == 11:  # Heartbeat ACK
            self._last_heartbeat_ack = True
        elif op_code == 0:  # Dispatch
            await self._handle_dispatch(event_type, event_data)
        elif op_code == 7:  # Reconnect
            log.info("[DISCORD] Received reconnect request")
            self._resume_gateway_url = None  # Force full reconnect
        elif op_code == 9:  # Invalid Session
            log.warning("[DISCORD] Invalid session, re-identifying")
            self._session_id = None
            self._sequence_number = None
            await asyncio.sleep(5)
            await self._send_identify()

    async def _handle_hello(self, data: dict) -> None:
        """Handle Gateway Hello event."""
        self._heartbeat_interval = data.get("heartbeat_interval", 45000) / 1000

        # Start heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Identify or resume
        if self._session_id and self._sequence_number:
            await self._send_resume()
        else:
            await self._send_identify()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._running and self._websocket:
            try:
                # Wait for heartbeat interval
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
                "intents": 33280,  # GUILD_MESSAGES (512) + MESSAGE_CONTENT (32768)
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

    async def _handle_message_create(self, data: dict) -> None:
        """Handle incoming message."""
        # Skip messages from the bot itself
        author = data.get("author", {})
        if author.get("id") == self._bot_user_id:
            return

        # Skip bot messages if desired (optional)
        if author.get("bot"):
            return

        # Get message info
        content = data.get("content", "")
        channel_id = data.get("channel_id")
        guild_id = data.get("guild_id")
        message_id = data.get("id")

        # Get channel info
        channel_data = data.get("channel", {})
        is_dm = guild_id is None

        log.info(
            f"[DISCORD] Received message - Channel: {channel_id}, DM: {is_dm}, "
            f"Author: {author.get('username')}, Content: {content[:100]}..."
        )

        # Check permissions
        if self.dm_only and not is_dm:
            log.debug("[DISCORD] Message filtered out - dm_only is True but this is not a DM")
            return

        if self.allowed_channel_ids and channel_id not in self.allowed_channel_ids:
            log.debug(f"[DISCORD] Message filtered out - Channel {channel_id} not in allowed list")
            return

        if self.allowed_guild_ids and guild_id and guild_id not in self.allowed_guild_ids:
            log.debug(f"[DISCORD] Message filtered out - Guild {guild_id} not in allowed list")
            return

        # Check for mentions
        mentions = data.get("mentions", [])
        bot_mentioned = any(m.get("id") == self._bot_user_id for m in mentions)

        # In guilds, check if we should respond to all messages or only mentions/commands
        if not is_dm:
            # If mention_only is True, only respond to mentions or commands
            if self.mention_only:
                if not bot_mentioned and not any(
                    content.startswith(cmd) for cmd in self._command_handlers
                ):
                    log.debug(
                        "[DISCORD] Message filtered out - mention_only is True but bot not mentioned"
                    )
                    return

        # Get or create session
        session = self._get_or_create_session(
            channel_id=channel_id,
            guild_id=guild_id,
            user_id=author.get("id"),
            username=author.get("username"),
        )

        log.info(
            f"[DISCORD] Processing message from {author.get('username')} in {channel_id}: {content[:100]}..."
        )

        # Check for commands
        for cmd, handler in self._command_handlers.items():
            if content.startswith(cmd):
                log.info(f"[DISCORD] Handling command: {cmd}")
                await handler(session, data, content)
                return

        # Process message through JARVIS
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
            await self._send_message(channel_id, "❌ JARVIS server not available")
            return

        # Add user message to session
        session.add_message("user", text)

        # Send "typing" indicator
        await self._send_typing(channel_id)

        try:
            # Process through JARVIS server
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
            log.debug(f"[DISCORD] Starting process_message call")
            await self.jarvis.process_message(text, broadcast_func=capture_broadcast)
            log.debug(f"[DISCORD] Finished process_message call")
            log.debug(
                f"[DISCORD] Response chunks: {len(response_chunks)}, Full response: {len(full_response) if full_response else 0} chars"
            )

            # If we got chunks but no complete message, join them
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)
                log.debug(
                    f"[DISCORD] Joined chunks to form full response: {len(full_response)} chars"
                )

            if full_response:
                log.info(f"[DISCORD] Got response from JARVIS: {len(full_response)} chars")
                # Add assistant response to session
                session.add_message("assistant", full_response)

                # Check if TTS service is available using the same health check as server
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
                    # Generate audio from response
                    try:
                        log.info("[DISCORD] Generating TTS audio...")
                        audio_data = await self.jarvis.tts.speak_to_audio(full_response)

                        # Send audio message
                        await self._send_audio_message(
                            channel_id,
                            audio_data,
                            filename="response.mp3",
                            text_content=full_response,  # Include text as well
                            reply_to=message_id,
                        )
                        log.info("[DISCORD] Audio message sent successfully")
                    except Exception as e:
                        log.error(f"[DISCORD] Error generating/sending TTS: {e}")
                        # Fallback to text message
                        await self._send_long_message(
                            channel_id, full_response, reply_to=message_id
                        )
                else:
                    # Send text response
                    await self._send_long_message(channel_id, full_response, reply_to=message_id)
            else:
                log.warning("[DISCORD] No response from JARVIS")
                await self._send_message(
                    channel_id, "🤔 I didn't get a response. Please try again.", reply_to=message_id
                )

        except Exception as e:
            log.error(f"[DISCORD] Error processing message: {e}", exc_info=True)
            await self._send_message(channel_id, f"❌ Error: {str(e)[:200]}", reply_to=message_id)

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

            # Create file-like object - important: set position to 0
            audio_file = io.BytesIO(audio_data)
            audio_file.seek(0)

            # For Discord file uploads with additional fields, we need to use payload_json
            payload_data = {}
            if text_content:
                payload_data["content"] = text_content
            if reply_to:
                payload_data["message_reference"] = {"message_id": reply_to}

            # If we don't have any payload data, we can send the file directly with content
            if not payload_data and text_content:
                # Simple case: just send file with text content
                files = {"file": (filename, audio_file, "audio/mpeg")}
                data = {"content": text_content}
            else:
                # Complex case: use payload_json for proper structuring
                files = {"file": (filename, audio_file, "audio/mpeg")}
                data = {"payload_json": json.dumps(payload_data)}

            # Use httpx multipart encoding directly
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

        try:
            import io

            # Create file-like object - important: set position to 0
            audio_file = io.BytesIO(audio_data)
            audio_file.seek(0)

            # Prepare multipart data using httpx standard approach
            files = {"file": (filename, audio_file, "audio/mpeg")}

            # For Discord API, we need to send the data as form fields
            # message_reference should be sent as a separate form field if needed
            data = {}
            if text_content:
                data["content"] = text_content
            if reply_to:
                # Message reference should be sent as JSON in the form data
                data["payload_json"] = json.dumps({"message_reference": {"message_id": reply_to}})

            # If we have both text content and reply_to, we need to use payload_json
            if text_content and reply_to:
                data = {
                    "payload_json": json.dumps(
                        {"content": text_content, "message_reference": {"message_id": reply_to}}
                    )
                }
            elif text_content:
                data["content"] = text_content
            elif reply_to:
                data["payload_json"] = json.dumps({"message_reference": {"message_id": reply_to}})

            # Use httpx multipart encoding directly
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
        # Discord message limit is 2000 characters
        max_length = 1950

        if len(text) <= max_length:
            await self._send_message(channel_id, text, reply_to)
            return

        # Split into chunks
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            # Find a good breaking point (newline or space)
            split_point = text.rfind("\n", 0, max_length)
            if split_point == -1:
                split_point = text.rfind(" ", 0, max_length)
            if split_point == -1:
                split_point = max_length

            chunks.append(text[:split_point])
            text = text[split_point:].strip()

        # Send chunks
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

    # Command Handlers

    async def _handle_help(self, session: DiscordSession, message_data: dict, text: str) -> None:
        """Handle help command."""
        help_text = """🤖 **JARVIS Discord Bot Help**

Just send me a message and I'll respond! I can:

🌐 Search the web
📝 Manage files and notes
📧 Check email (if configured)
📅 Manage calendar
🎵 Control music
🐳 Manage Docker containers
📸 Take screenshots
🔧 Run system commands
💬 And much more!

**Commands:**
`!jarvis` or `!help` - Show this help
`!status` - Check bot status
`!clear` - Clear conversation history

Your messages are processed through JARVIS AI."""
        await self._send_message(session.channel_id, help_text)

    async def _handle_status(self, session: DiscordSession, message_data: dict, text: str) -> None:
        """Handle status command."""
        stats = self.get_stats()
        status_text = f"""📊 **JARVIS Discord Bot Status**

🤖 Bot: {stats.get("bot_username", "Unknown")}
🔄 Running: {"✅ Yes" if stats.get("running") else "❌ No"}
💬 Active Sessions: {stats.get("active_sessions", 0)}

**Your Session:**
🆔 Channel: {session.channel_id}
👤 User: {session.username or "Unknown"}
💭 Messages: {len(session.messages)}
🕐 Last Activity: {session.last_activity.strftime("%H:%M:%S")}"""
        await self._send_message(session.channel_id, status_text)

    async def _handle_clear(self, session: DiscordSession, message_data: dict, text: str) -> None:
        """Handle clear command."""
        session.messages.clear()
        await self._send_message(session.channel_id, "🗑️ Conversation history cleared!")


# Global handler instance
discord_bot_handler = DiscordBotHandler()


async def start_discord_bot(jarvis_server: "JarvisServer | None" = None) -> bool:
    """Convenience function to start the Discord bot."""
    return await discord_bot_handler.start(jarvis_server)


async def stop_discord_bot() -> None:
    """Convenience function to stop the Discord bot."""
    await discord_bot_handler.stop()
