"""Telegram Bot Handler for JARVIS - Active two-way communication.

This module provides a background service that:
- Polls Telegram for incoming messages
- Routes messages to JARVIS for processing
- Sends responses back to Telegram users
- Maintains conversation context per chat

Usage:
    from core.telegram_bot import telegram_bot_handler
    await telegram_bot_handler.start()
    # ... JARVIS running ...
    await telegram_bot_handler.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from jarvis.server import JarvisServer

from core.security.rate_limit import RateLimitConfig, rate_limiter_manager
from tools.integrations.telegram import get_telegram_client

log = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Maintains conversation state for a Telegram chat."""

    chat_id: int | str
    chat_type: str  # private, group, supergroup, channel
    chat_title: str | None = None
    username: str | None = None
    first_name: str | None = None
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
            "chat_id": self.chat_id,
            "chat_type": self.chat_type,
            "chat_title": self.chat_title,
            "username": self.username,
            "first_name": self.first_name,
            "message_count": len(self.messages),
            "last_activity": self.last_activity.isoformat(),
        }


class TelegramBotHandler:
    """Handles active two-way Telegram communication with JARVIS."""

    def __init__(
        self,
        jarvis_server: JarvisServer | None = None,
        poll_interval: float = 2.0,
        allowed_chat_ids: list[int | str] | None = None,
    ):
        self.client = get_telegram_client()
        self.jarvis = jarvis_server
        self.poll_interval = poll_interval

        # Load allowed chat IDs from parameter, env var, or config
        if allowed_chat_ids is not None:
            self.allowed_chat_ids = set(allowed_chat_ids)
        else:
            # Try to load from environment variable
            env_ids = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
            if env_ids:
                self.allowed_chat_ids = set(id.strip() for id in env_ids.split(",") if id.strip())
            else:
                # Security default: block all chats until explicitly configured
                # Set TELEGRAM_ALLOWED_CHAT_IDS="*" to allow all (development only)
                self.allowed_chat_ids = set()

        # Check for explicit open access mode
        self._allow_all_chats = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "") == "*"
        if self._allow_all_chats:
            log.warning(
                "[TELEGRAM] SECURITY WARNING: Allowing all Telegram chats. Set TELEGRAM_ALLOWED_CHAT_IDS to restrict access."
            )

        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._last_update_id: int | None = None
        self._sessions: dict[int | str, ChatSession] = {}
        self._command_handlers: dict[str, Callable] = {
            "/start": self._handle_start,
            "/help": self._handle_help,
            "/status": self._handle_status,
            "/clear": self._handle_clear,
            "/restart": self._handle_restart,
        }

        # Restart callback - set by external code to handle restart
        self._restart_callback: Callable | None = None

        # Bot info
        self._bot_info: dict | None = None
        self._bot_username: str | None = None

        # Rate limiting (10 requests per 30 seconds per chat)
        self._rate_limiter = rate_limiter_manager.get_limiter(
            "telegram",
            RateLimitConfig(requests=10, window_seconds=30),
        )

    async def start(self, jarvis_server: JarvisServer | None = None) -> bool:
        """Start the Telegram bot handler."""
        if self._running:
            log.warning("[TELEGRAM] Handler already running")
            return True

        if jarvis_server:
            self.jarvis = jarvis_server

        if not self.client.token:
            log.error("[TELEGRAM] No bot token configured. Set TELEGRAM_BOT_TOKEN")
            return False

        # Get bot info
        self._bot_info = await self.client.get_me()
        if not self._bot_info:
            log.error("[TELEGRAM] Failed to connect to Telegram. Check your token.")
            return False

        self._bot_username = self._bot_info.get("username")
        log.info(f"[TELEGRAM] Bot connected: @{self._bot_username}")

        # Security check: warn if no chat restrictions configured
        if not self.allowed_chat_ids and not self._allow_all_chats:
            log.warning(
                "[TELEGRAM] SECURITY: No allowed_chat_ids configured. Bot will NOT respond to any chats until TELEGRAM_ALLOWED_CHAT_IDS is set."
            )

        # Get initial update offset (clear old messages)
        updates = await self.client.get_updates(limit=1)
        if updates:
            self._last_update_id = updates[-1]["update_id"]
            log.info(f"[TELEGRAM] Starting from update_id: {self._last_update_id}")

        # Start polling
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("[TELEGRAM] Handler started")
        return True

    async def stop(self) -> None:
        """Stop the Telegram bot handler."""
        if not self._running:
            return

        log.info("[TELEGRAM] Stopping handler...")
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        await self.client.close()
        log.info("[TELEGRAM] Handler stopped")

    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "running": self._running,
            "bot_username": self._bot_username,
            "active_sessions": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
            "last_update_id": self._last_update_id,
            "allowed_chat_ids": list(self.allowed_chat_ids) if self.allowed_chat_ids else None,
        }

    async def _poll_loop(self) -> None:
        """Main polling loop for Telegram updates."""
        log.info(f"[TELEGRAM] Polling started (interval: {self.poll_interval}s)")

        while self._running:
            try:
                await self._process_updates()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[TELEGRAM] Poll error: {e}")
                await asyncio.sleep(self.poll_interval)

        log.info("[TELEGRAM] Polling stopped")

    async def _process_updates(self) -> None:
        """Fetch and process new updates from Telegram."""
        # Calculate offset for next updates
        offset = self._last_update_id + 1 if self._last_update_id else None

        updates = await self.client.get_updates(
            offset=offset,
            limit=100,
            timeout=0,
        )

        if not updates:
            return

        for update in updates:
            try:
                await self._handle_update(update)
            except Exception as e:
                log.error(f"[TELEGRAM] Error handling update {update.get('update_id')}: {e}")

            # Track last processed update
            self._last_update_id = update.get("update_id")

    async def _handle_update(self, update: dict) -> None:
        """Handle a single Telegram update."""
        # Extract message data
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        # Skip messages without text
        text = message.get("text", "").strip()
        if not text:
            return

        # Get chat info
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        chat_title = chat.get("title") or chat.get("username")

        # Check if chat is allowed (if whitelist is configured)
        # Allow if: explicit open access mode OR chat_id in allowed list
        if (
            not self._allow_all_chats
            and self.allowed_chat_ids
            and chat_id not in self.allowed_chat_ids
        ):
            log.warning(f"[TELEGRAM] Ignoring message from unauthorized chat: {chat_id}")
            return

        # Rate limit check
        allowed, blocked_until = await self._rate_limiter.is_allowed(str(chat_id))
        if not allowed:
            log.warning(f"[TELEGRAM] Rate limited chat: {chat_id}")
            await self.client.send_message(
                chat_id,
                "Too many requests. Please wait a moment.",
            )
            return

        # Get sender info
        from_user = message.get("from", {})
        user_id = from_user.get("id")
        username = from_user.get("username")
        first_name = from_user.get("first_name")

        # Get or create session
        session = self._get_or_create_session(
            chat_id=chat_id,
            chat_type=chat_type,
            chat_title=chat_title,
            username=username,
            first_name=first_name,
        )

        log.info(
            f"[TELEGRAM] Message from {first_name or username} in {chat_title or chat_id}: {text[:50]}..."
        )

        # Check for bot commands
        if text.startswith("/"):
            command = text.split()[0].split("@")[0]  # Handle /command@botname
            if command in self._command_handlers:
                await self._command_handlers[command](session, text)
                return

        # Process message through JARVIS
        await self._process_message_through_jarvis(session, text, chat_id)

    async def _process_message_through_jarvis(
        self, session: ChatSession, text: str, chat_id: int | str
    ) -> None:
        """Send message to JARVIS and return response."""
        if not self.jarvis:
            log.error("[TELEGRAM] No JarvisServer configured")
            await self._send_message(chat_id, "❌ JARVIS server not available")
            return

        # Add user message to session
        session.add_message("user", text)

        # Send "typing" indicator (optional, would need additional API call)

        try:
            # Process through JARVIS server
            # Collect streaming response
            full_response = ""

            # Create a custom broadcast function to capture response
            response_chunks = []

            async def capture_broadcast(message: dict) -> None:
                nonlocal full_response
                if message.get("type") == "streaming_chunk":
                    response_chunks.append(message.get("content", ""))
                elif message.get("type") == "message_complete":
                    full_response = message.get("full_response", "")

            # Process the message
            await self.jarvis.process_message(text, broadcast_func=capture_broadcast)

            # If we got chunks but no complete message, join them
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)

            if full_response:
                # Add assistant response to session
                session.add_message("assistant", full_response)

                # Check if TTS service is available using the same health check as server
                tts_available = False
                if hasattr(self.jarvis, "tts") and self.jarvis.tts:
                    try:
                        tts_available = await self.jarvis.tts.health_check()
                        if tts_available:
                            log.info("[TELEGRAM] TTS service is available")
                        else:
                            log.debug("[TELEGRAM] TTS service is not available")
                    except Exception as e:
                        log.warning(f"[TELEGRAM] TTS health check failed: {e}")
                        tts_available = False

                if tts_available:
                    # Generate audio from response
                    try:
                        log.info("[TELEGRAM] Generating TTS audio...")
                        audio_data = await self.jarvis.tts.speak_to_audio(full_response)

                        # Send audio message to Telegram
                        await self._send_audio_message(
                            chat_id,
                            audio_data,
                            filename="response.mp3",
                            text_content=full_response,  # Include text as well
                        )
                        log.info("[TELEGRAM] Audio message sent successfully")
                    except Exception as e:
                        log.error(f"[TELEGRAM] Error generating/sending TTS: {e}")
                        # Fallback to text message
                        await self._send_long_message(chat_id, full_response)
                else:
                    # Send text response
                    await self._send_long_message(chat_id, full_response)
            else:
                await self._send_message(chat_id, "🤔 I didn't get a response. Please try again.")

        except Exception as e:
            log.error(f"[TELEGRAM] Error processing message: {e}")
            await self._send_message(chat_id, f"❌ Error: {str(e)[:200]}")

    async def _send_message(self, chat_id: int | str, text: str) -> None:
        """Send a message to Telegram."""
        try:
            await self.client.send_message(chat_id, text)
        except Exception as e:
            log.error(f"[TELEGRAM] Failed to send message: {e}")

    async def _send_long_message(self, chat_id: int | str, text: str) -> None:
        """Send a potentially long message, splitting if needed."""
        # Telegram message limit is 4096 characters
        max_length = 4000

        if len(text) <= max_length:
            await self._send_message(chat_id, text)
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

        # Send chunks with index
        for i, chunk in enumerate(chunks, 1):
            prefix = f"[{i}/{len(chunks)}] " if len(chunks) > 1 else ""
            await self._send_message(chat_id, prefix + chunk)
            if i < len(chunks):
                await asyncio.sleep(0.5)  # Small delay between chunks

    async def _send_audio_message(
        self,
        chat_id: int | str,
        audio_data: bytes,
        filename: str = "response.mp3",
        text_content: str | None = None,
    ) -> None:
        """Send an audio message to Telegram with optional text content."""
        try:
            # Send the audio file using the Telegram client directly from memory
            result = await self.client.send_document(
                chat_id=chat_id,
                document_data=audio_data,
                filename=filename,
                caption=text_content[:1024]
                if text_content
                else None,  # Telegram caption limit is 1024 chars
            )

            if result:
                log.info(f"[TELEGRAM] Audio message sent successfully to chat {chat_id}")
            else:
                log.error("[TELEGRAM] Failed to send audio message")
                # Fallback to text message
                if text_content:
                    await self._send_long_message(chat_id, text_content)
        except Exception as e:
            log.error(f"[TELEGRAM] Error sending audio message: {e}")
            # Fallback to text message
            if text_content:
                await self._send_long_message(chat_id, text_content)

    def _get_or_create_session(
        self,
        chat_id: int | str,
        chat_type: str,
        chat_title: str | None = None,
        username: str | None = None,
        first_name: str | None = None,
    ) -> ChatSession:
        """Get existing session or create new one."""
        if chat_id not in self._sessions:
            self._sessions[chat_id] = ChatSession(
                chat_id=chat_id,
                chat_type=chat_type,
                chat_title=chat_title,
                username=username,
                first_name=first_name,
            )
            log.info(f"[TELEGRAM] New session created for chat: {chat_title or chat_id}")
        return self._sessions[chat_id]

    # Command Handlers

    async def _handle_start(self, session: ChatSession, text: str) -> None:
        """Handle /start command."""
        welcome = f"""👋 Hello {session.first_name or "there"}!

I'm JARVIS, your AI assistant connected via Telegram.

You can:
• Chat with me normally
• Ask me questions
• Request tasks
• Use my tools and integrations

Commands:
/help - Show help
/status - Check bot status
/clear - Clear conversation history

How can I help you today?"""
        await self._send_message(session.chat_id, welcome)

    async def _handle_help(self, session: ChatSession, text: str) -> None:
        """Handle /help command."""
        help_text = """🤖 JARVIS Telegram Bot Help

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

Commands:
/start - Start conversation
/help - Show this help
/status - Check status
/clear - Clear history
/restart - Restart JARVIS server (admin only)

Your messages are processed through JARVIS AI."""
        await self._send_message(session.chat_id, help_text)

    async def _handle_status(self, session: ChatSession, text: str) -> None:
        """Handle /status command."""
        stats = self.get_stats()
        status_text = f"""📊 JARVIS Telegram Bot Status

🤖 Bot: @{stats.get("bot_username", "Unknown")}
🔄 Running: {"✅ Yes" if stats.get("running") else "❌ No"}
💬 Active Sessions: {stats.get("active_sessions", 0)}
📨 Last Update ID: {stats.get("last_update_id", "N/A")}

Your Session:
🆔 Chat ID: {session.chat_id}
👤 User: {session.first_name or session.username or "Unknown"}
💭 Messages: {len(session.messages)}
🕐 Last Activity: {session.last_activity.strftime("%H:%M:%S")}"""
        await self._send_message(session.chat_id, status_text)

    async def _handle_clear(self, session: ChatSession, text: str) -> None:
        """Handle /clear command."""
        session.messages.clear()
        await self._send_message(session.chat_id, "🗑️ Conversation history cleared!")

    async def _handle_restart(self, session: ChatSession, text: str) -> None:
        """Handle /restart command - restarts JARVIS server."""
        await self._send_message(
            session.chat_id,
            "🔄 Restarting JARVIS server...\nThis may take a few moments. The bot will be back online shortly.",
        )

        log.warning(
            f"[TELEGRAM] Restart requested by {session.first_name or session.username} (chat_id: {session.chat_id})"
        )

        # Trigger restart if callback is set
        if self._restart_callback:
            try:
                # Run restart in background so we can send the message first
                asyncio.create_task(self._trigger_restart())
            except Exception as e:
                log.error(f"[TELEGRAM] Error triggering restart: {e}")
                await self._send_message(session.chat_id, f"❌ Failed to restart: {e}")
        else:
            await self._send_message(
                session.chat_id, "⚠️ Restart is not configured. Please restart the server manually."
            )

    async def _trigger_restart(self) -> None:
        """Trigger the restart callback after a short delay to allow message to be sent."""
        await asyncio.sleep(2)  # Give time for the restart message to be delivered
        if self._restart_callback:
            await self._restart_callback()

    def set_restart_callback(self, callback: Callable) -> None:
        """Set the callback function that will be called to restart the server."""
        self._restart_callback = callback
        log.info("[TELEGRAM] Restart callback registered")


# Global handler instance
telegram_bot_handler = TelegramBotHandler()


async def start_telegram_bot(jarvis_server: JarvisServer | None = None) -> bool:
    """Convenience function to start the Telegram bot."""
    return await telegram_bot_handler.start(jarvis_server)


async def stop_telegram_bot() -> None:
    """Convenience function to stop the Telegram bot."""
    await telegram_bot_handler.stop()
