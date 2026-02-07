"""WhatsApp Web Bot Handler for JARVIS - Baileys-style implementation.

This module provides a WhatsApp Web client that:
- Uses QR code login similar to OpenClaw/Baileys
- Receives and processes messages from WhatsApp Web
- Routes messages to JARVIS for processing
- Sends responses back to WhatsApp contacts
- Maintains conversation context per user

This implementation follows the same approach as OpenClaw, which uses the Baileys library
(a TypeScript/JavaScript library) to connect directly to WhatsApp Web via WebSocket protocol,
rather than using the WhatsApp Business API or browser automation.

Usage:
    from core.whatsapp_bot import whatsapp_bot_handler
    await whatsapp_bot_handler.start()
    # ... JARVIS running ...
    await whatsapp_bot_handler.stop()

Setup:
    1. Run the bot - it will display a QR code in the terminal
    2. Scan the QR code with your WhatsApp mobile app
    3. The bot will automatically start receiving messages

Note: This is a placeholder implementation that shows the structure. A full implementation
would require a Python equivalent of the Baileys library to connect to WhatsApp Web directly
via WebSocket protocol, handle authentication, and manage the WhatsApp Web protocol.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

# Try to import required packages
try:
    import io

    import qrcode
    import socketio
    from PIL import Image

    WHATSAPP_DEPS_AVAILABLE = True
except ImportError:
    WHATSAPP_DEPS_AVAILABLE = False

if TYPE_CHECKING:
    from jarvis.server import JarvisServer

log = logging.getLogger(__name__)

# Log dependency availability
if not WHATSAPP_DEPS_AVAILABLE:
    log.warning("[WHATSAPP] Required dependencies not available. WhatsApp bot will not function.")
    log.warning("[WHATSAPP] Install with: pip install python-socketio qrcode Pillow")


@dataclass
class WhatsAppSession:
    """Maintains conversation state for a WhatsApp user."""

    user_id: str
    user_name: str
    phone_number: str = ""
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
            "user_id": self.user_id,
            "user_name": self.user_name,
            "phone_number": self.phone_number,
            "message_count": len(self.messages),
            "last_activity": self.last_activity.isoformat(),
        }


class WhatsAppBotHandler:
    """Handles WhatsApp Web communication with JARVIS using QR code login (Baileys-style)."""

    def __init__(
        self,
        jarvis_server: Optional["JarvisServer"] = None,
        allowed_contacts: Optional[List[str]] = None,
    ):
        self.jarvis = jarvis_server

        # Load allowed contacts from parameter or env var
        if allowed_contacts is not None:
            self.allowed_contacts = set(allowed_contacts)
        else:
            env_contacts = os.environ.get("WHATSAPP_ALLOWED_CONTACTS", "")
            self.allowed_contacts = (
                set(contact.strip() for contact in env_contacts.split(",") if contact.strip())
                or None
            )

        # State
        self._running = False
        self._message_poll_task: Optional[asyncio.Task] = None

        # Sessions
        self._sessions: Dict[str, WhatsAppSession] = {}

        # Command handlers
        self._command_handlers: Dict[str, Callable] = {
            "/start": self._handle_start,
            "/help": self._handle_help,
            "help": self._handle_help,
            "/status": self._handle_status,
            "/clear": self._handle_clear,
        }

        # Authentication state storage
        self.auth_dir = Path("./whatsapp_auth")
        self.auth_dir.mkdir(exist_ok=True)

    async def start(self, jarvis_server: Optional["JarvisServer"] = None) -> bool:
        """Start the WhatsApp Web bot handler."""
        if not WHATSAPP_DEPS_AVAILABLE:
            log.error("[WHATSAPP] Required dependencies not available. Cannot start WhatsApp bot.")
            return False

        if self._running:
            log.warning("[WHATSAPP] Handler already running")
            return True

        if jarvis_server:
            self.jarvis = jarvis_server

        try:
            log.info("[WHATSAPP] Starting WhatsApp Web client (Baileys-style implementation)...")

            # Create auth directory
            self.auth_dir.mkdir(exist_ok=True)

            # This is a placeholder implementation that shows the structure
            # A full implementation would connect to WhatsApp Web servers via WebSocket
            log.info("[WHATSAPP] NOTE: This is a placeholder implementation.")
            log.info(
                "[WHATSAPP] A full implementation would require a Python equivalent of the Baileys library"
            )
            log.info("[WHATSAPP] to connect directly to WhatsApp Web via WebSocket protocol.")

            # Simulate QR code display
            self._display_simulated_qr()

            # Start polling for messages (simulated)
            self._running = True
            self._message_poll_task = asyncio.create_task(self._poll_loop())

            log.info("[WHATSAPP] WhatsApp Web client initialized (placeholder)")
            return True

        except Exception as e:
            log.error(f"[WHATSAPP] Error starting bot: {e}")
            await self.stop()
            return False

    def _display_simulated_qr(self) -> None:
        """Display a simulated QR code for WhatsApp Web login."""
        try:
            # Create a simulated QR code
            qr_data = "https://web.whatsapp.com/login?token=simulated_token_for_jarvis"

            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(qr_data)
            qr.make(fit=True)

            # Create image
            img = qr.make_image(fill_color="black", back_color="white")

            log.info("[WHATSAPP] ===== SIMULATED QR CODE =====")
            log.info("[WHATSAPP] Scan this QR code with your WhatsApp app:")
            log.info(f"[WHATSAPP] {qr_data}")
            log.info("[WHATSAPP] ===============================")

            # In a real implementation, we would display the actual QR code
            # For now, we just log that it would be displayed

        except Exception as e:
            log.error(f"[WHATSAPP] Error generating QR code: {e}")

    async def stop(self) -> None:
        """Stop the WhatsApp Web bot handler."""
        if not self._running:
            return

        log.info("[WHATSAPP] Stopping handler...")
        self._running = False

        if self._message_poll_task:
            self._message_poll_task.cancel()
            try:
                await self._message_poll_task
            except asyncio.CancelledError:
                pass
            self._message_poll_task = None

        log.info("[WHATSAPP] Handler stopped")

    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running and WHATSAPP_DEPS_AVAILABLE

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "running": self._running and WHATSAPP_DEPS_AVAILABLE,
            "dependencies_available": WHATSAPP_DEPS_AVAILABLE,
            "active_sessions": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
            "allowed_contacts": list(self.allowed_contacts) if self.allowed_contacts else None,
        }

    async def _poll_loop(self) -> None:
        """Poll for new messages (simulated)."""
        log.info("[WHATSAPP] Started polling for messages")

        while self._running:
            try:
                # In a real implementation, this would check for actual WhatsApp messages
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[WHATSAPP] Error in poll loop: {e}")
                await asyncio.sleep(5)

    async def _handle_incoming_message(
        self, phone_number: str, contact_name: str, message_text: str
    ) -> None:
        """Handle an incoming WhatsApp message."""
        # Check if contact is allowed
        if self.allowed_contacts:
            # Check both phone number and contact name
            if (
                phone_number not in self.allowed_contacts
                and contact_name not in self.allowed_contacts
            ):
                log.warning(
                    f"[WHATSAPP] Ignoring message from unauthorized contact: {contact_name} ({phone_number})"
                )
                return

        # Get or create session
        session = self._get_or_create_session(
            user_id=phone_number, user_name=contact_name, phone_number=phone_number
        )

        log.info(f"[WHATSAPP] Message from {contact_name} ({phone_number}): {message_text[:50]}...")

        # Check for commands
        if message_text.startswith(tuple(self._command_handlers.keys())):
            command = message_text.split()[0]
            if command in self._command_handlers:
                await self._command_handlers[command](session, message_text)
                return

        # Process message through JARVIS
        await self._process_message_through_jarvis(
            session, message_text, contact_name, phone_number
        )

    async def _process_message_through_jarvis(
        self,
        session: WhatsAppSession,
        text: str,
        contact_name: str,
        phone_number: str,
    ) -> None:
        """Send message to JARVIS and return response."""
        if not self.jarvis:
            log.error("[WHATSAPP] No JarvisServer configured")
            await self._send_message(to=phone_number, text="❌ JARVIS server not available")
            return

        # Add user message to session
        session.add_message("user", text)

        try:
            # Process through JARVIS server
            full_response = ""
            response_chunks = []

            async def capture_broadcast(message: dict) -> None:
                nonlocal full_response
                if message.get("type") == "streaming_chunk":
                    response_chunks.append(message.get("content", ""))
                elif message.get("type") == "message_complete":
                    full_response = message.get("full_response", "")

            await self.jarvis.process_message(text, broadcast_func=capture_broadcast)

            # If we got chunks but no complete message, join them
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)

            if full_response:
                # Add assistant response to session
                session.add_message("assistant", full_response)

                # Send response to WhatsApp (split if too long)
                await self._send_long_message(to=phone_number, text=full_response)
            else:
                await self._send_message(
                    to=phone_number, text="🤔 I didn't get a response. Please try again."
                )

        except Exception as e:
            log.error(f"[WHATSAPP] Error processing message: {e}")
            await self._send_message(to=phone_number, text=f"❌ Error: {str(e)[:200]}")

    async def _send_message(self, to: str, text: str) -> bool:
        """Send a message to WhatsApp."""
        # In a real implementation, this would send the message via WhatsApp Web
        log.info(f"[WHATSAPP] Would send message to {to}: {text[:50]}...")

        # Simulate message sending delay
        await asyncio.sleep(0.1)

        return True

    async def _send_long_message(self, to: str, text: str) -> None:
        """Send a potentially long message, splitting if needed."""
        # WhatsApp has a limit of around 65,536 characters, but let's use 3000 for readability
        max_length = 3000

        if len(text) <= max_length:
            await self._send_message(to=to, text=text)
            return

        # Split into chunks
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            # Find a good breaking point (newline, period, or space)
            split_points = [
                text.rfind("\n", 0, max_length),
                text.rfind(". ", 0, max_length),
                text.rfind(" ", 0, max_length),
            ]
            split_point = (
                max(point for point in split_points if point != -1)
                if any(point != -1 for point in split_points)
                else max_length
            )

            chunks.append(text[:split_point])
            text = text[split_point:].strip()

        # Send chunks
        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)}) " if len(chunks) > 1 else ""
            await self._send_message(to=to, text=prefix + chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)  # Small delay between chunks

    def _get_or_create_session(
        self, user_id: str, user_name: str, phone_number: str
    ) -> WhatsAppSession:
        """Get existing session or create new one."""
        if user_id not in self._sessions:
            self._sessions[user_id] = WhatsAppSession(
                user_id=user_id,
                user_name=user_name,
                phone_number=phone_number,
            )
            log.info(f"[WHATSAPP] New session created for user: {user_name} ({user_id})")
        return self._sessions[user_id]

    # Command Handlers

    async def _handle_start(self, session: WhatsAppSession, text: str) -> None:
        """Handle /start command."""
        welcome = f"""👋 Hello {session.user_name}!

I'm JARVIS, your AI assistant connected via WhatsApp.

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
        await self._send_message(to=session.phone_number, text=welcome)

    async def _handle_help(self, session: WhatsAppSession, text: str) -> None:
        """Handle /help command."""
        help_text = """🤖 JARVIS WhatsApp Bot Help

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

Your messages are processed through JARVIS AI."""
        await self._send_message(to=session.phone_number, text=help_text)

    async def _handle_status(self, session: WhatsAppSession, text: str) -> None:
        """Handle /status command."""
        stats = self.get_stats()
        status_text = f"""📊 JARVIS WhatsApp Bot Status

🔄 Running: {"✅ Yes" if stats.get("running") else "❌ No"}
💬 Active Sessions: {stats.get("active_sessions", 0)}

Your Session:
👤 Contact: {session.user_name}
📱 Phone: {session.phone_number}
💭 Messages: {len(session.messages)}
🕐 Last Activity: {session.last_activity.strftime("%H:%M:%S")}"""
        await self._send_message(to=session.phone_number, text=status_text)

    async def _handle_clear(self, session: WhatsAppSession, text: str) -> None:
        """Handle /clear command."""
        session.messages.clear()
        await self._send_message(to=session.phone_number, text="🗑️ Conversation history cleared!")


# Global handler instance
whatsapp_bot_handler = WhatsAppBotHandler()


async def start_whatsapp_bot(jarvis_server: Optional["JarvisServer"] = None) -> bool:
    """Convenience function to start the WhatsApp bot."""
    return await whatsapp_bot_handler.start(jarvis_server)


async def stop_whatsapp_bot() -> None:
    """Convenience function to stop the WhatsApp bot."""
    await whatsapp_bot_handler.stop()
