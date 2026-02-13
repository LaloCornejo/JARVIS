from __future__ import annotations

import asyncio
import logging
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("jarvis.discord_voice.udp")


class UDPVoiceConnection:
    """Manages UDP connection to Discord voice server.

    Handles:
    - UDP socket management
    - Voice packet encryption (XSalsa20_Poly1305)
    - Audio packet transmission
    - Connection state tracking
    """

    DISCORD_PORT = 80
    DISCORD_PORT_ALT = 443
    FRAME_DURATION = 20

    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.endpoint: Optional[str] = None
        self.token: Optional[str] = None
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.secret_key: Optional[bytes] = None
        self.ssrc: int = 0
        self.ip: Optional[str] = None
        self.port: int = 0
        self.connected = False
        self._sequence = 0
        self._timestamp = 0
        self._running = False

    async def connect(
        self,
        endpoint: str,
        token: str,
        session_id: str,
        user_id: str,
        secret_key: Optional[bytes] = None,
    ) -> bool:
        """Connect to Discord voice UDP server.

        Args:
            endpoint: Voice server endpoint (e.g., "us-west123.discord.media")
            token: Voice token from Gateway
            session_id: Session ID from Gateway
            user_id: Bot user ID
            secret_key: Secret key for encryption (received via WebSocket)

        Returns:
            True if connected successfully
        """
        self.endpoint = endpoint
        self.token = token
        self.session_id = session_id
        self.user_id = user_id
        self.secret_key = secret_key

        try:
            # Resolve endpoint IP
            self.ip = await self._resolve_ip(endpoint)
            if not self.ip:
                log.error("Failed to resolve voice endpoint: %s", endpoint)
                return False

            log.info("Connecting to voice server: %s (%s)", endpoint, self.ip)

            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setblocking(False)

            # Try to connect (UDP is connectionless, but this sets default dest)
            port = self.DISCORD_PORT
            try:
                self.socket.connect((self.ip, port))
                self.port = port
            except OSError:
                # Try alternate port
                port = self.DISCORD_PORT_ALT
                self.socket.connect((self.ip, port))
                self.port = port

            self.connected = True
            self._running = True

            log.info("UDP voice connection established to %s:%d", self.ip, self.port)
            return True

        except Exception as e:
            log.error("Failed to connect to voice server: %s", e)
            await self.disconnect()
            return False

    async def _resolve_ip(self, endpoint: str) -> Optional[str]:
        """Resolve Discord endpoint to IP address."""
        try:
            # Remove port if present
            if ":" in endpoint:
                endpoint = endpoint.split(":")[0]

            # Use asyncio to resolve
            loop = asyncio.get_event_loop()
            addr_info = await loop.run_in_executor(
                None, socket.getaddrinfo, endpoint, None, socket.AF_INET
            )

            if addr_info:
                return addr_info[0][4][0]
            return None
        except Exception as e:
            log.error("DNS resolution failed: %s", e)
            return None

    def set_secret_key(self, secret_key: bytes) -> None:
        """Set encryption secret key received from Gateway."""
        self.secret_key = secret_key
        log.debug("Secret key set (%d bytes)", len(secret_key))

    def set_ssrc(self, ssrc: int) -> None:
        """Set SSRC (synchronization source) from Gateway."""
        self.ssrc = ssrc
        log.debug("SSRC set: %d", ssrc)

    def _create_voice_packet(self, opus_frame: bytes) -> bytes:
        """Create a Discord voice packet.

        Packet format:
        - Header: [1 byte version + flags, 1 byte payload type, 2 bytes sequence,
                  4 bytes timestamp, 4 bytes SSRC] = 12 bytes
        - Encrypted Opus data
        - 4 bytes nonce (used as tag)
        """
        # Increment sequence and timestamp
        self._sequence = (self._sequence + 1) % 65536
        self._timestamp = (self._timestamp + self.FRAME_DURATION * 48) % 4294967296

        # Build header (12 bytes)
        header = struct.pack(
            "!BBHII",
            0x80,  # Version (2) + flags
            0x78,  # Payload type (120)
            self._sequence,
            self._timestamp,
            self.ssrc,
        )

        if self.secret_key:
            # Encrypt with XSalsa20_Poly1305
            try:
                import nacl.secret

                box = nacl.secret.SecretBox(self.secret_key)
                # Discord uses random nonce (first 4 bytes)
                nonce = nacl.utils.random(24)
                # Encrypt
                encrypted = box.encrypt(opus_frame, nonce)
                # Return header + encrypted data + nonce suffix (4 bytes used for tag)
                return header + encrypted.ciphertext + encrypted.nonce[:4]
            except ImportError:
                log.error("PyNaCl not installed for encryption")
                # Send unencrypted (won't work with Discord)
                return header + opus_frame
        else:
            # No encryption (won't work with Discord)
            return header + opus_frame

    async def send_audio(self, opus_frames: List[bytes]) -> None:
        """Send Opus audio frames to Discord.

        Args:
            opus_frames: List of Opus-encoded audio frames
        """
        if not self.connected or not self.socket:
            raise RuntimeError("Not connected to voice server")

        if not self.secret_key:
            log.warning("No secret key set, cannot send encrypted audio")
            return

        for frame in opus_frames:
            packet = self._create_voice_packet(frame)
            try:
                loop = asyncio.get_event_loop()
                await loop.sock_sendall(self.socket, packet)
                await asyncio.sleep(0.02)  # 20ms between packets
            except Exception as e:
                log.error("Failed to send voice packet: %s", e)
                break

    async def disconnect(self) -> None:
        """Disconnect from voice server."""
        self._running = False
        self.connected = False

        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

        self._sequence = 0
        self._timestamp = 0
        self.secret_key = None

        log.info("UDP voice connection closed")

    def is_connected(self) -> bool:
        """Check if connected to voice server."""
        return self.connected and self.socket is not None
