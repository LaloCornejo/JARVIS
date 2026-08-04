"""Discord Voice WebSocket handler.

Manages the voice WebSocket connection and protocol handshake.
Discord voice protocol:
1. Connect to wss://endpoint?v=4
2. Send Identify with server_id, user_id, session_id, token
3. Receive Ready (ssrc, ip, port, modes)
4. Send UDP discovery packet
5. Receive our public IP/port
6. Send Select Protocol
7. Receive Session Description with secret_key
8. Can now send audio via UDP
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import socket
import struct
import uuid
from typing import Any, Callable, Optional

import websockets

log = logging.getLogger("jarvis.discord_voice.ws")


class VoiceWebSocket:
    """Discord Voice WebSocket connection handler."""

    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: Optional[float] = None
        self._sequence: Optional[int] = None

        # Connection info
        self.endpoint: Optional[str] = None
        self.token: Optional[str] = None
        self.server_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None

        # Voice server info (from Ready)
        self.ssrc: int = 0
        self.voice_ip: Optional[str] = None
        self.voice_port: int = 0
        self.modes: list[str] = []

        # Our public address (from IP discovery)
        self.public_ip: Optional[str] = None
        self.public_port: int = 0

        # Encryption
        self.secret_key: Optional[bytes] = None
        self.mode: str = "xsalsa20_poly1305"  # Discord's preferred mode

        # Callbacks
        self.on_ready: Optional[Callable[[], Any]] = None
        self.on_session_description: Optional[Callable[[bytes], Any]] = None
        self.on_close: Optional[Callable[[], Any]] = None

    async def connect(
        self,
        endpoint: str,
        token: str,
        server_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Connect to Discord voice WebSocket.

        Args:
            endpoint: Voice server endpoint (e.g., "us-west123.discord.media")
            token: Voice token from Gateway
            server_id: Guild ID
            user_id: Bot user ID
            session_id: Session ID from Gateway

        Returns:
            True if connected successfully
        """
        self.endpoint = endpoint
        self.token = token
        self.server_id = server_id
        self.user_id = user_id
        self.session_id = session_id

        if not session_id:
            log.error("[VOICE_WS] Cannot connect: session_id is empty or None")
            return False

        if not token:
            log.error("[VOICE_WS] Cannot connect: token is empty or None")
            return False

        ws_url = f"wss://{endpoint}?v=4"
        log.info(f"[VOICE_WS] Full endpoint URL: {ws_url}")

        log.info(
            f"[VOICE_WS] Connecting with server_id={server_id}, user_id={user_id}, "
            f"session_id={session_id[:10] if session_id else None}..., token={token[:10] if token else None}..."
        )

        try:
            log.info(f"[VOICE_WS] Connecting to {ws_url}")
            self._ws = await websockets.connect(ws_url)
            self._running = True

            hello_future = asyncio.Future()

            def on_hello():
                if not hello_future.done():
                    hello_future.set_result(True)

            self._on_hello_callback = on_hello
            asyncio.create_task(self._receive_loop())

            try:
                await asyncio.wait_for(hello_future, timeout=10.0)
                log.debug("[VOICE_WS] Received Hello, waiting 1s before Identify")
                await asyncio.sleep(1.0)
            except asyncio.TimeoutError:
                log.error("[VOICE_WS] Timeout waiting for Hello")
                return False
            except Exception as e:
                log.error(f"[VOICE_WS] Error waiting for Hello: {e}", exc_info=True)
                return False

            try:
                await self._send_identify()
            except Exception as e:
                log.error(f"[VOICE_WS] Error sending identify: {e}", exc_info=True)
                return False

            return True
        except Exception as e:
            log.error(f"[VOICE_WS] Failed to connect: {e}", exc_info=True)
        return False

    async def _send_identify(self) -> None:
        """Send voice identify payload."""
        payload = {
            "op": 0,
            "d": {
                "server_id": self.server_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "token": self.token,
            },
        }
        log.info(
            f"[VOICE_WS] Sending identify: server_id={self.server_id}, user_id={self.user_id}, session_id={self.session_id}, token={self.token[:10] if self.token else None}..."
        )
        await self._send(payload)
        log.debug("[VOICE_WS] Sent identify")

    async def _send(self, payload: dict) -> None:
        """Send a payload to the WebSocket."""
        if self._ws:
            await self._ws.send(json.dumps(payload))

    async def _send_heartbeat(self) -> None:
        """Send heartbeat payload."""
        payload = {"op": 3, "d": self._sequence}
        await self._send(payload)
        log.debug(f"[VOICE_WS] Sent heartbeat (seq: {self._sequence})")

    async def _receive_loop(self) -> None:
        """Main receive loop."""
        try:
            async for message in self._ws:
                if not self._running:
                    break
                try:
                    await self._handle_message(message)
                except Exception as e:
                    log.error(f"[VOICE_WS] Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"[VOICE_WS] Connection closed: {e}")
        except Exception as e:
            log.error(f"[VOICE_WS] Receive error: {e}")
        finally:
            self._running = False
            if self.on_close:
                try:
                    self.on_close()
                except Exception:
                    pass

    async def _handle_message(self, message: str) -> None:
        """Handle a WebSocket message."""
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            log.error("[VOICE_WS] Failed to parse JSON")
            return

        op = payload.get("op")
        data = payload.get("d", {})

        if "s" in payload:
            self._sequence = payload["s"]

        if op == 8:
            await self._handle_hello(data)
        elif op == 2:
            await self._handle_ready(data)
        elif op == 4:
            await self._handle_session_description(data)
        elif op is None:
            log.warning(f"[VOICE_WS] Received message without opcode: {message[:200]}")
        else:
            log.debug(f"[VOICE_WS] Received op {op}: {data}")

    async def _handle_hello(self, data: dict) -> None:
        """Handle Hello (opcode 8)."""
        self._heartbeat_interval = data.get("heartbeat_interval", 13750) / 1000
        log.debug(f"[VOICE_WS] Hello received, heartbeat interval: {self._heartbeat_interval}s")

        if hasattr(self, "_on_hello_callback") and self._on_hello_callback:
            try:
                self._on_hello_callback()
            except Exception:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _handle_ready(self, data: dict) -> None:
        """Handle Ready (opcode 2)."""
        self.ssrc = data.get("ssrc", 0)
        self.voice_ip = data.get("ip")
        self.voice_port = data.get("port", 0)
        self.modes = data.get("modes", [])

        log.info(
            f"[VOICE_WS] Ready received: ssrc={self.ssrc}, ip={self.voice_ip}, port={self.voice_port}"
        )
        log.debug(f"[VOICE_WS] Available modes: {self.modes}")

        # Do IP discovery
        await self._do_ip_discovery()

        # Send protocol selection
        await self._send_select_protocol()

        if self.on_ready:
            try:
                self.on_ready()
            except Exception as e:
                log.error(f"[VOICE_WS] Error in on_ready callback: {e}")

    async def _do_ip_discovery(self) -> None:
        """Perform UDP IP discovery to get our public IP/port."""
        if not self.voice_ip or not self.voice_port:
            log.error("[VOICE_WS] No voice server info for IP discovery")
            await self._fallback_ip_discovery()
            return

        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(5.0)

                packet = struct.pack(
                    "!HHII",
                    0x0000,
                    70,
                    self.ssrc,
                    0,
                )
                packet += b"\x00" * 62

                log.debug(
                    f"[VOICE_WS] IP discovery attempt {attempt + 1} to {self.voice_ip}:{self.voice_port}"
                )

                sock.sendto(packet, (self.voice_ip, self.voice_port))

                response, addr = sock.recvfrom(70)
                sock.close()

                if len(response) >= 74:
                    ip_bytes = response[8:72]
                    ip_end = ip_bytes.find(b"\x00")
                    if ip_end != -1:
                        ip_bytes = ip_bytes[:ip_end]
                    self.public_ip = ip_bytes.decode("utf-8")
                    self.public_port = struct.unpack("!H", response[72:74])[0]
                    log.info(
                        f"[VOICE_WS] IP discovery complete: {self.public_ip}:{self.public_port}"
                    )
                    return
                else:
                    log.error(f"[VOICE_WS] Invalid IP discovery response: {len(response)} bytes")

            except Exception as e:
                log.error(f"[VOICE_WS] IP discovery attempt {attempt + 1} failed: {e}")

        log.warning("[VOICE_WS] All IP discovery attempts failed, using fallback")
        await self._fallback_ip_discovery()

    async def _fallback_ip_discovery(self) -> None:
        log.info("[VOICE_WS] Using fallback IP discovery")
        try:
            stun_servers = [
                ("stun.l.google.com", 19302),
                ("stun1.l.google.com", 19302),
            ]
            public_ip = None
            public_port = None

            for stun_host, stun_port in stun_servers:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.settimeout(3.0)
                    sock.bind(("", 0))
                    local_port = sock.getsockname()[1]

                    stun_request = b"\x00\x01\x00\x00\x00\x00\x00\x00"
                    sock.sendto(stun_request, (stun_host, stun_port))

                    try:
                        response, _ = sock.recvfrom(1024)
                        if len(response) >= 20:
                            public_ip = ".".join(str(b) for b in response[4:8])
                            mapped_port = struct.unpack("!H", response[2:4])[0]
                            public_port = mapped_port if mapped_port != 0 else local_port
                            sock.close()
                            log.info(f"[VOICE_WS] STUN success: {public_ip}:{public_port}")
                            break
                    except socket.timeout:
                        sock.close()
                        continue
                except Exception as e:
                    log.debug(f"[VOICE_WS] STUN server {stun_host} failed: {e}")
                    continue

            if not public_ip:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", 0))
                local_port = sock.getsockname()[1]
                sock.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                public_ip = s.getsockname()[0]
                s.close()
                public_port = local_port

            self.public_ip = public_ip
            self.public_port = public_port
            log.info(
                f"[VOICE_WS] Fallback IP discovery complete: {self.public_ip}:{self.public_port}"
            )
        except Exception as e:
            log.error(f"[VOICE_WS] Fallback IP discovery also failed: {e}")
            self.public_ip = "127.0.0.1"
            self.public_port = random.randint(10000, 60000)

    async def _send_select_protocol(self) -> None:
        """Send Select Protocol payload."""
        if "aead_aes256_gcm_rtpsize" in self.modes:
            self.mode = "aead_aes256_gcm_rtpsize"
        elif "aead_xchacha20_poly1305_rtpsize" in self.modes:
            self.mode = "aead_xchacha20_poly1305_rtpsize"
        elif "xsalsa20_poly1305_lite" in self.modes:
            self.mode = "xsalsa20_poly1305_lite"
        elif "xsalsa20_poly1305_suffix" in self.modes:
            self.mode = "xsalsa20_poly1305_suffix"
        elif "xsalsa20_poly1305" in self.modes:
            self.mode = "xsalsa20_poly1305"
        else:
            log.warning(f"[VOICE_WS] No known mode available, using first: {self.modes}")
            self.mode = self.modes[0] if self.modes else "xsalsa20_poly1305"

        payload = {
            "op": 1,
            "d": {
                "protocol": "udp",
                "data": {
                    "address": self.public_ip or "0.0.0.0",
                    "port": self.public_port or 0,
                    "mode": self.mode,
                },
                "rtc_connection_id": str(uuid.uuid4()),
            },
        }
        log.info(f"[VOICE_WS] Select Protocol payload: {payload}")
        await self._send(payload)
        log.debug(f"[VOICE_WS] Sent Select Protocol: {self.mode}")

    async def _handle_session_description(self, data: dict) -> None:
        """Handle Session Description (opcode 4)."""
        log.info(f"[VOICE_WS] Session Description received: {data}")
        secret_key_b64 = data.get("secret_key", [])
        if secret_key_b64:
            self.secret_key = bytes(secret_key_b64)
            log.info(f"[VOICE_WS] Received secret key ({len(self.secret_key)} bytes)")
        else:
            log.error("[VOICE_WS] No secret key in session description")

        if self.on_session_description and self.secret_key:
            try:
                self.on_session_description(self.secret_key)
            except Exception as e:
                log.error(f"[VOICE_WS] Error in on_session_description callback: {e}")

        await self._send_speaking()

    async def _send_speaking(self) -> None:
        payload = {
            "op": 5,
            "d": {
                "speaking": 1,
                "delay": 0,
                "ssrc": self.ssrc,
            },
        }
        await self._send(payload)
        log.info(f"[VOICE_WS] Sent Speaking indicator (ssrc={self.ssrc})")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running and self._heartbeat_interval:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self._running:
                    await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[VOICE_WS] Heartbeat error: {e}")

    async def disconnect(self) -> None:
        """Disconnect from voice WebSocket."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self.secret_key = None
        log.info("[VOICE_WS] Disconnected")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected and has secret key."""
        return self._running and self._ws is not None and self.secret_key is not None
