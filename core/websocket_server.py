"""WebSocket server for JARVIS TUI client communication"""

import json
import logging
import sys
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.logging import RichHandler

from jarvis.server import JarvisServer

log = logging.getLogger("jarvis.websocket")


def setup_logging(debug: bool = True) -> None:
    """Setup logging for the server"""
    level = logging.DEBUG if debug else logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reduce verbosity of HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler for terminal output
    console = Console(file=sys.stdout, width=120)
    ch = RichHandler(console=console, rich_tracebacks=True, show_time=True, show_level=True)
    ch.setLevel(level)
    root_logger.addHandler(ch)

    # File handler for debug log
    try:
        file_handler = logging.FileHandler("jarvis_server_debug.log", encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        log.info("[WEBSOCKET] Debug log file handler added")
    except Exception as e:
        log.error(f"[WEBSOCKET] Failed to setup file logging: {e}")


app = FastAPI(title="JARVIS Server")

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Startup event handler - preload TTS duckie model if TTS is online"""
    import httpx

    log.info("[WEBSOCKET] Server startup")

    # Preload TTS duckie model if TTS is online
    try:
        tts_online = await jarvis_server.tts.health_check()
        if tts_online:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{jarvis_server.tts.base_url}/preload_duckie", timeout=30.0
                    )
                    if response.status_code == 200:
                        log.info("[WEBSOCKET] TTS duckie model preloaded successfully")
                    else:
                        log.warning(
                            f"[WEBSOCKET] TTS preload returned status {response.status_code}"
                        )
            except Exception as e:
                log.error(f"[WEBSOCKET] Failed to preload TTS duckie model: {e}")
        else:
            log.info("[WEBSOCKET] TTS service offline, skipping preload")
    except Exception as e:
        log.error(f"[WEBSOCKET] Error checking TTS health: {e}")


# Global server instance
jarvis_server = JarvisServer()
connected_clients: Set[WebSocket] = set()


async def broadcast(message: dict):
    """Broadcast a message to all connected clients"""
    print(f"Broadcasting {message.get('type', 'unknown')} to {len(connected_clients)} clients")
    log.warning(
        f"[WEBSOCKET] Broadcasting {message.get('type', 'unknown')} to {len(connected_clients)} clients"
    )
    for client in connected_clients.copy():  # Copy to avoid modification during iteration
        try:
            await client.send_json(message)
        except Exception as e:
            log.error(f"[WEBSOCKET] Failed to send to client: {e}")
            connected_clients.discard(client)


async def broadcast_to_ip(message: dict, ip: str):
    """Broadcast a message to all clients with the same IP"""
    if ip in connected_clients:
        for client in connected_clients[ip].copy():  # Copy to avoid modification during iteration
            try:
                await client.send_json(message)
            except Exception as e:
                log.error(f"[WEBSOCKET] Failed to send to client {client}: {e}")
                connected_clients[ip].discard(client)
                if not connected_clients[ip]:
                    del connected_clients[ip]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections from TUI clients"""
    await websocket.accept()
    connected_clients.add(websocket)
    log.info(f"[WEBSOCKET] Client connected. Total clients: {len(connected_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "user_message":
                    user_input = message.get("content", "")
                    await jarvis_server.process_message(user_input, broadcast)
                elif message_type == "set_model":
                    model = message.get("model")
                    jarvis_server.set_model(model)
                    await websocket.send_json({"type": "model_set", "model": model})
                elif message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    log.warning(f"[WEBSOCKET] Unknown message type: {message_type}")

            except json.JSONDecodeError:
                log.error("[WEBSOCKET] Invalid JSON received")
            except Exception as e:
                log.error(f"[WEBSOCKET] Error processing message: {e}")

    except WebSocketDisconnect:
        log.info("[WEBSOCKET] Client disconnected")
    except Exception as e:
        log.error(f"[WEBSOCKET] Unexpected error: {e}")
    finally:
        connected_clients.discard(websocket)
        log.info(f"[WEBSOCKET] Client removed. Total clients: {len(connected_clients)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "clients": len(connected_clients)}


def run_server(host: str = "localhost", port: int = 6969, debug: bool = True):
    """Run the WebSocket server"""
    import uvicorn

    # Setup logging
    setup_logging(debug=debug)

    log.info(f"[WEBSOCKET] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
