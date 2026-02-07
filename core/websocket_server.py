"""WebSocket server for JARVIS TUI client communication"""

import asyncio
import json
import logging
import sys
from typing import Set

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.logging import RichHandler

from core.discord_bot import discord_bot_handler
from core.telegram_bot import telegram_bot_handler
from core.whatsapp_bot import whatsapp_bot_handler
from jarvis.server import JarvisServer

log = logging.getLogger("jarvis.websocket")


def setup_logging(debug: bool = False) -> None:  # Changed default to False
    """Setup logging for the server"""
    level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reduce verbose logging from libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    telegram_status = "running" if telegram_bot_handler.is_running() else "stopped"
    discord_status = "running" if discord_bot_handler.is_running() else "stopped"
    whatsapp_status = "running" if whatsapp_bot_handler.is_running() else "stopped"
    return {
        "status": "healthy",
        "clients": len(connected_clients),
        "telegram_bot": telegram_status,
        "discord_bot": discord_status,
        "whatsapp_bot": whatsapp_status,
    }


@app.get("/telegram/status")
async def telegram_status():
    """Get Telegram bot status"""
    return telegram_bot_handler.get_stats()


@app.get("/discord/status")
async def discord_status():
    """Get Discord bot status"""
    return discord_bot_handler.get_stats()


@app.get("/whatsapp/status")
async def whatsapp_status():
    """Get WhatsApp bot status"""
    return whatsapp_bot_handler.get_stats()


@app.post("/telegram/start")
async def telegram_start():
    """Start Telegram bot"""
    success = await telegram_bot_handler.start(jarvis_server)
    return {"started": success, "running": telegram_bot_handler.is_running()}


@app.post("/discord/start")
async def discord_start():
    """Start Discord bot"""
    success = await discord_bot_handler.start(jarvis_server)
    return {"started": success, "running": discord_bot_handler.is_running()}


@app.post("/whatsapp/start")
async def whatsapp_start():
    """Start WhatsApp bot"""
    success = await whatsapp_bot_handler.start(jarvis_server)
    return {"started": success, "running": whatsapp_bot_handler.is_running()}


@app.post("/telegram/stop")
async def telegram_stop():
    """Stop Telegram bot"""
    await telegram_bot_handler.stop()
    return {"running": telegram_bot_handler.is_running()}


@app.post("/discord/stop")
async def discord_stop():
    """Stop Discord bot"""
    await discord_bot_handler.stop()
    return {"running": discord_bot_handler.is_running()}


@app.post("/whatsapp/stop")
async def whatsapp_stop():
    """Stop WhatsApp bot"""
    await whatsapp_bot_handler.stop()
    return {"running": whatsapp_bot_handler.is_running()}


@app.on_event("startup")
async def startup_event():
    """Start Telegram, Discord, and WhatsApp bots on server startup."""
    log.info("[SERVER] Starting up...")
    import os

    # Start Telegram bot if configured
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        try:
            # Set restart callback before starting bot
            from core.telegram_bot import telegram_bot_handler

            async def restart_server():
                """Callback to restart the server."""
                log.warning("[SERVER] Restart requested via Telegram")
                # Signal shutdown - wrapper script should restart
                import sys

                # Exit with special code that wrapper script will detect
                sys.exit(42)  # 42 = restart code

            telegram_bot_handler.set_restart_callback(restart_server)

            success = await telegram_bot_handler.start(jarvis_server)
            if success:
                log.info("[TELEGRAM] Bot started automatically with server")
            else:
                log.warning("[TELEGRAM] Bot failed to start (check TELEGRAM_BOT_TOKEN)")
        except Exception as e:
            log.error(f"[TELEGRAM] Error starting bot: {e}")
    else:
        log.info("[TELEGRAM] Bot not started (TELEGRAM_BOT_TOKEN not set)")

    # Start Discord bot if configured
    if os.environ.get("DISCORD_BOT_TOKEN"):
        try:
            success = await discord_bot_handler.start(jarvis_server)
            if success:
                log.info("[DISCORD] Bot started automatically with server")
            else:
                log.warning("[DISCORD] Bot failed to start (check DISCORD_BOT_TOKEN)")
        except Exception as e:
            log.error(f"[DISCORD] Error starting bot: {e}")
    else:
        log.info("[DISCORD] Bot not started (DISCORD_BOT_TOKEN not set)")

    # Initialize WhatsApp bot (QR code login - no credentials needed)
    try:
        # Schedule WhatsApp bot startup asynchronously to avoid blocking startup
        async def start_whatsapp_async():
            success = await whatsapp_bot_handler.start(jarvis_server)
            if success:
                log.info("[WHATSAPP] Bot handler initialized")
            else:
                log.warning("[WHATSAPP] Bot handler failed to initialize")

        # Run in background to avoid blocking startup
        # We need to get the event loop and create task properly
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.create_task(start_whatsapp_async())
    except Exception as e:
        log.error(f"[WHATSAPP] Error initializing bot handler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop Telegram, Discord, and WhatsApp bots on server shutdown."""
    log.info("[SERVER] Shutting down...")
    if telegram_bot_handler.is_running():
        await telegram_bot_handler.stop()
        log.info("[TELEGRAM] Bot stopped")
    if discord_bot_handler.is_running():
        await discord_bot_handler.stop()
        log.info("[DISCORD] Bot stopped")
    # Stop WhatsApp bot handler
    await whatsapp_bot_handler.stop()
    log.info("[WHATSAPP] Bot handler stopped")


def run_server(
    host: str = "localhost", port: int = 8000, debug: bool = True, enable_telegram: bool = True
):
    """Run the WebSocket server"""
    import uvicorn

    # Setup logging
    setup_logging(debug=debug)

    log.info(f"[WEBSOCKET] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
