"""WebSocket server for JARVIS TUI client communication"""

import json
import logging
import os
import sys
from typing import Set

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from core.discord_bot import discord_bot_handler
from core.telegram_bot import telegram_bot_handler
from core.whatsapp_bailey_client import whatsapp_bailey_client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.logging import RichHandler

from jarvis.server import JarvisServer


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[SERVER] Starting up...")

    import os
    import asyncio
    from core.config import Config

    config = Config()

    if config.llm_backend == "ollama":
        log.info("[OLLAMA] Ollama backend selected, checking availability...")
        try:
            from core.llm import get_vision_client, get_fast_client
            from core.llm.ollama import OllamaClient

            ollama_url = config.get("ollama.api_url", "http://localhost:11434")
            vision_model = config.get("ollama.vision_model")
            fast_model = config.get("ollama.fast_model")
            primary_model = config.get("ollama.primary_model")

            test_client = OllamaClient(model=vision_model or "llama2")
            ollama_available = await test_client.health_check()

            if ollama_available:
                log.info("[OLLAMA] Ollama detected, preloading models...")

                models_to_preload = []
                if primary_model:
                    models_to_preload.append(primary_model)
                if fast_model and fast_model != primary_model:
                    models_to_preload.append(fast_model)
                if vision_model and vision_model != primary_model:
                    models_to_preload.append(vision_model)

                models_to_preload = list(set(models_to_preload))

                for model in models_to_preload:
                    log.info(f"[OLLAMA] Preloading model: {model}")
                    try:
                        preload_client = OllamaClient(model=model, base_url=ollama_url)
                        success = await preload_client.preload_model()
                        if success:
                            log.info(f"[OLLAMA] Successfully preloaded: {model}")
                        else:
                            log.warning(f"[OLLAMA] Failed to preload: {model}")
                    except Exception as e:
                        log.warning(f"[OLLAMA] Error preloading {model}: {e}")
            else:
                log.warning("[OLLAMA] Not available at startup, will load on demand")
        except Exception as e:
            log.warning(f"[OLLAMA] Preload setup error: {e}")
    elif config.llm_backend == "omniroute":
        log.info("[OMNIROUTE] Omniroute backend selected, checking availability...")
        try:
            from core.llm.openai_compat import OpenAICompatClient

            api_url = config.get("omniroute.api_url", "http://localhost:3000/v1")
            api_key = config.get("omniroute.api_key", "none")
            model = config.get("omniroute.primary_model", "auto")

            omniroute_client = OpenAICompatClient(base_url=api_url, model=model, api_key=api_key)
            available = await omniroute_client.health_check()
            if available:
                log.info("[OMNIROUTE] Omniroute endpoint available")
            else:
                log.warning("[OMNIROUTE] Omniroute not available at startup")
        except Exception as e:
            log.warning(f"[OMNIROUTE] Initialization error: {e}")

    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        try:
            from core.telegram_bot import telegram_bot_handler

            async def restart_server():
                log.warning("[SERVER] Restart requested via Telegram")
                import sys
                sys.exit(42)

            telegram_bot_handler.set_restart_callback(restart_server)

            success = await telegram_bot_handler.start(jarvis_server)
            if success:
                log.info("[TELEGRAM] Bot started automatically with server")
            else:
                log.warning("[TELEGRAM] Bot failed to start (check TELEGRAM_BOT_TOKEN)")
        except Exception as e:
            log.error(f"[TELEGRAM] Error starting bot: {e}")
    else:
        log.info("[TELEGRAM] Skipping bot start (TELEGRAM_BOT_TOKEN not set)")

    if os.environ.get("DISCORD_BOT_TOKEN"):
        try:
            from core.discord_bot import discord_bot_handler

            success = await discord_bot_handler.start(jarvis_server)
            if success:
                log.info("[DISCORD] Bot started automatically with server")
            else:
                log.warning("[DISCORD] Bot failed to start (check DISCORD_BOT_TOKEN)")
        except Exception as e:
            log.error(f"[DISCORD] Error starting bot: {e}")
    else:
        log.info("[DISCORD] Skipping bot start (DISCORD_BOT_TOKEN not set)")

    whatsapp_bailey_client.jarvis = jarvis_server
    log.info("[WHATSAPP] Client initialized. Run 'node services/whatsapp-bailey/server.js' to connect.")

    try:
        log.info("[MCP] Initializing MCP servers...")
        mcp_results = await jarvis_server.tools.initialize_mcp()
        if mcp_results:
            connected = sum(1 for success in mcp_results.values() if success)
            total = len(mcp_results)
            log.info(f"[MCP] Initialized {connected}/{total} MCP servers")
            for server, success in mcp_results.items():
                status = "connected" if success else "failed"
                log.info(f"[MCP]   - {server}: {status}")
        else:
            log.warning("[MCP] No MCP servers configured or MCP disabled")
    except Exception as e:
        log.error(f"[MCP] Error initializing MCP: {e}")

    await start_proactive_monitors(config, broadcast)

    yield

    global _proactive_monitor

    log.info("[SERVER] Shutting down...")
    if telegram_bot_handler.is_running():
        await telegram_bot_handler.stop()
        log.info("[TELEGRAM] Bot stopped")
    if discord_bot_handler.is_running():
        await discord_bot_handler.stop()
        log.info("[DISCORD] Bot stopped")
    await whatsapp_bailey_client.close()

    if _proactive_monitor:
        _proactive_monitor.stop()
        log.info("[PROACTIVE_MONITOR] Stopped")

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


app = FastAPI(title="JARVIS Server", lifespan=lifespan)

allowed_cors_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():  # type: ignore[reportRedeclaration]
    """Health check endpoint for production monitoring"""
    from core.security.rate_limit import rate_limiter_manager

    return {
        "status": "healthy",
        "services": {
            "telegram": (telegram_bot_handler.is_running() if telegram_bot_handler else False),
            "discord": (discord_bot_handler.is_running() if discord_bot_handler else False),
            "whatsapp": (whatsapp_bailey_client.is_running() if whatsapp_bailey_client else False),  # type: ignore[attr-defined]
        },
        "rate_limiting": rate_limiter_manager.get_limiters_status(),
    }


# Global server instance
jarvis_server = JarvisServer()
connected_clients: Set[WebSocket] = set()


async def broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients"""
    # print(f"Broadcasting {message.get('type', 'unknown')} to {len(connected_clients)} clients")
    if len(connected_clients) > 0:
        log.info(
            f"[WEBSOCKET] Broadcasting {message.get('type', 'unknown')} to {len(connected_clients)} clients"
        )
    for client in connected_clients.copy():  # Copy to avoid modification during iteration
        try:
            await client.send_json(message)
        except Exception as e:
            log.error(f"[WEBSOCKET] Failed to send to client: {e}")
            connected_clients.discard(client)


async def broadcast_to_websockets(message: dict):
    """Alias for broadcast - can be used by external modules to send to WebSocket clients"""
    await broadcast(message)


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


@app.get("/health")  # type: ignore[reportRedeclaration]
async def health_check():
    whatsapp_status = await whatsapp_bailey_client.check_status()
    return {
        "status": "healthy",
        "clients": len(connected_clients),
        "telegram_bot": "running" if telegram_bot_handler.is_running() else "stopped",
        "discord_bot": "running" if discord_bot_handler.is_running() else "stopped",
        "whatsapp_bot": ("connected" if whatsapp_status.get("connected") else "disconnected"),
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
    return await whatsapp_bailey_client.check_status()


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


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(data: dict):
    await whatsapp_bailey_client.handle_incoming_webhook(data)
    return {"received": True}


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


@app.get("/whatsapp/qr")
async def whatsapp_qr():
    return await whatsapp_bailey_client.get_qr_code()


@app.post("/whatsapp/disconnect")
async def whatsapp_disconnect():
    return await whatsapp_bailey_client.disconnect()


_proactive_monitor = None


async def start_proactive_monitors(config, broadcast_func):
    global _proactive_monitor

    from core.proactive import ProactiveMonitor

    try:
        log.info("[PROACTIVE_MONITOR] Starting...")
        _proactive_monitor = ProactiveMonitor()
        _proactive_monitor.setup_standard_monitors(
            calendar_interval_minutes=15,
            email_interval_minutes=30,
            deadline_interval_minutes=60,
            morning_briefing_hour=8,
        )
        _proactive_monitor.start()
        log.info("[PROACTIVE_MONITOR] Started (calendar, email, deadlines)")
    except Exception as e:
        log.error(f"[PROACTIVE_MONITOR] Failed to start: {e}")


def run_server(
    host: str = "localhost",
    port: int = 8000,
    debug: bool = True,
    enable_telegram: bool = True,
):
    """Run the WebSocket server"""
    import uvicorn

    # Setup logging
    setup_logging(debug=debug)

    log.info(f"[WEBSOCKET] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=not args.no_debug)
