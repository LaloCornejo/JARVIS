from .copilot import CopilotClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .router import ModelRouter, ModelSelection, classify_query

_vision_client: OllamaClient | None = None
_fast_client: OllamaClient | None = None


def get_vision_client() -> OllamaClient:
    global _vision_client
    if _vision_client is None:
        _vision_client = OllamaClient(model="huihui_ai/qwen3-vl-abliterated:4b")
        # Preload the model asynchronously
        import asyncio

        try:
            # Try to preload in background
            asyncio.create_task(_vision_client.preload_model())
        except Exception:
            pass  # Ignore if preloading fails
    return _vision_client


def get_fast_client() -> OllamaClient:
    global _fast_client
    if _fast_client is None:
        _fast_client = OllamaClient(model="qwen3:1.7b")
        # Preload the model asynchronously
        import asyncio

        try:
            # Try to preload in background
            asyncio.create_task(_fast_client.preload_model())
        except Exception:
            pass  # Ignore if preloading fails
    return _fast_client


__all__ = [
    "CopilotClient",
    "GeminiClient",
    "OllamaClient",
    "ModelRouter",
    "ModelSelection",
    "classify_query",
    "get_vision_client",
    "get_fast_client",
]
