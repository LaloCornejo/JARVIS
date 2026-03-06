from .copilot import CopilotClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .router import ModelRouter, ModelSelection, classify_query
from core.config import Config

_config = Config()

_vision_client: OllamaClient | None = None
_fast_client: OllamaClient | None = None


def get_vision_client() -> OllamaClient:
    global _vision_client
    if _vision_client is None:
        model = _config.llm_vision_model
        if not model:
            raise ValueError("vision_model not configured in config")
        _vision_client = OllamaClient(model=model)
        import asyncio

        try:
            asyncio.create_task(_vision_client.preload_model())
        except Exception:
            pass
    return _vision_client


def get_fast_client() -> OllamaClient:
    global _fast_client
    if _fast_client is None:
        model = _config.llm_fast_model
        if not model:
            raise ValueError("fast_model not configured in config")
        _fast_client = OllamaClient(model=model)
        import asyncio

        try:
            asyncio.create_task(_fast_client.preload_model())
        except Exception:
            pass
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
