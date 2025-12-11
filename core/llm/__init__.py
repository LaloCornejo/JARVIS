from .copilot import CopilotClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .router import ModelRouter, ModelSelection, classify_query

_vision_client: OllamaClient | None = None


def get_vision_client() -> OllamaClient:
    global _vision_client
    if _vision_client is None:
        _vision_client = OllamaClient(model="qwen3-vl")
    return _vision_client


__all__ = [
    "CopilotClient",
    "GeminiClient",
    "OllamaClient",
    "ModelRouter",
    "ModelSelection",
    "classify_query",
    "get_vision_client",
]
