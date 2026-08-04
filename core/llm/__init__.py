from .copilot import CopilotClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient
from .router import ModelRouter, ModelSelection, classify_query
from core.config import Config
import os

_config = Config()

_vision_client: OllamaClient | OpenAICompatClient | None = None
_fast_client: OllamaClient | OpenAICompatClient | None = None


def _get_backend_for_model_type(model_type: str) -> str:
    if model_type == "vision":
        return _config.vision_backend or _config.llm_backend
    elif model_type == "fast":
        return _config.fast_backend or _config.llm_backend
    elif model_type == "primary":
        return _config.primary_backend or _config.llm_backend
    return _config.llm_backend


def _get_url_for_backend(backend: str) -> str:
    if backend == "vllm":
        return _config.get("vllm.api_url", "http://localhost:8000/v1")
    elif backend == "lmstudio":
        return _config.get("lmstudio.api_url", "http://localhost:1234/v1")
    elif backend == "openrouter":
        return _config.get("openrouter.api_url", "https://openrouter.ai/api/v1")
    elif backend == "gemini":
        return _config.get("gemini.api_url", "https://generativelanguage.googleapis.com/v1beta")
    elif backend == "nvidia":
        return _config.get("nvidia.api_url", "https://integrate.api.nvidia.com/v1")
    elif backend == "copilot":
        return _config.get("copilot.api_url", "https://api.githubcopilot.com")
    elif backend == "omniroute":
        return _config.get("omniroute.api_url", "http://localhost:3000/v1")
    elif backend == "llamacpp":
        return _config.get("llamacpp.api_url", "http://127.0.0.1:8080/v1")
    return _config.get("ollama.api_url", "http://localhost:11434")


def _get_api_key_for_backend(backend: str) -> str:
    if backend == "omniroute":
        return _config.get("omniroute.api_key", "none")

    api_key_env = None
    if backend == "openrouter":
        api_key_env = _config.get("openrouter.api_key_env")
    elif backend == "gemini":
        api_key_env = _config.get("gemini.api_key_env")
    elif backend == "nvidia":
        api_key_env = _config.get("nvidia.api_key_env")
    elif backend == "copilot":
        api_key_env = _config.get("copilot.api_key_env")

    if api_key_env:
        return os.environ.get(api_key_env, "none")
    return "none"


def _get_client_for_model(model: str, model_type: str):
    backend = _get_backend_for_model_type(model_type)

    if backend in ("vllm", "lmstudio", "openrouter", "nvidia", "gemini", "copilot", "omniroute", "llamacpp"):
        url = _get_url_for_backend(backend)
        api_key = _get_api_key_for_backend(backend)
        return OpenAICompatClient(base_url=url, model=model, api_key=api_key)
    return OllamaClient(model=model)


def get_vision_client() -> OllamaClient | OpenAICompatClient:
    global _vision_client
    if _vision_client is None:
        model = _config.llm_vision_model
        if not model:
            raise ValueError("vision_model not configured in config")
        _vision_client = _get_client_for_model(model, "vision")
    return _vision_client


def get_fast_client() -> OllamaClient | OpenAICompatClient:
    global _fast_client
    if _fast_client is None:
        model = _config.llm_fast_model
        if not model:
            raise ValueError("fast_model not configured in config")
        _fast_client = _get_client_for_model(model, "fast")
    return _fast_client


__all__ = [
    "CopilotClient",
    "GeminiClient",
    "OllamaClient",
    "OpenAICompatClient",
    "ModelRouter",
    "ModelSelection",
    "classify_query",
    "get_vision_client",
    "get_fast_client",
]
