from __future__ import annotations

from pathlib import Path

import yaml


class Config:
    def __init__(self, config_path: str | Path | None = None):
        self._data: dict = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                self._data = yaml.safe_load(f) or {}

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def llm_url(self) -> str:
        return self.get("llm.api_url", "http://localhost:11434")

    @property
    def llm_model(self) -> str:
        return self.get("llm.primary_model", "qwen3:1.7b")

    @property
    def llm_vision_model(self) -> str:
        return self.get("llm.vision_model", "huihui_ai/qwen3-vl-abliterated:8b-instruct ")

    def llm_fast_model(self) -> str:
        return self.get("llm.fast_model", "qwen3:1.7b")

    @property
    def tts_base_url(self) -> str:
        return self.get("tts.base_url", "http://localhost:8020")

    @property
    def tts_speaker(self) -> str:
        return self.get("tts.speaker", "duckie")

    @property
    def tts_language(self) -> str:
        return self.get("tts.language", "en")

    @property
    def tts_sample_rate(self) -> int:
        return self.get("tts.sample_rate", 24000)

    @property
    def stt_model(self) -> str:
        return self.get("voice_input.model", "base.en")

    @property
    def stt_device(self) -> str:
        return self.get("voice_input.device", "cuda")

    @property
    def wake_word(self) -> str:
        return self.get("jarvis.wake_word", "hey jarvis")

    @property
    def input_device(self) -> int | None:
        return self.get("voice_input.input_device", None)

    @property
    def llm_temperature(self) -> float:
        return self.get("llm.temperature", 0.7)

    @property
    def llm_stream(self) -> bool:
        return self.get("llm.stream", True)

    @property
    def llm_fallback_model(self) -> str:
        return self.get("llm.fallback_model", "qwen3:1.7b")

    @property
    def llm_backend(self) -> str:
        return self.get("llm.backend", "ollama")

    @property
    def gemini_model(self) -> str:
        return self.get("gemini.default_model", "gemini-2.5-flash")

    @property
    def gemini_models(self) -> list[str]:
        return self.get("gemini.models", ["gemini-2.5-flash"])

    @property
    def ollama_url(self) -> str:
        return self.get("ollama.api_url", "http://localhost:11434")

    @property
    def copilot_model(self) -> str:
        primary = self.get("llm.primary_model", "claude-sonnet-4.5")
        if primary.startswith("copilot/"):
            return primary.split("/", 1)[1]
        return "claude-sonnet-4.5"
