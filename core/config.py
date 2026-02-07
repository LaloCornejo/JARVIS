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
        backend = self.get("llm.backend", "nvidia")  # Changed default to nvidia
        if backend == "nvidia":
            return self.get("nvidia.api_url", "https://integrate.api.nvidia.com/v1")
        elif backend == "ollama":
            return self.get("ollama.api_url", "http://localhost:11434")
        else:
            return self.get("llm.api_url", "https://integrate.api.nvidia.com/v1")

    @property
    def llm_model(self) -> str:
        backend = self.get("llm.backend", "nvidia")  # Changed default to nvidia
        if backend == "nvidia":
            return self.get(
                "llm.primary_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name
        elif backend == "ollama":
            return self.get("llm.primary_model", "qwen3:1.7b")
        else:
            return self.get(
                "llm.primary_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name

    @property
    def llm_vision_model(self) -> str:
        backend = self.get("llm.backend", "nvidia")  # Changed default to nvidia
        if backend == "nvidia":
            return self.get(
                "llm.vision_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name
        elif backend == "ollama":
            return self.get("llm.vision_model", "huihui_ai/qwen3-vl-abliterated:8b-instruct ")
        else:
            return self.get(
                "llm.vision_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name

    def llm_fast_model(self) -> str:
        backend = self.get("llm.backend", "nvidia")  # Changed default to nvidia
        if backend == "nvidia":
            return self.get(
                "llm.fast_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name
        elif backend == "ollama":
            return self.get("llm.fast_model", "qwen3:1.7b")
        else:
            return self.get(
                "llm.fast_model", "moonshotai/kimi-k2.5"
            )  # Changed default to correct model name

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
