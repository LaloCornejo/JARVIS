from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    DEFAULT_CONFIG_PATH = "config/settings.yaml"

    def __init__(self, config_path: str | Path | None = None):
        self._data: dict[str, Any] = {}
        if config_path:
            self.load(config_path)
        else:
            self.load(self.DEFAULT_CONFIG_PATH)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                self._data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def llm_backend(self) -> str:
        return self.get("llm.backend", "ollama")

    def _get_backend_config(self, key: str, default: Any = None) -> Any:
        backend = self.llm_backend
        return self.get(f"{backend}.{key}", default)

    @property
    def llm_url(self) -> str:
        backend = self.llm_backend
        if backend == "vllm":
            return self._get_backend_config("api_url", "http://localhost:8000/v1")
        elif backend == "lmstudio":
            return self._get_backend_config("api_url", "http://localhost:1234/v1")
        elif backend == "openrouter":
            return self._get_backend_config("api_url", "https://openrouter.ai/api/v1")
        return self._get_backend_config("api_url", "http://localhost:11434")

    @property
    def llm_api_key_env(self) -> str | None:
        return self._get_backend_config("api_key_env")

    @property
    def llm_model(self) -> str:
        return self._get_backend_config("primary_model")

    def _get_model_for_backend_type(self, model_key: str, backend_type: str) -> Any:
        backend: str | None = None
        if backend_type == "vision":
            backend = self.vision_backend or self.llm_backend
        elif backend_type == "fast":
            backend = self.fast_backend or self.llm_backend
        elif backend_type == "primary":
            backend = self.primary_backend or self.llm_backend
        else:
            backend = self.llm_backend
        return self.get(f"{backend}.{model_key}")

    @property
    def llm_vision_model(self) -> str:
        return self._get_model_for_backend_type("vision_model", "vision")

    @property
    def llm_fast_model(self) -> str:
        return self._get_model_for_backend_type("fast_model", "fast")

    @property
    def llm_primary_model(self) -> str:
        return self._get_model_for_backend_type("primary_model", "primary")

    @property
    def tts_engine(self) -> str:
        return self.get("tts.engine", "xtts")

    @property
    def tts_base_url(self) -> str:
        return self.get("tts.xtts.base_url") or self.get("tts.base_url", "http://localhost:8020")

    @property
    def tts_speaker(self) -> str:
        return self.get("tts.xtts.speaker") or self.get("tts.speaker", "duckie")

    @property
    def tts_language(self) -> str:
        return self.get("tts.language", "en")

    @property
    def tts_sample_rate(self) -> int:
        xtts_rate: int | None = self.get("tts.xtts.sample_rate")
        return xtts_rate or self.get("tts.sample_rate", 24000)

    @property
    def supertonic_voice(self) -> str:
        return self.get("tts.supertonic.voice", "M1")

    @property
    def supertonic_custom_voice_path(self) -> str | None:
        return self.get("tts.supertonic.custom_voice_path")

    @property
    def supertonic_speed(self) -> float:
        return float(self.get("tts.supertonic.speed", 1.05))

    @property
    def supertonic_steps(self) -> int:
        return int(self.get("tts.supertonic.steps", 8))

    # --- MOSS-TTS-Nano config ---

    @property
    def mosstts_nano_voice(self) -> str:
        return self.get("tts.mosstts_nano.voice", "Ava")

    @property
    def mosstts_nano_prompt_audio_path(self) -> str | None:
        return self.get("tts.mosstts_nano.prompt_audio_path")

    @property
    def mosstts_nano_audio_temperature(self) -> float:
        return float(self.get("tts.mosstts_nano.audio_temperature", 0.8))

    @property
    def mosstts_nano_audio_top_p(self) -> float:
        return float(self.get("tts.mosstts_nano.audio_top_p", 0.95))

    @property
    def mosstts_nano_audio_top_k(self) -> int:
        return int(self.get("tts.mosstts_nano.audio_top_k", 25))

    @property
    def mosstts_nano_audio_repetition_penalty(self) -> float:
        return float(self.get("tts.mosstts_nano.audio_repetition_penalty", 1.2))

    @property
    def mosstts_nano_max_new_frames(self) -> int:
        return int(self.get("tts.mosstts_nano.max_new_frames", 375))

    @property
    def mosstts_nano_voice_clone_max_text_tokens(self) -> int:
        return int(self.get("tts.mosstts_nano.voice_clone_max_text_tokens", 75))

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
    def primary_backend(self) -> str | None:
        return self.get("llm.primary_backend")

    @property
    def fast_backend(self) -> str | None:
        return self.get("llm.fast_backend")

    @property
    def vision_backend(self) -> str | None:
        return self.get("llm.vision_backend")
