from __future__ import annotations

import threading
from typing import Any

import numpy as np


class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration: float = 0.5,
        cache_clear_interval: int = 100,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_silence_duration = min_silence_duration
        self._model: Any = None
        self._lock = threading.Lock()
        self._cache_clear_interval = cache_clear_interval
        self._call_count = 0

    def _load_model(self) -> None:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]

    def _maybe_clear_cache(self) -> None:
        self._call_count += 1
        if self._call_count >= self._cache_clear_interval:
            self._call_count = 0
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_speech(self, audio: np.ndarray) -> bool:
        import torch

        with self._lock:
            if self._model is None:
                self._load_model()

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            tensor = torch.from_numpy(audio)
            confidence = self._model(tensor, self.sample_rate).item()
            self._maybe_clear_cache()
            return confidence > self.threshold

    def get_speech_segments(
        self,
        audio: np.ndarray,
    ) -> list[tuple[int, int]]:
        import torch

        with self._lock:
            if self._model is None:
                self._load_model()

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            tensor = torch.from_numpy(audio)
            timestamps = self._get_speech_timestamps(
                tensor,
                self._model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
            )
            self._maybe_clear_cache()
            return [(ts["start"], ts["end"]) for ts in timestamps]
