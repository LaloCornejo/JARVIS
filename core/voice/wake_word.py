from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.wake")


class WakeWordDetector:
    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
        input_device: int | None = None,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.input_device = input_device
        self._model: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._callback: Callable[[], None] | None = None

    def _load_model(self) -> None:
        from openwakeword.model import Model

        if self.model_path:
            self._model = Model(wakeword_models=[self.model_path], inference_framework="onnx")
        else:
            self._model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    def start(self, callback: Callable[[], None]) -> None:
        if self._running:
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Wake word detector started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_loop(self) -> None:
        import sounddevice as sd

        log.info("Wake word loop starting, loading model...")
        if self._model is None:
            self._load_model()
        log.info("Wake word model loaded, starting audio stream...")

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if not self._running:
                return
            audio_chunk = indata[:, 0]
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            prediction = self._model.predict(audio_int16)
            for score in prediction.values():
                if score > self.threshold:
                    log.info("Wake word detected! Score: %s", score)
                    if self._callback:
                        self._callback()
                    self._model.reset()
                    break

        with sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        ):
            import time

            while self._running:
                time.sleep(0.1)

    def detect_once(self, audio: np.ndarray) -> bool:
        if self._model is None:
            self._load_model()
        audio_int16 = (audio * 32767).astype(np.int16) if audio.dtype == np.float32 else audio
        prediction = self._model.predict(audio_int16)
        for score in prediction.values():
            if score > self.threshold:
                self._model.reset()
                return True
        return False
