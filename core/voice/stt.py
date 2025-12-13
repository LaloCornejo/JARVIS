from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.stt")


class SpeechToText:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            log.info(
                "Loading Whisper model: %s on %s (%s)", self.model_size, self.device, self.compute_type
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info("Whisper model loaded on %s", self.device)
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg extension issue, attempting to load model with basic configuration")
                # Try with minimal configuration
                from faster_whisper import WhisperModel
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                log.info("Whisper model loaded on CPU (int8) with basic configuration")
            else:
                log.warning("Failed to load on %s: %s, falling back to CPU", self.device, e)
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                log.info("Whisper model loaded on CPU (int8)")

    def _reload_on_cpu(self) -> None:
        from faster_whisper import WhisperModel

        log.warning("Reloading Whisper model on CPU due to CUDA error")
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )
        log.info("Whisper model reloaded on CPU (int8)")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = "en",
    ) -> str:
        if self._model is None:
            self._load_model()

        log.debug("Transcribing %d samples", len(audio))
        if audio.dtype == np.float32:
            pass
        elif audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        try:
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                self._reload_on_cpu()
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
            elif "FFmpeg" in str(e):
                log.warning("FFmpeg extension issue during transcription, attempting with reduced features")
                # Try with simplified parameters
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,  # Reduced beam size
                    vad_filter=False,  # Disable VAD filter
                )
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
            else:
                raise
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg extension issue during transcription, attempting with reduced features")
                # Try with simplified parameters
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,  # Reduced beam size
                    vad_filter=False,  # Disable VAD filter
                )
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
            else:
                raise

        result = " ".join(text_parts).strip()
        log.info("Transcribed: %s", result if result else "(empty)")
        return result

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None = "en",
    ):
        if self._model is None:
            self._load_model()

        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        try:
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            for segment in segments:
                yield segment.text.strip()
        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                self._reload_on_cpu()
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                for segment in segments:
                    yield segment.text.strip()
            elif "FFmpeg" in str(e):
                log.warning("FFmpeg extension issue during streaming transcription, attempting with reduced features")
                # Try with simplified parameters
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,  # Reduced beam size
                    vad_filter=False,  # Disable VAD filter
                )
                for segment in segments:
                    yield segment.text.strip()
            else:
                raise
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg extension issue during streaming transcription, attempting with reduced features")
                # Try with simplified parameters
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,  # Reduced beam size
                    vad_filter=False,  # Disable VAD filter
                )
                for segment in segments:
                    yield segment.text.strip()
            else:
                raise

    def health_check(self) -> bool:
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                if "FFmpeg" in str(e):
                    log.warning("FFmpeg extension issue during health check, attempting basic model load")
                    # Try to load with minimal configuration
                    try:
                        self.device = "cpu"
                        self.compute_type = "int8"
                        from faster_whisper import WhisperModel
                        self._model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                        )
                    except Exception:
                        return False
                else:
                    return False
        return self._model is not None
