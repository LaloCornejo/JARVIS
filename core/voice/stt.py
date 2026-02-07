from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

import numpy as np

log = logging.getLogger("jarvis.stt")


class CircularAudioBuffer:
    """Circular buffer optimized for streaming audio processing"""

    def __init__(self, max_samples: int, dtype=None):
        if dtype is None:
            dtype = np.float32

        self.max_samples = max_samples
        self.dtype = dtype
        self.buffer = np.zeros(max_samples, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self._lock = asyncio.Lock()

    async def async_write(self, data: np.ndarray):
        """Asynchronous write to buffer"""
        async with self._lock:
            self.write(data)

    def write(self, data: np.ndarray):
        """Write data to circular buffer"""
        data_len = len(data)

        if data_len > self.max_samples:
            # If data is larger than buffer, only keep the most recent samples
            data = data[-self.max_samples :]
            data_len = self.max_samples

        # Calculate available space
        available = self.max_samples - self.size

        if data_len > available:
            # Need to overwrite old data
            overwrite = data_len - available
            self.read_pos = (self.read_pos + overwrite) % self.max_samples
            self.size = self.max_samples
        else:
            self.size += data_len

        # Write data
        end_pos = (self.write_pos + data_len) % self.max_samples

        if end_pos > self.write_pos:
            # Single contiguous write
            self.buffer[self.write_pos : end_pos] = data
        else:
            # Wrap around write
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos :] = data[:first_part]
            self.buffer[:end_pos] = data[first_part:]

        self.write_pos = end_pos

    def read(self, samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Read data from circular buffer"""
        if self.size == 0:
            return None

        if samples is None:
            samples = self.size
        else:
            samples = min(samples, self.size)

        result = np.zeros(samples, dtype=self.dtype)

        # Calculate read positions
        end_pos = (self.read_pos + samples) % self.max_samples

        if end_pos > self.read_pos:
            # Single contiguous read
            result[:] = self.buffer[self.read_pos : end_pos]
        else:
            # Wrap around read
            first_part = self.max_samples - self.read_pos
            result[:first_part] = self.buffer[self.read_pos :]
            result[first_part:] = self.buffer[:end_pos]

        self.read_pos = end_pos
        self.size -= samples

        return result

    def peek(self, samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Peek at data without removing it"""
        if self.size == 0:
            return None

        if samples is None:
            samples = self.size
        else:
            samples = min(samples, self.size)

        result = np.zeros(samples, dtype=self.dtype)

        # Calculate peek positions (same as read but don't update positions)
        end_pos = (self.read_pos + samples) % self.max_samples

        if end_pos > self.read_pos:
            result[:] = self.buffer[self.read_pos : end_pos]
        else:
            first_part = self.max_samples - self.read_pos
            result[:first_part] = self.buffer[self.read_pos :]
            result[first_part:] = self.buffer[:end_pos]

        return result

    def clear(self):
        """Clear the buffer"""
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.buffer.fill(0)

    def available(self) -> int:
        """Get number of available samples"""
        return self.size

    def space_available(self) -> int:
        """Get available space for writing"""
        return self.max_samples - self.size


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
                "Loading Whisper model: %s on %s (%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info("Whisper model loaded on %s", self.device)
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg issue, loading model with basic config")
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
        log.info("Starting STT transcription for %d samples", len(audio))
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
            log.info(
                "Calling faster_whisper transcribe with beam_size=5, vad_filter=False (disabled for debugging)"
            )
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=False,  # Disable VAD filter temporarily
            )
            segments_list = list(segments)
            log.info("Converted to list, got %d segments", len(segments_list))
            text_parts = []
            for segment in segments_list:
                text_parts.append(segment.text)
        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                self._reload_on_cpu()
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    vad_filter=False,  # Disable VAD filter
                )
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
            elif "FFmpeg" in str(e):
                log.warning("FFmpeg issue during transcription, using reduced features")
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
                log.warning("FFmpeg issue during transcription, using reduced features")
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
                vad_filter=False,  # Disable VAD filter
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
                    vad_filter=False,  # Disable VAD filter
                )
                for segment in segments:
                    yield segment.text.strip()
            elif "FFmpeg" in str(e):
                log.warning("FFmpeg issue during streaming transcription, using reduced features")
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
                log.warning("FFmpeg issue during streaming transcription, using reduced features")
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
                    log.warning("FFmpeg issue during health check, attempting basic model load")
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


class OptimizedSpeechToText:
    """Optimized STT with model caching and async preloading for maximum performance"""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
        preload: bool = True,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._model_loaded = False
        self._model_lock = asyncio.Lock()

        # Start async preloading if requested
        if preload:
            asyncio.create_task(self._preload_model_async())

    async def _preload_model_async(self) -> None:
        """Preload model in background to avoid blocking transcription"""
        try:
            await self._ensure_model_loaded()
            log.info("STT model preloaded successfully")
        except Exception as e:
            log.warning(f"Failed to preload STT model: {e}")

    async def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded, loading asynchronously if needed"""
        if self._model_loaded and self._model is not None:
            return

        async with self._model_lock:
            if self._model_loaded and self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel

                log.info(
                    "Loading Whisper model: %s on %s (%s)",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )

                # Load model in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    ),
                )

                self._model_loaded = True
                log.info("Whisper model loaded on %s", self.device)

            except Exception as e:
                if "FFmpeg" in str(e):
                    log.warning("FFmpeg issue, loading model with basic config")
                    # Try with minimal configuration on CPU
                    self.device = "cpu"
                    self.compute_type = "int8"

                    self._model = await loop.run_in_executor(
                        None,
                        lambda: WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                        ),
                    )

                    self._model_loaded = True
                    log.info("Whisper model loaded on CPU (int8) with basic configuration")
                else:
                    log.warning("Failed to load on %s: %s, falling back to CPU", self.device, e)
                    self.device = "cpu"
                    self.compute_type = "int8"

                    self._model = await loop.run_in_executor(
                        None,
                        lambda: WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                        ),
                    )

                    self._model_loaded = True
                    log.info("Whisper model loaded on CPU (int8)")

    async def _reload_on_cpu_async(self) -> None:
        """Reload model on CPU asynchronously due to CUDA error"""
        async with self._model_lock:
            from faster_whisper import WhisperModel

            log.warning("Reloading Whisper model on CPU due to CUDA error")
            self.device = "cpu"
            self.compute_type = "int8"

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                ),
            )

            log.info("Whisper model reloaded on CPU (int8)")

    async def transcribe_async(
        self,
        audio: np.ndarray,
        language: str | None = "en",
    ) -> str:
        """Async transcription with zero-copy audio processing and model caching"""
        # Ensure model is loaded
        await self._ensure_model_loaded()

        log.debug("Transcribing %d samples", len(audio))

        # Optimize audio format conversion with in-place operations when possible
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32, copy=False) / 32768.0
            else:
                audio = audio.astype(np.float32, copy=False)

        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=3,  # Reduced for speed
                    vad_filter=True,
                    best_of=1,  # Single best result
                    temperature=0.0,  # Deterministic output
                ),
            )

            text_parts = [segment.text for segment in segments]
            result = " ".join(text_parts).strip()

        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                await self._reload_on_cpu_async()
                # Retry with reloaded model
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        audio,
                        language=language,
                        beam_size=3,
                        vad_filter=True,
                        best_of=1,
                        temperature=0.0,
                    ),
                )
                text_parts = [segment.text for segment in segments]
                result = " ".join(text_parts).strip()

            elif "FFmpeg" in str(e):
                log.warning("FFmpeg issue during transcription, using reduced features")
                # Try with simplified parameters
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        audio,
                        language=language,
                        beam_size=1,  # Reduced beam size
                        vad_filter=False,  # Disable VAD filter
                    ),
                )
                text_parts = [segment.text for segment in segments]
                result = " ".join(text_parts).strip()
            else:
                raise
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg issue during transcription, using reduced features")
                # Try with simplified parameters
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        audio,
                        language=language,
                        beam_size=1,  # Reduced beam size
                        vad_filter=False,  # Disable VAD filter
                    ),
                )
                text_parts = [segment.text for segment in segments]
                result = " ".join(text_parts).strip()
            else:
                raise

        log.info("Transcribed: %s", result if result else "(empty)")
        return result

    async def transcribe_stream_async(
        self,
        audio: np.ndarray,
        language: str | None = "en",
    ):
        """Async streaming transcription with optimized parameters"""
        await self._ensure_model_loaded()

        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32, copy=False) / 32768.0
            else:
                audio = audio.astype(np.float32, copy=False)

        try:
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=3,
                    vad_filter=True,
                    best_of=1,
                    temperature=0.0,
                ),
            )

            for segment in segments:
                yield segment.text.strip()

        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                await self._reload_on_cpu_async()
                # Retry with reloaded model
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        audio,
                        language=language,
                        beam_size=3,
                        vad_filter=True,
                        best_of=1,
                        temperature=0.0,
                    ),
                )

                for segment in segments:
                    yield segment.text.strip()

    async def health_check_async(self) -> bool:
        """Async health check with model caching awareness"""
        try:
            await self._ensure_model_loaded()
            return self._model is not None
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg issue during health check, attempting basic model load")
                try:
                    async with self._model_lock:
                        self.device = "cpu"
                        self.compute_type = "int8"
                        from faster_whisper import WhisperModel

                        loop = asyncio.get_event_loop()
                        self._model = await loop.run_in_executor(
                            None,
                            lambda: WhisperModel(
                                self.model_size,
                                device="cpu",
                                compute_type="int8",
                            ),
                        )
                        self._model_loaded = True
                    return True
                except Exception:
                    return False
            else:
                return False

    # Backward compatibility methods
    def transcribe(self, audio: np.ndarray, language: str | None = "en") -> str:
        """Synchronous wrapper for backward compatibility"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, we need to handle differently
                # This is a fallback for synchronous usage
                if not self._model_loaded:
                    self._load_model_sync()
                return self._transcribe_sync(audio, language)
            else:
                # Create new event loop
                return loop.run_until_complete(self.transcribe_async(audio, language))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.transcribe_async(audio, language))

    def _load_model_sync(self) -> None:
        """Synchronous model loading for backward compatibility"""
        if self._model_loaded:
            return

        try:
            from faster_whisper import WhisperModel

            log.info(
                "Loading Whisper model: %s on %s (%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_loaded = True
            log.info("Whisper model loaded on %s", self.device)
        except Exception as e:
            if "FFmpeg" in str(e):
                log.warning("FFmpeg issue, loading model with basic config")
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._model_loaded = True
            else:
                log.warning("Failed to load on %s: %s, falling back to CPU", self.device, e)
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._model_loaded = True

    def _transcribe_sync(self, audio: np.ndarray, language: str | None = "en") -> str:
        """Synchronous transcription implementation"""
        if not self._model_loaded or self._model is None:
            self._load_model_sync()

        log.debug("Transcribing %d samples", len(audio))
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32, copy=False) / 32768.0
            else:
                audio = audio.astype(np.float32, copy=False)

        try:
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=3,
                vad_filter=True,
                best_of=1,
                temperature=0.0,
            )
            text_parts = [segment.text for segment in segments]
            result = " ".join(text_parts).strip()
        except RuntimeError as e:
            if "cuBLAS" in str(e) or "CUDA" in str(e):
                self._reload_on_cpu_sync()
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=3,
                    vad_filter=True,
                    best_of=1,
                    temperature=0.0,
                )
                text_parts = [segment.text for segment in segments]
                result = " ".join(text_parts).strip()
            else:
                raise

        return result

    def _reload_on_cpu_sync(self) -> None:
        """Synchronous CPU reload"""
        from faster_whisper import WhisperModel

        log.warning("Reloading Whisper model on CPU due to CUDA error")
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

    def health_check(self) -> bool:
        """Synchronous health check for backward compatibility"""
        try:
            if not asyncio.get_event_loop().is_running():
                return asyncio.run(self.health_check_async())
            else:
                # Event loop is running, check synchronously
                if not self._model_loaded:
                    self._load_model_sync()
                return self._model is not None
        except RuntimeError:
            # No event loop
            if not self._model_loaded:
                try:
                    self._load_model_sync()
                except Exception:
                    return False
            return self._model is not None


class StreamingSpeechToText(OptimizedSpeechToText):
    """Real-time streaming STT with overlapping windows and adaptive processing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Streaming-specific configuration
        self._streaming_buffer = CircularAudioBuffer(max_samples=16000 * 5)  # 5 seconds buffer
        self._overlap_samples = int(16000 * 0.5)  # 500ms overlap
        self._min_chunk_samples = int(16000 * 0.1)  # 100ms minimum chunk
        self._vad_threshold = 0.02  # Voice activity detection threshold
        self._stream_active = False
        self._processing_task = None
        self._audio_preprocessor = None
        self._chunk_accumulator = []
        self._last_transcription = ""

    async def start_streaming_transcription(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """Real-time streaming transcription with overlapping windows

        Args:
            audio_stream: Async iterator yielding audio bytes

        Yields:
            str: Partial transcriptions as they become available
        """
        self._stream_active = True
        self._chunk_accumulator = []
        self._last_transcription = ""

        log.info("Starting real-time streaming transcription")

        try:
            async for audio_bytes in audio_stream:
                if not self._stream_active:
                    break

                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                # Apply audio preprocessing if available
                if self._audio_preprocessor:
                    audio_chunk = self._audio_preprocessor.preprocess_streaming_audio(audio_chunk)

                # Add to streaming buffer
                self._streaming_buffer.write(audio_chunk)

                # Process chunk if we have enough data
                await self._process_streaming_chunk()

        except Exception as e:
            log.error(f"Streaming transcription error: {e}")
        finally:
            self._stream_active = False
            log.info("Streaming transcription stopped")

    async def _process_streaming_chunk(self):
        """Process available streaming chunks with intelligent timing"""
        # Check if we have enough data to process
        available_samples = self._streaming_buffer.available()

        if available_samples >= self._min_chunk_samples:
            # Extract chunk with overlap consideration
            chunk_samples = min(available_samples, int(16000 * 0.5))  # Process in 500ms chunks
            audio_chunk = self._streaming_buffer.peek(chunk_samples)

            if audio_chunk is not None and len(audio_chunk) > 0:
                # Check for voice activity
                if self._detect_voice_activity(audio_chunk):
                    # Accumulate chunks until we have meaningful content
                    self._chunk_accumulator.append(audio_chunk)

                    # Combine accumulated chunks
                    combined_audio = np.concatenate(self._chunk_accumulator)

                    # Process if we have enough data
                    if len(combined_audio) >= self._min_chunk_samples:
                        transcription = await self._transcribe_streaming_chunk(combined_audio)

                        if transcription and transcription != self._last_transcription:
                            self._last_transcription = transcription
                            yield transcription
                            self._chunk_accumulator = []  # Reset accumulator after
                            # successful transcription
                else:
                    # No voice activity, clear accumulator
                    self._chunk_accumulator = []

    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Advanced VAD with adaptive thresholding and noise floor estimation"""
        if len(audio_chunk) == 0:
            return False

        # Voice activity detection logic

        # Adaptive threshold based on noise floor
        noise_floor = np.percentile(np.abs(audio_chunk), 10)  # 10th percentile as noise estimate
        adaptive_threshold = max(self._vad_threshold, noise_floor * 2)

        # Check for sustained voice activity (not just spikes)
        sustained_energy = np.mean(audio_chunk**2) > (adaptive_threshold**2)

        # Additional checks for voice-like characteristics
        zero_crossings = np.sum(np.diff(np.sign(audio_chunk))) / (2 * len(audio_chunk))
        spectral_centroid = np.sum(
            np.arange(len(audio_chunk)) * np.abs(np.fft.fft(audio_chunk))[: len(audio_chunk) // 2]
        ) / np.sum(np.abs(np.fft.fft(audio_chunk))[: len(audio_chunk) // 2])

        # Voice typically has moderate zero crossings and spectral centroid
        voice_like = 0.1 < zero_crossings < 0.5 and spectral_centroid > len(audio_chunk) * 0.1

        return sustained_energy and voice_like

    async def _transcribe_streaming_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a streaming audio chunk with optimized parameters"""
        if not self._model_loaded or self._model is None:
            await self._ensure_model_loaded()

        try:
            # Use optimized parameters for streaming
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_chunk,
                    language="en",
                    beam_size=1,  # Reduced for speed
                    best_of=1,  # Single best result
                    temperature=0.0,  # Deterministic
                    vad_filter=False,  # Already filtered
                    condition_on_previous_text=True,  # Use context
                    initial_prompt=self._last_transcription[:100]
                    if self._last_transcription
                    else None,
                ),
            )

            text_parts = [segment.text for segment in segments]
            result = " ".join(text_parts).strip()

            return result

        except Exception as e:
            log.warning(f"Streaming transcription chunk failed: {e}")
            return ""

    def set_audio_preprocessor(self, preprocessor):
        """Set audio preprocessor for streaming pipeline"""
        self._audio_preprocessor = preprocessor

    def stop_streaming(self):
        """Stop streaming transcription"""
        self._stream_active = False
        self._chunk_accumulator = []
        self._last_transcription = ""

    async def transcribe_streaming_file(
        self, audio_file_path: str, chunk_duration_ms: int = 100
    ) -> AsyncIterator[str]:
        """Transcribe an audio file with streaming simulation

        Args:
            audio_file_path: Path to audio file
            chunk_duration_ms: Duration of each streaming chunk in milliseconds

        Yields:
            str: Partial transcriptions
        """
        try:
            import soundfile as sf

            # Load audio file
            audio_data, sample_rate = sf.read(audio_file_path, dtype="float32")

            if sample_rate != 16000:
                # Resample to 16kHz if needed
                from scipy.signal import resample

                audio_data = resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                sample_rate = 16000

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Calculate chunk size
            chunk_samples = int(sample_rate * chunk_duration_ms / 1000)

            # Simulate streaming by yielding chunks
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i : i + chunk_samples]

                if len(chunk) > 0:
                    # Convert to bytes for streaming interface
                    chunk_bytes = chunk.astype(np.float32).tobytes()
                    yield chunk_bytes

                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(chunk_duration_ms / 1000)

        except ImportError:
            log.error("soundfile library required for file streaming transcription")
            raise
        except Exception as e:
            log.error(f"File streaming transcription error: {e}")
            raise
