from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

from core.config import Config
from core.learning.improvement import SelfImprovement
from core.llm import ModelRouter, OllamaClient
from core.llm.copilot import CopilotClient
from core.llm.gemini import GeminiClient
from core.proactive.monitor import ProactiveMonitor
from core.security.permissions import PermissionManager
from core.voice.stt import SpeechToText
from core.voice.tts import TextToSpeech
from core.voice.vad import VoiceActivityDetector
from core.voice.wake_word import WakeWordDetector

if TYPE_CHECKING:
    from tools import ToolRegistry

log = logging.getLogger("jarvis.assistant")


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class VoiceAssistant:
    def __init__(
        self,
        config_path: str | None = None,
        debug: bool = False,
        tools: ToolRegistry | None = None,
        copilot_client: CopilotClient | None = None,
        gemini_client: GeminiClient | None = None,
    ):
        config_path = config_path or "config/settings.yaml"
        self.config = Config(config_path)
        self.debug = debug
        self.tools = tools

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        log.debug("Initializing OllamaClient: %s", self.config.llm_url)
        self.ollama = OllamaClient(
            base_url=self.config.llm_url,
            model=self.config.get("ollama.vision_model", "qwen3:1.7b"),
        )
        self._preload_ollama = self.config.get("ollama.preload", False)

        if gemini_client:
            self.gemini = gemini_client
        else:
            self.gemini = None
            gemini_key = self.config.get("gemini.api_key_env")
            if gemini_key:
                import os

                api_key = os.environ.get(gemini_key, "")
                if api_key:
                    log.debug("Initializing GeminiClient")
                    self.gemini = GeminiClient(
                        api_key=api_key,
                        model=self.config.get("gemini.default_model", "gemini-2.5-flash"),
                    )

        if copilot_client:
            self.copilot = copilot_client
        else:
            log.debug("Initializing CopilotClient")
            self.copilot = CopilotClient()

        primary_backend = self.config.get("llm.backend", "ollama")
        self.router = ModelRouter(
            ollama_client=self.ollama,
            copilot_client=self.copilot,
            gemini_client=self.gemini,
            ollama_model=self.config.get("ollama.vision_model", "qwen3:1.7b"),
            gemini_model=self.config.get("gemini.default_model", "gemini-2.5-flash"),
            primary_backend=primary_backend,
            use_copilot_for_complex=True,
        )
        self.llm = self.ollama
        log.debug("Initializing TTS: %s", self.config.tts_base_url)
        self.tts = TextToSpeech(
            base_url=self.config.tts_base_url,
            speaker=self.config.tts_speaker,
        )
        log.debug(
            "Initializing STT: model=%s device=%s", self.config.stt_model, self.config.stt_device
        )
        self.stt = SpeechToText(
            model_size=self.config.stt_model,
            device=self.config.stt_device,
        )
        self.vad = VoiceActivityDetector()
        self.wake_word = WakeWordDetector(input_device=self.config.input_device)

        self.learning = SelfImprovement(data_dir / "learning.db")
        self.permissions = PermissionManager(data_dir / "permissions.db")
        self.proactive = ProactiveMonitor()

        self.state = AssistantState.IDLE
        self._running = False
        self._audio_buffer: list[np.ndarray] = []
        self._sample_rate = 16000
        self._chunk_size = 512
        self._silence_frames = 0
        self._silence_threshold = int(1.5 * self._sample_rate / self._chunk_size)
        self._min_audio_chunks = 50
        self._messages: list[dict] = []
        self._max_messages = 50
        self._on_state_change: Callable[[AssistantState], None] | None = None
        self._on_transcription: Callable[[str], None] | None = None
        self._on_partial_transcription: Callable[[str], None] | None = None
        self._on_response: Callable[[str], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: asyncio.Queue[np.ndarray] | None = None
        self._on_alert: Callable[[Any], None] | None = None
        self._current_task: asyncio.Task | None = None
        self._interrupted = False
        self._transcribe_interval = 30
        self._last_transcription = ""
        self._forced_model: tuple[str, str] | None = ("copilot", "claude-sonnet-4.5")
        log.debug("VoiceAssistant initialized")

    def set_model(self, backend: str, model: str) -> None:
        if backend == "auto":
            self._forced_model = None
            log.info("Voice model set to auto-select")
        else:
            self._forced_model = (backend, model)
            log.info("Voice model forced to: %s/%s", backend, model)

    def set_state(self, state: AssistantState) -> None:
        log.info("State changing: %s -> %s", self.state, state)
        self.state = state
        if self._on_state_change:
            log.debug("Calling state change callback")
            self._on_state_change(state)

    def on_state_change(self, callback: Callable[[AssistantState], None]) -> None:
        self._on_state_change = callback

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        self._on_transcription = callback

    def on_partial_transcription(self, callback: Callable[[str], None]) -> None:
        self._on_partial_transcription = callback

    def on_response(self, callback: Callable[[str], None]) -> None:
        self._on_response = callback

    def on_alert(self, callback: Callable[[Any], None]) -> None:
        self._on_alert = callback
        self.proactive.alerts.register_handler(callback)

    async def process_speech(self, audio: np.ndarray) -> str | None:
        self.set_state(AssistantState.PROCESSING)
        log.debug("Processing audio: %d samples", len(audio))
        text = self.stt.transcribe(audio)
        if not text or len(text.strip()) < 2:
            log.debug("Transcription empty or too short")
            return None
        text = text.strip()
        log.debug("Transcribed: %s", text)
        if self._on_transcription:
            self._on_transcription(text)
        return text

    async def generate_response(self, text: str) -> str | None:
        log.info("generate_response called with: %s", text)
        self._messages.append({"role": "user", "content": text})

        if self._forced_model:
            backend, model = self._forced_model
            log.info("Using forced model: %s/%s", backend, model)
        else:
            selection = self.router.select_model(text)
            backend, model = selection.backend, selection.model
            log.info("Model auto-selected: %s/%s", backend, model)

        system_prompt = (
            "You are JARVIS, a helpful AI assistant. Be concise and direct. "
            "You have access to many tools. ALWAYS use tools when you need to get information "
            "about the system, files, applications, or perform any actions. "
            "Do NOT say you cannot do something if there is a tool available for it."
        )
        full_response = ""
        tool_calls = []
        success = True
        error_msg = None

        tool_schemas = self.tools.get_filtered_schemas(text) if self.tools else None

        try:
            log.info("Starting LLM chat with backend: %s", backend)
            client = None
            if backend == "gemini" and self.gemini:
                client = self.gemini
            elif backend == "copilot" and self.copilot:
                client = self.copilot
            else:
                client = self.ollama
                client.model = model

            async for chunk in client.chat(
                messages=self._messages,
                system=system_prompt,
                tools=tool_schemas,
            ):
                if self._interrupted:
                    log.info("LLM generation interrupted")
                    return None
                if msg := chunk.get("message", {}):
                    if content := msg.get("content"):
                        full_response += content
                    if calls := msg.get("tool_calls"):
                        tool_calls = calls

            log.info(
                "First LLM pass done. tool_calls=%d, response_len=%d",
                len(tool_calls),
                len(full_response),
            )

            if tool_calls and self.tools:
                tool_names = [c.get("function", {}).get("name") for c in tool_calls]
                log.info("Processing %d tool calls: %s", len(tool_calls), tool_names)
                self._messages.append(
                    {"role": "assistant", "content": full_response, "tool_calls": tool_calls}
                )
                tool_results = await self._process_tool_calls(tool_calls)
                log.info("Tool results: %d messages", len(tool_results))
                self._messages.extend(tool_results)

                is_vision_tool = "screenshot_analyze" in tool_names
                if is_vision_tool and tool_results:
                    try:
                        vision_data = json.loads(tool_results[0].get("content", "{}"))
                        full_response = vision_data.get("analysis", "")
                        log.info("Using vision response directly: %d chars", len(full_response))
                    except Exception as e:
                        log.warning("Failed to extract vision response: %s", e)
                        is_vision_tool = False

                if not is_vision_tool:
                    log.info("Starting second LLM pass to summarize tool results")
                    full_response = ""
                    async for chunk in client.chat(
                        messages=self._messages,
                        system=system_prompt,
                    ):
                        if self._interrupted:
                            log.info("LLM generation interrupted")
                            return None
                        if msg := chunk.get("message", {}):
                            if content := msg.get("content"):
                                full_response += content
                    log.info("Second LLM pass done, response_len=%d", len(full_response))

            log.info("LLM chat complete, response length: %d", len(full_response))
        except asyncio.CancelledError:
            log.info("LLM generation cancelled")
            return None
        except Exception as e:
            success = False
            error_msg = str(e)
            log.error("LLM error: %s", e)
            full_response = "I encountered an error processing your request."

        self._messages.append({"role": "assistant", "content": full_response})

        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

        log.debug(
            "Response: %s",
            full_response[:100] + "..." if len(full_response) > 100 else full_response,
        )
        if self._on_response:
            self._on_response(full_response)

        self.learning.log_command(
            user_input=text,
            response=full_response,
            success=success,
            context={"error": error_msg} if error_msg else None,
        )

        self.proactive.extract_deadline_from_text(text)

        return full_response

    async def _process_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        unique_calls = {}
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            if name not in unique_calls:
                unique_calls[name] = call

        log.info(
            "Deduplicated %d tool calls to %d unique tools", len(tool_calls), len(unique_calls)
        )

        results = []
        for name, call in unique_calls.items():
            fn = call.get("function", {})
            args = fn.get("arguments", {})
            tool_call_id = call.get("id", "")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}

            log.info("Executing tool: %s with args: %s", name, args)
            result = await self.tools.execute(name, **args)
            log.info("Tool result: %s", result)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        result.data if result.success else {"error": result.error}
                    ),
                }
            )

        return results

    async def speak_response(self, text: str) -> None:
        self.set_state(AssistantState.SPEAKING)
        log.info("Speaking response: %s", text[:50] + "..." if len(text) > 50 else text)
        try:
            await self.tts.play_stream_interruptible(text, lambda: self._interrupted)
            log.info("TTS playback complete")
        except asyncio.CancelledError:
            log.info("TTS cancelled")
        except Exception as e:
            log.error("TTS error: %s", e)
        if not self._interrupted:
            self.set_state(AssistantState.IDLE)

    def interrupt(self) -> None:
        log.info("Interrupting current operation")
        self._interrupted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def handle_wake_word(self) -> None:
        if self.state != AssistantState.IDLE:
            self.interrupt()
            await asyncio.sleep(0.1)

        log.info("Wake word detected! Switching to LISTENING state")
        self._interrupted = False
        self._last_transcription = ""
        self.set_state(AssistantState.LISTENING)
        self._audio_buffer = []
        self._silence_frames = 0

    async def handle_audio_chunk(self, audio: np.ndarray) -> None:
        if self.state != AssistantState.LISTENING:
            return

        self._audio_buffer.append(audio)

        is_speech = self.vad.is_speech(audio)
        if not is_speech:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        if (
            self._on_partial_transcription
            and len(self._audio_buffer) >= self._transcribe_interval
            and len(self._audio_buffer) % self._transcribe_interval == 0
        ):
            asyncio.create_task(self._do_partial_transcription())

        if (
            self._silence_frames >= self._silence_threshold
            and len(self._audio_buffer) >= self._min_audio_chunks
        ):
            log.info("Silence detected, processing %d audio chunks", len(self._audio_buffer))
            full_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

            self._current_task = asyncio.create_task(self._process_and_respond(full_audio))

    async def _do_partial_transcription(self) -> None:
        if not self._audio_buffer or self.state != AssistantState.LISTENING:
            return
        try:
            audio = np.concatenate(self._audio_buffer)
            text = await asyncio.to_thread(self.stt.transcribe, audio)
            if text and text.strip() and text != self._last_transcription:
                self._last_transcription = text.strip()
                if self._on_partial_transcription:
                    self._on_partial_transcription(self._last_transcription)
        except Exception as e:
            log.debug("Partial transcription failed: %s", e)

    async def _process_and_respond(self, audio: np.ndarray) -> None:
        log.info("_process_and_respond called with %d samples", len(audio))
        try:
            text = await self.process_speech(audio)
            log.info("Transcription result: %s", text)
            if text and not self._interrupted:
                log.info("Generating response...")
                response = await self.generate_response(text)
                log.info("Response generated: %s", response[:50] if response else None)
                if response and not self._interrupted:
                    log.info("Calling speak_response...")
                    await self.speak_response(response)
                elif self._interrupted:
                    log.info("Response interrupted before speaking")
            else:
                log.info("No text or interrupted, setting IDLE")
                if not self._interrupted:
                    self.set_state(AssistantState.IDLE)
        except asyncio.CancelledError:
            log.info("Processing cancelled")
        except Exception as e:
            log.error("_process_and_respond error: %s", e, exc_info=True)
        finally:
            self._current_task = None
            self._audio_buffer.clear()
            self._silence_frames = 0

    def _wake_callback(self) -> None:
        log.info("Wake word callback triggered!")
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self.handle_wake_word()))

    async def _process_audio_queue(self) -> None:
        while self._running:
            try:
                audio = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                await self.handle_audio_chunk(audio)
            except asyncio.TimeoutError:
                continue

    async def run(self) -> None:
        log.info("VoiceAssistant.run() starting")
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._audio_queue = asyncio.Queue()

        if self._preload_ollama:
            log.info("Preloading Ollama model: %s", self.ollama.model)
            await self.ollama.preload_model()

        self.learning.start_session()
        self.proactive.setup_standard_monitors()
        self.proactive.start()

        self.vad.is_speech(np.zeros(self._chunk_size, dtype=np.float32))

        log.info("Starting wake word detector")
        self.wake_word.start(self._wake_callback)

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if not self._running:
                return
            audio = indata[:, 0].copy()
            if self._loop and self._audio_queue:
                self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, audio)

        audio_task = asyncio.create_task(self._process_audio_queue())

        with sd.InputStream(
            device=self.config.input_device,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=audio_callback,
        ):
            while self._running:
                await asyncio.sleep(0.1)

        audio_task.cancel()

    async def stop(self) -> None:
        self._running = False
        self.wake_word.stop()
        self.proactive.stop()
        await self.llm.close()
        await self.tts.close()

    def record_positive_feedback(self) -> None:
        self.learning.record_positive_feedback()

    def record_negative_feedback(self, reason: str | None = None) -> None:
        self.learning.record_negative_feedback(reason)

    async def check_permission(self, action: str) -> bool:
        return await self.permissions.check_permission_async(action)

    def get_pending_alerts(self) -> list:
        return self.proactive.get_pending_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        return self.proactive.acknowledge_alert(alert_id)

    def get_improvement_report(self, days: int = 7) -> dict:
        return self.learning.get_improvement_report(days)
