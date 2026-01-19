from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import openwakeword
import sounddevice as sd

from core.config import Config
from core.learning.improvement import SelfImprovement
from core.llm import ModelRouter, OllamaClient
from core.llm.copilot import CopilotClient
from core.llm.gemini import GeminiClient
from core.memory.semantic_memory import get_enhanced_memory
from core.proactive import ProactiveMonitor
from core.reasoning import get_planner, get_reasoner
from core.security.permissions import PermissionManager
from core.threading_manager import StreamManager, TaskCoordinator, ThreadingManager
from core.vision import get_vision_processor
from core.voice.stt import SpeechToText
from core.voice.tts import TextToSpeech
from core.voice.vad import VoiceActivityDetector
from core.voice.wake_word import WakeWordDetector

if TYPE_CHECKING:
    from tools import ToolRegistry

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
        websocket=None,
        on_send_websocket=None,
    ):
        log.info("VoiceAssistant __init__ starting")
        config_path = config_path or "config/settings.yaml"
        self.config = Config(config_path)
        self.debug = debug
        self.tools = tools
        self.websocket = websocket

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        log.debug("Initializing OllamaClient: %s", self.config.llm_url)
        self.ollama = OllamaClient(
            base_url=self.config.llm_url,
            model=self.config.get("llm.primary_model", "qwen3:1.7b"),
        )
        self._preload_ollama = self.config.get("ollama.preload", False)

        # Remove Gemini and Copilot clients since we're only using local models
        self.gemini = None
        self.copilot = None

        primary_backend = self.config.get("llm.backend", "ollama")
        self.router = ModelRouter(
            ollama_client=self.ollama,
            ollama_model=self.config.get("llm.primary_model", "qwen3:1.7b"),
            primary_backend=primary_backend,
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

        # Phase 1: Core Intelligence Enhancements
        self.vision_processor = None  # Initialize lazily
        self.enhanced_memory = None  # Initialize lazily
        self.reasoner = None  # Initialize lazily
        self.planner = None  # Initialize lazily

        # Concurrency management
        self.threading_manager = ThreadingManager()
        self.task_coordinator = TaskCoordinator()
        self.stream_manager = StreamManager()

        self.state = AssistantState.IDLE
        self._running = False
        self._audio_buffer: list[np.ndarray] = []
        self._sample_rate = 16000
        self._chunk_size = 512
        self._silence_frames = 0
        self._silence_threshold = int(
            0.5 * self._sample_rate / self._chunk_size
        )  # Reduced for debugging
        self._min_audio_chunks = 50
        # Use shared conversation buffer instead of local messages list
        from core.streaming_interface import conversation_buffer

        self._conversation_buffer = conversation_buffer
        self._max_messages = 25  # Reduced from 50 to decrease memory usage
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
        self._forced_model: tuple[str, str] | None = None

    async def _get_vision_processor(self):
        """Get vision processor (lazy initialization)"""
        if self.vision_processor is None:
            from core.vision import get_vision_processor

            self.vision_processor = await get_vision_processor()
        return self.vision_processor

    async def _get_enhanced_memory(self):
        """Get enhanced memory system (lazy initialization)"""
        if self.enhanced_memory is None:
            self.enhanced_memory = await get_enhanced_memory()
        return self.enhanced_memory

    async def _get_reasoner(self):
        """Get reasoning system (lazy initialization)"""
        if self.reasoner is None:
            self.reasoner = await get_reasoner()
        return self.reasoner

    async def _get_planner(self):
        """Get planning system (lazy initialization)"""
        if self.planner is None:
            self.planner = await get_planner()
        return self.planner

    def set_model(self, backend: str, model: str) -> None:
        if backend == "auto":
            self._forced_model = None
            log.info("Voice model set to auto-select")
        else:
            self._forced_model = (backend, model)
            log.info("Voice model forced to: %s/%s", backend, model)

    def set_state(self, state: AssistantState) -> None:
        old_state = self.state
        log.info("State changing: %s -> %s", old_state, state)
        self.state = state

        # Stop audio stream when transitioning out of LISTENING state
        if old_state == AssistantState.LISTENING and state != AssistantState.LISTENING:
            self._stop_audio_stream()

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

    async def _recall_relevant_memories(self, query: str, limit: int = 3):
        """Automatically recall relevant memories based on the query."""
        if not self.tools:
            return []

        try:
            result = await self.tools.execute("recall_memory", query=query, limit=limit)
            if result.success:
                memories = result.data.get("memories", [])
                # Filter out low relevance memories
                return [m for m in memories if m.get("relevance", 0) > 0.5]
        except Exception as e:
            log.warning(f"Memory recall failed: {e}")
        return []

    async def generate_response(self, text: str) -> str | None:
        log.info("generate_response called with: %s", text)

        if self.websocket:
            # Send to WebSocket server for processing
            import json

            log.info(f"Sending to WebSocket: {text[:100]}...")
            await self.websocket.send(json.dumps({"type": "user_message", "content": text}))
            if self.on_send_websocket:
                self.on_send_websocket(text)
            return None  # Response will come through WebSocket

        await self._conversation_buffer.add_message({"role": "user", "content": text})

        # Automatically recall relevant memories
        relevant_memories = await self._recall_relevant_memories(text)

        if self._forced_model:
            backend, model = self._forced_model
            log.info("Using forced model: %s/%s", backend, model)
        else:
            selection = self.router.select_model(text)
            backend, model = selection.backend, selection.model
            log.info("Model auto-selected: %s/%s", backend, model)

        # Enhanced system prompt with memory context
        system_prompt = (
            "You are JARVIS, a helpful AI assistant. Be concise and direct. "
            "You have access to many tools. ALWAYS use tools when you need to get information "
            "about the system, files, applications, or perform any actions. "
            "Do NOT say you cannot do something if there is a tool available for it."
        )

        # Add memory context if available
        if relevant_memories:
            system_prompt += "\n\nRelevant information about the user:"
            for memory in relevant_memories:
                system_prompt += f"\n- {memory['fact']} ({memory['category']})"
            system_prompt += "\n\nUse this information to personalize your responses."

        full_response = ""
        tool_calls = []
        success = True
        error_msg = None

        tool_schemas = self.tools.get_filtered_schemas(text) if self.tools else None

        try:
            log.info("Starting LLM chat with backend: %s", backend)
            # Always use Ollama since we removed Copilot and Gemini
            client = self.ollama
            client.model = model

            # Get recent messages from shared buffer for context
            recent_messages = await self._conversation_buffer.get_recent_messages(
                self._max_messages
            )

            async for chunk in client.chat(
                messages=recent_messages,
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
                await self._conversation_buffer.add_message(
                    {"role": "assistant", "content": full_response, "tool_calls": tool_calls}
                )
                tool_results = await self._process_tool_calls(tool_calls)
                log.info("Tool results: %d messages", len(tool_results))
                for result in tool_results:
                    await self._conversation_buffer.add_message(result)

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
                    # Get recent messages from shared buffer for context
                    recent_messages = await self._conversation_buffer.get_recent_messages(
                        self._max_messages
                    )

                    async for chunk in client.chat(
                        messages=recent_messages,
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

        # Check for tool calls in the response text
        additional_tool_calls = self._parse_tool_calls_from_text(full_response)
        if additional_tool_calls and self.tools:
            log.info("Found %d tool calls in response text", len(additional_tool_calls))
            tool_calls.extend(additional_tool_calls)
            # Re-process tool calls
            await self._conversation_buffer.add_message(
                {"role": "assistant", "content": full_response}
            )
            tool_results = await self._process_tool_calls(tool_calls)
            for result in tool_results:
                await self._conversation_buffer.add_message(result)
            # Generate final response
            full_response = ""
            # Get recent messages from shared buffer for context
            recent_messages = await self._conversation_buffer.get_recent_messages(
                self._max_messages
            )

            async for chunk in client.chat(
                messages=recent_messages,
                system=system_prompt,
            ):
                if self._interrupted:
                    log.info("LLM generation interrupted")
                    return None
                if msg := chunk.get("message", {}):
                    if content := msg.get("content"):
                        full_response += content
            log.info("Final response after tool calls: %d chars", len(full_response))

        await self._conversation_buffer.add_message({"role": "assistant", "content": full_response})

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

        if len(unique_calls) <= 1:
            # Single tool call - use sequential execution for simplicity
            return await self._process_tool_calls_sequential(unique_calls)

        # Multiple tool calls - try parallel execution
        return await self._process_tool_calls_parallel(unique_calls)

    async def _process_tool_calls_sequential(self, unique_calls: dict) -> list[dict]:
        """Process tool calls sequentially (fallback for single or dependent calls)"""
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

    async def _process_tool_calls_parallel(self, unique_calls: dict) -> list[dict]:
        """Process independent tool calls in parallel for maximum performance"""
        # Group tools by dependency and execution characteristics
        independent_groups = self._group_independent_tools(unique_calls)

        all_results = []

        # Process each group in parallel
        for group in independent_groups:
            if len(group) == 1:
                # Single tool in group - execute directly
                name, call = group[0]
                result = await self._execute_single_tool(call)
                all_results.append(result)
            else:
                # Multiple tools in group - execute in parallel
                log.info("Executing %d tools in parallel", len(group))

                # Create concurrent tasks for this group
                tasks = []
                for name, call in group:
                    task = asyncio.create_task(self._execute_single_tool_async(call))
                    tasks.append(task)

                # Wait for all tools in this group to complete
                group_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results (handle exceptions)
                for i, result in enumerate(group_results):
                    if isinstance(result, Exception):
                        name, call = group[i]
                        log.error("Tool %s failed with exception: %s", name, result)
                        # Create error result
                        tool_call_id = call.get("id", "")
                        result = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": str(result)}),
                        }
                    all_results.append(result)

        return all_results

    def _group_independent_tools(self, unique_calls: dict) -> list[list[tuple[str, dict]]]:
        """Group tools that can be executed independently"""
        # Define tool categories and their execution constraints
        sequential_tools = {
            # Tools that must run sequentially due to shared resources
            "system_control",
            "volume_control",
            "spotify",
            "vlc",
            "docker",
            "obsidian",
            "gmail",
            "calendar",
        }

        read_only_tools = {
            # Read-only tools that can run in parallel
            "web_search",
            "fetch_url",
            "github_list",
            "github_search",
            "screenshot",
            "clipboard_paste",
            "clipboard_history",
            "memory_recall",
            "memory_list",
            "time_current",
        }

        # Categorize tools
        sequential_group = []
        parallel_groups = []

        for name, call in unique_calls.items():
            fn = call.get("function", {})
            tool_name = fn.get("name", "")

            # Extract base tool category (remove suffixes like _tool)
            base_category = (
                tool_name.lower().split("_")[0] if "_" in tool_name else tool_name.lower()
            )

            if any(seq in base_category for seq in sequential_tools):
                # Sequential tools go in their own group
                sequential_group.append((name, call))
            else:
                # Check if it's a read-only tool that can be parallelized
                is_read_only = any(ro in base_category for ro in read_only_tools) or tool_name in [
                    "get_current_time",
                    "list_open_apps",
                    "list_memory_categories",
                ]

                if is_read_only:
                    # Add to first parallel group or create new one
                    if not parallel_groups:
                        parallel_groups.append([])
                    parallel_groups[0].append((name, call))
                else:
                    # Other tools - each gets its own parallel group for safety
                    parallel_groups.append([(name, call)])

        # Return sequential groups first, then parallel groups
        result_groups = []
        if sequential_group:
            result_groups.append(sequential_group)

        result_groups.extend(parallel_groups)
        return result_groups

    async def _execute_single_tool(self, call: dict) -> dict:
        """Execute a single tool call synchronously"""
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        tool_call_id = call.get("id", "")

        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}

        log.info("Executing tool: %s with args: %s", name, args)
        result = await self.tools.execute(name, **args)
        log.info("Tool result: %s", result)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result.data if result.success else {"error": result.error}),
        }

    async def _execute_single_tool_async(self, call: dict) -> dict:
        """Execute a single tool call asynchronously (for parallel execution)"""
        return await self._execute_single_tool(call)

    def _parse_tool_calls_from_text(self, response_text: str) -> list[dict]:
        """Parse tool calls embedded in the response text."""
        tool_calls = []
        import re

        # Pattern for quoted queries like **"Barcelona next matches 2025"**
        query_pattern = r'\*\*"([^"]+)"\*\*'
        matches = re.findall(query_pattern, response_text)
        for match in matches:
            tool_calls.append(
                {
                    "function": {"name": "web_search", "arguments": {"query": match}},
                    "id": f"parsed_{len(tool_calls)}",
                }
            )

        # Pattern for JSON tool calls in <tool_call> tags
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        json_matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
        for json_str in json_matches:
            try:
                call = json.loads(json_str.strip())
                if isinstance(call, dict) and "function" in call:
                    tool_calls.append(call)
            except json.JSONDecodeError:
                log.debug("Failed to parse tool call JSON: %s", json_str)

        return tool_calls

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

        # Cancel coordinated tasks
        async def cancel_all_tasks():
            await self.task_coordinator.cancel_task("process_and_respond")
            await self.task_coordinator.cancel_task("generate_response")
            await self.task_coordinator.cancel_task("partial_transcription")

        if self._loop:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(cancel_all_tasks()))

    async def handle_wake_word(self) -> None:
        # If already listening, just reset the buffer and continue
        if self.state == AssistantState.LISTENING:
            log.info("Wake word detected while already listening - resetting buffer")
            self._audio_buffer = []
            self._silence_frames = 0
            self._last_transcription = ""
            return

        # If in another state (like PROCESSING or SPEAKING), interrupt first
        if self.state != AssistantState.IDLE:
            log.info("Wake word detected - interrupting current operation")
            self.interrupt()
            await asyncio.sleep(0.1)

        log.info("Wake word detected! Switching to LISTENING state")
        self._interrupted = False
        self._last_transcription = ""
        self.set_state(AssistantState.LISTENING)
        self._audio_buffer = []
        self._silence_frames = 0

        # Start audio stream for speech capture
        self._start_audio_stream()

    async def handle_audio_chunk(self, audio: np.ndarray) -> None:
        if self.state != AssistantState.LISTENING:
            return

        # Add to buffer for VAD and transcription
        self._audio_buffer.append(audio)

        # Use VAD to detect speech
        is_speech = self.vad.is_speech(audio)
        if not is_speech:
            self._silence_frames += 1
            if self._silence_frames % 10 == 0:  # Log every 10 silence frames
                log.debug(
                    "Silence frames: %d/%d, buffer size: %d",
                    self._silence_frames,
                    self._silence_threshold,
                    len(self._audio_buffer),
                )
        else:
            if self._silence_frames > 0:
                log.debug(
                    "Speech detected after %d silence frames, buffer size: %d",
                    self._silence_frames,
                    len(self._audio_buffer),
                )
            self._silence_frames = 0

        # Use coordinated task management for partial transcription
        if (
            self._on_partial_transcription
            and len(self._audio_buffer) >= self._transcribe_interval
            and len(self._audio_buffer) % self._transcribe_interval == 0
        ):
            await self.task_coordinator.start_task(
                "partial_transcription", self._do_partial_transcription()
            )

        # Check for silence to trigger speech processing
        max_chunks = int(5.0 * self._sample_rate / self._chunk_size)  # 5 second max
        if (
            (
                self._silence_frames >= self._silence_threshold
                and len(self._audio_buffer) >= self._min_audio_chunks
            )
            or len(self._audio_buffer) >= max_chunks  # Timeout after 5 seconds
        ):
            log.info(
                "Triggering speech processing: %d audio chunks (%d silence frames, threshold %d)",
                len(self._audio_buffer),
                self._silence_frames,
                self._silence_threshold,
            )
            full_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

            # Use coordinated task management for main processing
            await self.task_coordinator.start_task(
                "process_and_respond", self._process_and_respond(full_audio)
            )

    async def _do_partial_transcription(self) -> None:
        if not self._audio_buffer or self.state != AssistantState.LISTENING:
            return
        try:
            audio = np.concatenate(self._audio_buffer)
            # Use threading manager for CPU-intensive STT
            # Call the transcribe method directly without passing it as a function
            text = self.stt.transcribe(audio)
            if text and text.strip() and text != self._last_transcription:
                self._last_transcription = text.strip()
                if self._on_partial_transcription:
                    # Stream the partial transcription
                    await self.stream_manager.push_to_stream(
                        "transcription_stream", self._last_transcription
                    )
                    self._on_partial_transcription(self._last_transcription)
        except Exception as e:
            log.debug("Partial transcription failed: %s", e)

    async def _process_and_respond(self, audio: np.ndarray) -> None:
        log.info("_process_and_respond called with %d samples", len(audio))
        try:
            log.info("Calling process_speech_sync...")
            # Process speech synchronously (already runs in executor)
            text = self.process_speech_sync(audio)

            log.info("Transcription result: %s", text)
            if text and not self._interrupted:
                log.info("Generating response...")

                # Stream the user input
                await self.stream_manager.push_to_stream(
                    "conversation_stream", {"role": "user", "content": text}
                )

                # Generate response directly (it's already a coroutine)
                response = await self.generate_response(text)

                # Ensure response is a string
                if response is not None and not isinstance(response, str):
                    response = str(response)

                log.info("Response generated: %s", response[:50] if response else None)
                if response and not self._interrupted:
                    log.info("Calling speak_response...")

                    # Stream the assistant response
                    await self.stream_manager.push_to_stream(
                        "conversation_stream", {"role": "assistant", "content": response}
                    )

                    # Speak response synchronously
                    self.speak_response_sync(response)
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
            # Always return to IDLE state after processing
            if self.state != AssistantState.IDLE and not self._interrupted:
                self.set_state(AssistantState.IDLE)

    def process_speech_sync(self, audio: np.ndarray) -> str | None:
        """Synchronous version of process_speech for threading"""
        try:
            log.debug("Processing audio: %d samples", len(audio))
            text = self.stt.transcribe(audio)
            if not text or len(text.strip()) < 2:
                log.debug("Transcription empty or too short")
                return None
            text = text.strip()
            log.debug("Transcribed: %s", text)
            return text
        except Exception as e:
            log.error("Speech processing error: %s", e)
            return None

    def speak_response_sync(self, text: str) -> None:
        """Synchronous version of speak_response for threading"""
        try:
            self.set_state(AssistantState.SPEAKING)
            log.info("Speaking response: %s", text[:50] + "..." if len(text) > 50 else text)
            # Run the TTS in a separate thread with its own event loop
            import asyncio
            import threading

            def run_tts():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(
                        self.tts.play_stream_interruptible(text, lambda: self._interrupted)
                    )
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_tts, daemon=True)
            thread.start()
            thread.join()
            log.info("TTS playback complete")
        except Exception as e:
            log.error("TTS error: %s", e)
        finally:
            if not self._interrupted:
                self.set_state(AssistantState.IDLE)

    def _start_audio_stream(self) -> None:
        """Start audio stream for speech capture after wake word"""
        # Stop any existing stream first
        self._stop_audio_stream()

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if self.state == AssistantState.LISTENING and self._audio_queue:
                # Convert to float32 and send to queue
                audio_chunk = indata[:, 0].astype(np.float32)
                try:
                    self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, audio_chunk)
                except Exception as e:
                    log.warning("Failed to queue audio chunk: %s", e)

        try:
            self._audio_stream = sd.InputStream(
                device=self.config.input_device,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_size,
                callback=audio_callback,
            )
            self._audio_stream.start()
            log.info("Audio stream started for speech capture")
        except Exception as e:
            log.error("Failed to start audio stream: %s", e)
            self._audio_stream = None

    def _stop_audio_stream(self) -> None:
        """Stop the current audio stream"""
        if hasattr(self, "_audio_stream") and self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
                log.debug("Audio stream stopped")
            except Exception as e:
                log.debug("Error stopping audio stream: %s", e)
            finally:
                self._audio_stream = None

    async def _process_audio_queue(self) -> None:
        while self._running:
            try:
                audio = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                await self.handle_audio_chunk(audio)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error("Error in audio queue processing: %s", e)

    def _wake_callback(self) -> None:
        """Callback for wake word detection"""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self.handle_wake_word(), self._loop)

    async def run(self) -> None:
        audio_task = None
        try:
            log.info("VoiceAssistant.run() starting")
            self._running = True
            self._loop = asyncio.get_running_loop()
            self._audio_queue = asyncio.Queue()

            async with self.threading_manager:
                if self._preload_ollama:
                    log.info("Preloading Ollama model: %s", self.ollama.model)
                    await self.ollama.preload_model()

                self.learning.start_session()
                self.proactive.setup_standard_monitors()
                self.proactive.start()

                self.vad.is_speech(np.zeros(self._chunk_size, dtype=np.float32))

                log.info("Downloading wake word models if needed")
                openwakeword.utils.download_models()

                log.info("Starting wake word detector")
                self.wake_detector = WakeWordDetector()
                self.wake_detector.start(self._wake_callback)

                log.info("Starting audio stream")
                audio_task = asyncio.create_task(self._process_audio_queue())

                while self._running:
                    await asyncio.sleep(0.1)
        except Exception as e:
            log.error("Exception in VoiceAssistant.run(): %s", e, exc_info=True)
        finally:
            self._running = False
            if audio_task:
                audio_task.cancel()
            await self.threading_manager.cancel_all_tasks()

    async def stop(self) -> None:
        self._running = False
        self.wake_word.stop()
        self.proactive.stop()
        await self.llm.close()
        await self.tts.close()
        await self.threading_manager.cancel_all_tasks()

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

    async def health_check(self) -> dict:
        results = {}
        results["tts"] = await self.tts.health_check()
        results["stt"] = self.stt.health_check()
        results["vad"] = self.vad.health_check()
        results["wake_word"] = self.wake_word.health_check()
        return results

    async def reason_about_complex_task(
        self, task_description: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use enhanced reasoning for complex multi-step tasks"""
        log.info(f"Starting enhanced reasoning for: {task_description}")

        try:
            planner = await self._get_planner()

            # Analyze task complexity
            complexity = await planner.analyze_task_complexity(task_description)

            # Create reasoning chain
            chain = await planner.plan_complex_task(
                task_description=task_description,
                task_type=complexity.get("approach", "general"),
                constraints={"max_steps": complexity.get("recommended_max_steps", 10)},
            )

            # Execute the reasoning chain
            # Note: In a full implementation, this would integrate with the tool system
            # For now, we'll return the planning results

            return {
                "success": True,
                "task_description": task_description,
                "complexity_analysis": complexity,
                "reasoning_chain": chain.to_dict(),
                "recommendations": await self._generate_reasoning_recommendations(chain),
            }

        except Exception as e:
            log.error(f"Enhanced reasoning failed: {e}")
            return {"success": False, "error": str(e), "task_description": task_description}

    async def _generate_reasoning_recommendations(self, chain) -> List[str]:
        """Generate actionable recommendations from reasoning chain"""
        recommendations = []

        for step in chain.steps:
            if step.action and step.expected_outcome:
                recommendations.append(
                    f"Execute '{step.action}' to achieve: {step.expected_outcome}"
                )

        return recommendations[:5]  # Limit to top 5 recommendations

    async def analyze_image_with_vision(
        self, image_data: Union[str, bytes], analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Use enhanced vision AI for image analysis"""
        log.info(f"Starting vision analysis: {analysis_type}")

        try:
            vision_processor = await self._get_vision_processor()

            result = await vision_processor.analyze_image(
                image=image_data, analysis_type=analysis_type
            )

            return {"success": True, "analysis_type": analysis_type, "result": result}

        except Exception as e:
            log.error(f"Vision analysis failed: {e}")
            return {"success": False, "error": str(e), "analysis_type": analysis_type}

    async def search_memories_semantically(
        self, query: str, limit: int = 5, include_context: bool = True
    ) -> Dict[str, Any]:
        """Use enhanced semantic memory search"""
        log.info(f"Starting semantic memory search: {query}")

        try:
            memory_system = await self._get_enhanced_memory()

            results = await memory_system.retrieve_relevant_memories(query=query, limit=limit)

            return {
                "success": True,
                "query": query,
                "results": results,
                "total_found": len(results.get("semantic_results", []))
                + len(results.get("contextual_results", [])),
            }

        except Exception as e:
            log.error(f"Semantic memory search failed: {e}")
            return {"success": False, "error": str(e), "query": query}

    async def get_memory_insights(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive memory insights for a topic"""
        try:
            memory_system = await self._get_enhanced_memory()
            return await memory_system.get_memory_insights(topic)
        except Exception as e:
            log.error(f"Memory insights failed: {e}")
            return {"error": str(e)}
