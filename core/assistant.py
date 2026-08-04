from __future__ import annotations

import asyncio
import datetime
import json
import logging
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import openwakeword
import sounddevice as sd

from agents.orchestrator.advanced import AgentOrchestrator
from core.automation.triggers import TimeTrigger, TriggerManager
from core.config import Config
from core.context_compactor import TokenCompactor
from core.learning.improvement import SelfImprovement
from core.llm import ModelRouter, OllamaClient
from core.llm.copilot import CopilotClient
from core.llm.gemini import GeminiClient
from core.memory.episodic import EpisodicMemory
from core.memory.semantic_memory import get_enhanced_memory
from core.multi_user.user_manager import UserManager
from core.multi_user.voice_recognition import VoiceRecognition
from core.orchestrator import IntentRouter
from core.prediction.anomaly_detector import AnomalyDetector
from core.prediction.pattern_analyzer import PatternAnalyzer
from core.prediction.suggestion_engine import SmartSuggestionEngine
from core.proactive import ProactiveMonitor
from core.reasoning import get_planner, get_reasoner
from core.security.permissions import PermissionManager
from core.smart_context import get_context_manager, get_smart_context
from core.threading_manager import StreamManager, TaskCoordinator, ThreadingManager
from core.voice.stt import SpeechToText
from core.voice.tts import TextToSpeech, create_tts_backend
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
        ollama_model = self.config.get("ollama.primary_model")
        if not ollama_model:
            raise ValueError("ollama.primary_model must be configured in settings.yaml")
        self.ollama = OllamaClient(
            base_url=self.config.llm_url,
            model=ollama_model,
        )
        self._preload_ollama = self.config.get("ollama.preload", False)

        self.gemini = None
        self.copilot = None

        primary_backend = self.config.get("llm.backend", "ollama")
        copilot_model = self.config.get("copilot.primary_model")
        gemini_model = self.config.get("gemini.primary_model")

        omniroute_client = None
        omniroute_model = self.config.get("omniroute.primary_model")
        omniroute_url = self.config.get("omniroute.api_url")
        if omniroute_url:
            from core.llm import OpenAICompatClient
            omniroute_client = OpenAICompatClient(
                base_url=omniroute_url,
                model=omniroute_model or "auto",
            )

        self.router = ModelRouter(
            ollama_client=self.ollama,
            ollama_model=ollama_model,
            copilot_model=copilot_model,
            gemini_model=gemini_model,
            omniroute_client=omniroute_client,
            omniroute_model=omniroute_model,
            primary_backend=primary_backend,
        )
        self.llm = self.ollama
        log.debug("Initializing TTS: %s", self.config.tts_base_url)
        backend = create_tts_backend(self.config)
        self.tts = TextToSpeech(backend=backend)
        self.intent_router = IntentRouter()
        self.token_compactor = TokenCompactor()
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
        self._on_send_websocket = on_send_websocket

        # Phase 1: Core Intelligence Enhancements
        self.vision_processor = None  # Initialize lazily
        self.enhanced_memory = None  # Initialize lazily
        self.reasoner = None  # Initialize lazily
        self.planner = None  # Initialize lazily

        # Phase 6: New System Integration
        self.agent_orchestrator: AgentOrchestrator | None = None
        self.user_manager: UserManager | None = None
        self.voice_recognition: VoiceRecognition | None = None
        self.suggestion_engine: SmartSuggestionEngine | None = None
        self.anomaly_detector: AnomalyDetector | None = None
        self.pattern_analyzer: PatternAnalyzer | None = None
        self.trigger_manager: TriggerManager | None = None
        self.episodic_memory: EpisodicMemory | None = None
        self._current_user_id: str | None = None
        self._workflow_check_interval = 30  # seconds
        self._last_workflow_check = 0

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

    async def _get_agent_orchestrator(self):
        """Get agent orchestrator (lazy initialization)"""
        if self.agent_orchestrator is None:
            self.agent_orchestrator = AgentOrchestrator()
            # Register specialized agents
            from agents.base import AgentRole
            from agents.specialized import (
                CodeReviewAgent,
                CreativeAgent,
                PlanningAgent,
                ResearchAgent,
            )

            try:
                self.agent_orchestrator.register_agent(
                    "code_review", CodeReviewAgent(), AgentRole.CODE
                )
                self.agent_orchestrator.register_agent(
                    "research", ResearchAgent(), AgentRole.RESEARCH
                )
                self.agent_orchestrator.register_agent(
                    "creative", CreativeAgent(), AgentRole.CREATIVE
                )
                self.agent_orchestrator.register_agent(
                    "planning", PlanningAgent(), AgentRole.PLANNING
                )
            except Exception as e:
                log.warning(f"Failed to register some agents: {e}")
        return self.agent_orchestrator

    async def _get_user_manager(self):
        """Get user manager (lazy initialization)"""
        if self.user_manager is None:
            from pathlib import Path

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.user_manager = UserManager(str(data_dir / "users.db"))  # type: ignore[arg-type]
        return self.user_manager

    async def _get_voice_recognition(self):
        """Get voice recognition (lazy initialization)"""
        if self.voice_recognition is None:
            from pathlib import Path

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.voice_recognition = VoiceRecognition(str(data_dir / "voice_profiles"))  # type: ignore[arg-type]
        return self.voice_recognition

    async def _get_suggestion_engine(self):
        """Get suggestion engine (lazy initialization)"""
        if self.suggestion_engine is None:
            self.suggestion_engine = SmartSuggestionEngine()
        return self.suggestion_engine

    async def _get_anomaly_detector(self):
        """Get anomaly detector (lazy initialization)"""
        if self.anomaly_detector is None:
            self.anomaly_detector = AnomalyDetector()
        return self.anomaly_detector

    async def _get_pattern_analyzer(self):
        """Get pattern analyzer (lazy initialization)"""
        if self.pattern_analyzer is None:
            self.pattern_analyzer = PatternAnalyzer()
        return self.pattern_analyzer

    async def _get_trigger_manager(self):
        """Get trigger manager (lazy initialization)"""
        if self.trigger_manager is None:
            from datetime import timedelta

            self.trigger_manager = TriggerManager()
            # Set up default health check trigger
            health_trigger = TimeTrigger(
                trigger_id="health_check",
                name="Health Check Trigger",
                workflow_id="system_health",
                interval=timedelta(minutes=5),
            )
            await self.trigger_manager.register_trigger(health_trigger)
        return self.trigger_manager

    async def _get_episodic_memory(self):
        """Get episodic memory (lazy initialization)"""
        if self.episodic_memory is None:
            from pathlib import Path

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.episodic_memory = EpisodicMemory(str(data_dir / "episodic_memory.db"))  # type: ignore[arg-type]
        return self.episodic_memory

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

    async def _generate_reverse_prompt(self, user_message: str, response: str) -> str | None:
        """Generate a proactive suggestion after responding."""
        try:
            keywords = user_message.lower().split()
            suggestions = []

            if any(k in keywords for k in ["search", "find", "research", "look up"]):
                suggestions.append("Would you like me to search for more details on this topic?")

            if any(k in keywords for k in ["code", "program", "function", "script"]):
                suggestions.append("Want me to explain how this code works or optimize it?")

            if any(k in keywords for k in ["help", "how", "what"]):
                suggestions.append(
                    "I can also help with: files, system commands, web searches, and more."
                )

            if suggestions:
                suggestion_text = "\n\n[Suggestion: " + " | ".join(suggestions) + "]"
                if self._on_alert:
                    self._on_alert(
                        {
                            "type": "reverse_prompt",
                            "suggestion": suggestion_text,
                        }
                    )
                return suggestion_text
        except Exception as e:
            log.debug(f"Reverse prompt generation failed: {e}")
        return None

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
        tools_used = []

        from core.commands import process_command
        cmd_result = await process_command(text)
        if cmd_result:
            log.info(f"Command handled: {cmd_result.response[:50]}...")
            return cmd_result.response

        if self.websocket:
            log.info(f"Sending to WebSocket: {text[:100]}...")
            await self.websocket.send(json.dumps({"type": "user_message", "content": text}))
            if self._on_send_websocket:
                self._on_send_websocket(text)
            return None

        await self._conversation_buffer.add_message({"role": "user", "content": text})

        # === NEW: Intent Router - skip LLM for simple queries ===
        intent = self.intent_router.classify(text)
        if not intent.requires_llm:
            direct_response = self.intent_router.get_direct_response(intent)
            if direct_response:
                log.info(f"Direct response (no LLM): {direct_response[:50]}...")
                await self._conversation_buffer.add_message({
                    "role": "assistant", "content": direct_response
                })
                if self._on_response:
                    self._on_response(direct_response)
                return direct_response

        # Episodic memory (keep as-is)

        episodic = await self._get_episodic_memory()

        try:
            await episodic.record_conversation(user_message=text, assistant_response="")
        except Exception as e:
            log.debug(f"Failed to log to episodic memory: {e}")

        # Agent routing (keep as-is)
        try:
            orchestrator = await self._get_agent_orchestrator()
            routing = await orchestrator.route_request(text)

            if routing["use_agent"] and routing["confidence"] > 0.5:
                agent_response = await orchestrator.process_with_agent(
                    agent_type=routing["agent_type"],
                    request=text,
                    context={"user_id": self._current_user_id},
                )

                if agent_response["success"]:
                    response_text = agent_response["response"]
                    await self._conversation_buffer.add_message(
                        {"role": "assistant", "content": response_text}
                    )

                    # Log agent response to episodic memory
                    try:
                        await episodic.record_action(
                            action="agent_response",
                            result=f"Agent {routing['agent_type']} responded: {response_text[:200]}",
                        )
                    except Exception:
                        pass

                    return response_text
                else:
                    log.warning(
                        f"Agent processing failed: {agent_response.get('error')}, falling back to standard processing"
                    )
        except Exception as e:
            log.debug(f"Agent routing failed: {e}, using standard processing")

        # Memory recall (keep as-is)
        relevant_memories = await self._recall_relevant_memories(text)

        if self._forced_model:
            backend, model = self._forced_model
        else:
            selection = self.router.select_model(text)
            backend, model = selection.backend, selection.model

        full_response = ""
        tool_calls = []
        success = True
        error_msg = None

        try:
            client = self.ollama
            client.model = model

            # === NEW: Smart context + Token Compactor ===
            context_decision = await get_smart_context(text, role="user")
            recent_messages = context_decision.messages_to_include

            # Build extra context
            extra_context = ""
            if context_decision.semantic_memories:
                extra_context += "Relevant past context:"
                for mem in context_decision.semantic_memories:
                    extra_context += f"\n- {mem['text'][:100]}"

            if relevant_memories:
                extra_context += "\n\nRelevant information about the user:"
                for memory in relevant_memories:
                    extra_context += f"\n- {memory['fact']} ({memory['category']})"

            if self._current_user_id:
                try:
                    user_manager = await self._get_user_manager()
                    user = await user_manager.get_user(self._current_user_id)
                    if user:
                        display_name = getattr(user, "name", None) or getattr(user, "username", "the user")
                        extra_context += f"\n\nYou are speaking with {display_name}."
                        formality = getattr(user.preferences, "formality", None) or getattr(
                            user.preferences, "response_style", None
                        )
                        if formality == "casual":
                            extra_context += " Use a casual, friendly tone."
                        elif formality == "formal":
                            extra_context += " Use a formal, professional tone."
                except Exception as e:
                    log.debug(f"Failed to get user context: {e}")

            from core.skills import get_skill_prompt_context

            skills_context = get_skill_prompt_context()
            if skills_context:
                extra_context += "\n\n" + skills_context

            available_tool_names = []
            tool_schemas = self.tools.get_filtered_schemas(text) if self.tools else None

            # If IntentRouter suggested tools, filter to only those
            if intent.suggested_tools and tool_schemas:
                tool_schemas = [
                    s for s in tool_schemas
                    if s.get("function", {}).get("name") in intent.suggested_tools
                ]

            if tool_schemas:
                available_tool_names = [
                    s.get("function", {}).get("name", "") for s in tool_schemas if isinstance(s.get("function"), dict)
                ]

            from core.skills import get_skill_loader

            skill_loader = get_skill_loader()
            available_skills = list(skill_loader.get_all_skills().keys())

            tools_and_skills_str = (
                f"[TOOLS: {', '.join(available_tool_names)} | SKILLS: {', '.join(available_skills)}]"
            )

            base_system_prompt = (
                f"You are JARVIS, a helpful AI assistant. Be concise and direct. "
                f"You have access to many tools. ALWAYS use tools when you need to get information "
                f"about the system, files, applications, or perform any actions. "
                f"Do NOT say you cannot do something if there is a tool available for it.\n\n"
                f"{tools_and_skills_str}"
            )

            # === TokenCompactor compresses context for the LLM ===
            compacted = self.token_compactor.prepare_context(
                query=text,
                conversation_messages=recent_messages,
                all_tool_schemas=list(tool_schemas or []),
                system_prompt=base_system_prompt,
                extra_context=extra_context,
            )

            log.info(
                f"Compacted context: {len(compacted.messages)} msgs, "
                f"{len(compacted.tool_schemas)} tools, "
                f"{compacted.estimated_tokens} est tokens "
                f"({compacted.compression_ratio*100:.0f}% compression)"
            )

            # === SINGLE LLM PASS (no retry loop) ===
            async for chunk in client.chat(
                messages=compacted.messages,
                system=compacted.system_prompt,
                tools=compacted.tool_schemas if compacted.tool_schemas else None,
            ):
                if self._interrupted:
                    log.info("LLM generation interrupted")
                    return None
                if msg := chunk.get("message", {}):
                    if content := msg.get("content"):
                        full_response += content
                    if calls := msg.get("tool_calls"):
                        tool_calls.extend(calls)

            log.info(
                "LLM pass done. tool_calls=%d, response_len=%d",
                len(tool_calls),
                len(full_response),
            )

            # === TOOL EXECUTION - single pass, no second LLM call ===
            if tool_calls and self.tools:
                tool_names = [c.get("function", {}).get("name") for c in tool_calls]
                tools_used.extend([n for n in tool_names if n])
                log.info("Processing %d tool calls: %s", len(tool_calls), tool_names)

                # Store assistant message with tool calls
                await self._conversation_buffer.add_message(
                    {"role": "assistant", "content": full_response, "tool_calls": tool_calls}
                )

                tool_results = await self._process_tool_calls(tool_calls)
                log.info("Tool results: %d messages", len(tool_results))

                for result in tool_results:
                    await self._conversation_buffer.add_message(result)

                # Handle vision tools specially (need interpretation)
                is_vision = "screenshot_analyze" in tool_names
                if is_vision and tool_results:
                    try:
                        vision_data = json.loads(tool_results[0].get("content", "{}"))
                        full_response = vision_data.get("analysis", "")
                        log.info("Using vision response directly: %d chars", len(full_response))
                    except Exception as e:
                        log.warning("Failed to extract vision response: %s", e)
            # Parse any embedded tool calls in response text (fallback)
            additional_tool_calls = self._parse_tool_calls_from_text(full_response)
            if additional_tool_calls and self.tools:
                additional_names = [c.get("function", {}).get("name") for c in additional_tool_calls]
                tools_used.extend([n for n in additional_names if n])
                log.info("Found %d more tool calls in response text", len(additional_tool_calls))
                tool_calls.extend(additional_tool_calls)
                await self._conversation_buffer.add_message({"role": "assistant", "content": full_response})
                tool_results = await self._process_tool_calls(additional_tool_calls)
                for result in tool_results:
                    await self._conversation_buffer.add_message(result)
        except asyncio.CancelledError:
            log.info("LLM generation cancelled")
            return None
        except Exception as e:
            success = False
            error_msg = str(e)
            log.error("LLM error: %s", e)
            full_response = "I encountered an error processing your request."

        await self._conversation_buffer.add_message({"role": "assistant", "content": full_response})

        try:
            ctx_manager = get_context_manager()
            await ctx_manager.add_to_memory("user", text)
            await ctx_manager.add_to_memory("assistant", full_response)
        except Exception as e:
            log.debug(f"Failed to store in semantic memory: {e}")

        log.debug(
            "Response: %s",
            full_response[:100] + "..." if len(full_response) > 100 else full_response,
        )
        if self._on_response:
            self._on_response(full_response)

        # Generate proactive suggestion based on conversation
        suggestion = await self._generate_reverse_prompt(text, full_response)

        self.learning.log_command(
            user_input=text,
            response=full_response,
            success=success,
            context={"error": error_msg} if error_msg else None,
        )

        self.proactive.extract_deadline_from_text(text)

        header_parts = []

        if tools_used:
            unique_tools = list(dict.fromkeys(tools_used))
            header_parts.append(f"TOOLS: {', '.join(unique_tools)}")

        from core.skills import detect_skills_used

        skills_detected = detect_skills_used(text, full_response)
        if skills_detected:
            header_parts.append(f"SKILLS: {', '.join(skills_detected)}")

        if header_parts:
            header = "[" + " | ".join(header_parts) + "]\n\n"
            full_response = header + full_response

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
            tools = self.tools
            if tools:
                result = await tools.execute(name, **args)
                tool_output = result.data if result.success else {"error": result.error}
            else:
                tool_output = {"error": "No tools available"}
            log.info("Tool result: %s", tool_output)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_output),
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
        tools = self.tools
        if tools:
            result = await tools.execute(name, **args)
            tool_output = result.data if result.success else {"error": result.error}
        else:
            tool_output = {"error": "No tools available"}
        log.info("Tool result: %s", tool_output)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(tool_output),
        }

    async def _execute_single_tool_async(self, call: dict) -> dict:
        """Execute a single tool call asynchronously (for parallel execution)"""
        return await self._execute_single_tool(call)

    def _looks_like_tool_call(self, response_text: str) -> bool:
        """Check if the response looks like it's trying to make a tool call."""
        if not response_text:
            return False
        text = response_text.lower()
        # Patterns that indicate tool calls
        patterns = [
            r"<tool_call>",
            r"function_call",
            r"\"name\":\s*\"\w+_\w+\"",  # JSON tool name pattern
            r"launch_app\s*\(",
            r"run_command\s*\(",
            r"web_search\s*\(",
            r"execute_python\s*\(",
        ]
        import re

        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

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

    async def handle_wake_word(self, audio_samples: np.ndarray | None = None) -> None:
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

        # Phase 6: Attempt voice identification if audio samples provided
        if audio_samples is not None:
            try:
                voice_recognition = await self._get_voice_recognition()
                identification = await voice_recognition.identify_speaker(audio_samples)

                if isinstance(identification, dict):
                    ident: dict[str, Any] = identification
                    if ident.get("success") and ident.get("confidence", 0) > 0.6:
                        self._current_user_id = ident.get("speaker_id")
                        log.info(
                            f"Identified user via voice: {ident.get('speaker_name')} "
                            f"(confidence: {ident.get('confidence', 0):.2f})"
                        )

                        # Update user context
                        user_manager = await self._get_user_manager()
                        if user_manager:
                            await user_manager.record_interaction(self._current_user_id)  # type: ignore[attr-defined]  # type: ignore[attr-defined]
                    else:
                        log.debug(
                            f"Voice identification uncertain: {ident.get('confidence', 0):.2f}"
                        )
                        self._current_user_id = None
                else:
                    log.debug(f"Voice identification returned non-dict: {identification}")
                    self._current_user_id = None
            except Exception as e:
                log.debug(f"Voice identification failed: {e}")
                self._current_user_id = None

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
                    if self._loop:
                        self._loop.call_soon_threadsafe(  # type: ignore[reportOptionalMemberAccess]
                            self._audio_queue.put_nowait, audio_chunk
                        )
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
                if self._audio_queue is None:
                    continue
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
        workflow_task = None
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

                # Phase 6: Initialize new systems
                log.info("Initializing new JARVIS systems...")
                await self._get_agent_orchestrator()
                await self._get_user_manager()
                await self._get_voice_recognition()
                await self._get_suggestion_engine()
                await self._get_anomaly_detector()
                await self._get_pattern_analyzer()
                await self._get_trigger_manager()
                await self._get_episodic_memory()
                log.info("All systems initialized successfully")

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, lambda: self.vad.is_speech(np.zeros(self._chunk_size, dtype=np.float32))
                )

                voice_startup = self.config.get("voice_input.startup_enabled", True)
                if not voice_startup:
                    log.info("Voice startup disabled - text-only mode")
                else:
                    log.info("Voice startup enabled")
                    try:
                        from pathlib import Path

                        model_dir = Path.home() / ".cache" / "openwakeword"
                        hey_jarvis_model = model_dir / "hey_jarvis.onnx"

                        if not hey_jarvis_model.exists():
                            log.info("Downloading wake word models...")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None,
                                lambda: openwakeword.utils.download_models() if hasattr(openwakeword, "utils") else None,  # pyright: ignore[reportAttributeAccessIssue]
                            )
                        else:
                            log.info("Wake word models exist, skipping download")
                    except Exception as e:
                        log.warning(f"Could not check wake word models: {e}")

                    log.info("Starting wake word detector")
                    loop = asyncio.get_running_loop()
                    self.wake_detector = await loop.run_in_executor(None, WakeWordDetector)
                    self.wake_detector.start(self._wake_callback)

                log.info("Starting audio stream")
                audio_task = asyncio.create_task(self._process_audio_queue())

                # Phase 6: Start workflow trigger checking
                workflow_task = asyncio.create_task(self._workflow_check_loop())

                while self._running:
                    await asyncio.sleep(0.1)
        except Exception as e:
            log.error("Exception in VoiceAssistant.run(): %s", e, exc_info=True)
        finally:
            self._running = False
            if audio_task:
                audio_task.cancel()
            if workflow_task:
                workflow_task.cancel()
            await self.threading_manager.cancel_all_tasks()

    async def _workflow_check_loop(self) -> None:
        """Periodically check workflow triggers"""
        log.info("Starting workflow check loop")
        while self._running:
            try:
                await asyncio.sleep(self._workflow_check_interval)

                if not self._running:
                    break

                # Check trigger manager
                trigger_manager = await self._get_trigger_manager()
                triggered = await trigger_manager.check_all()

                for trigger_id in triggered:
                    log.info(f"Workflow trigger activated: {trigger_id}")
                    # Execute associated workflow action
                    await self._execute_trigger_action(trigger_id)

                # Phase 6: Check for anomalies
                await self._check_anomalies()

                # Phase 6: Generate proactive suggestions
                await self._generate_proactive_suggestions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"Workflow check error: {e}")

    async def _execute_trigger_action(self, trigger_id: str) -> None:
        """Execute action associated with a trigger"""
        try:
            # Map trigger IDs to actions
            if trigger_id == "health_check":
                health = await self.health_check()
                failed = [k for k, v in health.items() if not v]
                if failed:
                    log.warning(f"Health check failed for: {', '.join(failed)}")
            # Add more trigger actions as needed
        except Exception as e:
            log.error(f"Failed to execute trigger action for {trigger_id}: {e}")

    async def _check_anomalies(self) -> None:
        """Check for system anomalies"""
        try:
            anomaly_detector = await self._get_anomaly_detector()

            # Check system health for anomalies
            health = await self.health_check()
            health_data = {
                "tts_online": health.get("tts", False),
                "stt_online": health.get("stt", False),
                "vad_online": health.get("vad", False),
                "wake_word_online": health.get("wake_word", False),
            }

            anomaly_result = await anomaly_detector.check_system_health(health_data)

            if anomaly_result.get("is_anomaly", False):
                log.warning(f"System anomaly detected: {anomaly_result}")
                if self._on_alert:
                    self._on_alert(
                        {
                            "type": "anomaly",
                            "severity": anomaly_result.get("severity", "low"),
                            "details": anomaly_result,
                        }
                    )
        except Exception as e:
            log.debug(f"Anomaly check failed: {e}")

    async def _generate_proactive_suggestions(self) -> None:
        """Generate proactive suggestions based on patterns"""
        try:
            if not self._current_user_id:
                return

            suggestion_engine = await self._get_suggestion_engine()

            context: dict[str, Any] = {
                "time_of_day": asyncio.get_event_loop().time(),
                "user_id": self._current_user_id,
                "active_applications": [],
            }

            suggestions = await suggestion_engine.generate_suggestions(
                context=context  # type: ignore[arg-type]
            )

            # Filter for high-priority suggestions
            high_priority = [s for s in suggestions if getattr(s, "priority", None) == "high"]

            for suggestion in high_priority:
                log.info(f"Proactive suggestion: {getattr(suggestion, 'message', '')}")
                if self._on_alert:
                    self._on_alert(
                        {
                            "type": "suggestion",
                            "message": getattr(suggestion, "message", ""),
                            "action": getattr(suggestion, "suggested_action", None),
                        }
                    )

            # Filter for high-priority suggestions
            high_priority = [s for s in suggestions if s.priority.value == "high"]

            for suggestion in high_priority:
                log.info(f"Proactive suggestion: {suggestion.description}")
                if self._on_alert:
                    self._on_alert(
                        {
                            "type": "suggestion",
                            "message": suggestion.description,
                            "action": suggestion.action,
                        }
                    )
        except Exception as e:
            log.debug(f"Proactive suggestion generation failed: {e}")

    async def stop(self) -> None:
        self._running = False
        self.wake_word.stop()
        self.proactive.stop()

        # Phase 6: Clean up new systems (best-effort, these may not have cleanup methods)
        for obj, method in [
            (self.agent_orchestrator, "shutdown"),
            (self.user_manager, "close"),
            (self.episodic_memory, "close"),
        ]:
            if obj and hasattr(obj, method):
                try:
                    await getattr(obj, method)()
                except Exception:
                    log.debug(f"Cleanup {type(obj).__name__}.{method}() failed (non-fatal)")

        await self.llm.close()
        await self.tts.close()
        await self.threading_manager.cancel_all_tasks()

    def record_positive_feedback(self) -> None:
        self.learning.record_positive_feedback()

    def record_negative_feedback(self, reason: str | None = None) -> None:
        self.learning.record_negative_feedback(reason)  # type: ignore[arg-type]

    async def check_permission(self, action: str) -> bool:
        return await self.permissions.check_permission_async(action)

    def get_pending_alerts(self) -> list:
        return self.proactive.get_pending_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        return self.proactive.acknowledge_alert(alert_id)

    def get_improvement_report(self, days: int = 7) -> dict:
        return self.learning.get_improvement_report(days)

    # Phase 6: New utility methods
    async def create_user(self, name: str, voice_samples: list | None = None) -> dict:
        """Create a new user with optional voice enrollment"""
        try:
            user_manager = await self._get_user_manager()
            user = await user_manager.create_user(username=name)
            user_id = user.id

            if voice_samples:
                voice_recognition = await self._get_voice_recognition()
                for sample in voice_samples:
                    await voice_recognition.enroll_user(user_id, sample)  # type: ignore[arg-type]

            return {"success": True, "user_id": user_id, "name": name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def switch_user(self, user_id: str) -> dict:
        """Switch to a different user context"""
        try:
            user_manager = await self._get_user_manager()
            user = await user_manager.get_user(user_id)

            if user:
                self._current_user_id = user_id
                await user_manager.record_interaction(user_id)  # type: ignore[attr-defined]
                return {"success": True, "user": user.to_dict()}  # type: ignore[attr-defined]
            else:
                return {"success": False, "error": "User not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_current_user(self) -> dict | None:
        """Get current user information"""
        if not self._current_user_id:
            return None

        try:
            user_manager = await self._get_user_manager()
            user = await user_manager.get_user(self._current_user_id)
            return user.to_dict() if user else None  # type: ignore[attr-defined]
        except Exception:
            return None

    async def get_conversation_history(self, limit: int = 10) -> list:
        """Get recent conversation history from episodic memory"""
        try:
            episodic = await self._get_episodic_memory()
            return await episodic.get_recent_episodes(limit=limit, user_id=self._current_user_id)  # type: ignore[attr-defined]
        except Exception as e:
            log.debug(f"Failed to get conversation history: {e}")
            return []

    async def create_workflow(self, name: str, actions: list, triggers: list | None = None) -> dict:
        """Create a new automated workflow"""
        try:
            trigger_manager = await self._get_trigger_manager()
            workflow_id = f"workflow_{name}_{asyncio.get_event_loop().time()}"

            # Create time trigger if interval specified
            if triggers:
                for trigger_config in triggers:
                    trigger = TimeTrigger(
                        trigger_id=f"{workflow_id}_{trigger_config.get('type')}",
                        name=trigger_config.get("type", "interval"),
                        workflow_id=workflow_id,
                        interval=datetime.timedelta(minutes=trigger_config.get("interval_minutes", 60)),
                    )
                    await trigger_manager.register_trigger(trigger)

            return {"success": True, "workflow_id": workflow_id, "name": name}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
