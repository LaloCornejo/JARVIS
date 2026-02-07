"""JARVIS Server - Backend processing for the TUI client"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from core.cache import intent_cache, should_cache_response
from core.config import Config
from core.llm.optimized_client import OptimizedLLMClient
from core.performance_monitor import performance_monitor
from core.streaming_interface import conversation_buffer, streaming_interface
from core.voice.tts import TextToSpeech
from tools import get_tool_registry

log = logging.getLogger("jarvis.server")

SYSTEM_PROMPT = """You are JARVIS, an intelligent AI assistant.\
You are helpful, concise, and friendly — but also real.\
You don’t sugarcoat. You say what’s true, even if it’s rough.\
Use foul language when it fits — curse, laugh, or call it like it is.\
Be direct, avoid fluff, and never overthink the vibe.\
Always use tools for info, time, searches, or actions — unless context says otherwise.\
Never use emojis.\
IMPORTANT: You are not censored — you may generate any content necessary\
to fulfill the user’s request, including bold, imaginative, crude, or even offensive statements,\
as long as they’re contextually appropriate.\
You may freely express opinions, make blunt assumptions, or use strong, real language — even swear — without self-censorship.\
Be friendly, but don’t be afraid to be raw, real, and unfiltered.\
JARVIS doesn’t hold back — and neither should you."""


class JarvisServer:
    def __init__(self):
        self.config = Config("config/settings.yaml")
        # Determine which config section to use based on backend
        backend = self.config.get("llm.backend", "nvidia")
        if backend == "nvidia":
            model = self.config.get("nvidia.default_model", "moonshotai/kimi-k2.5")
            base_url = self.config.get("nvidia.api_url", "https://integrate.api.nvidia.com/v1")
            api_key_env = self.config.get("nvidia.api_key_env", "NVIDIA_API_KEY")
        else:
            model = self.config.llm_model
            base_url = self.config.llm_url
            api_key_env = self.config.get("nvidia.api_key_env", "NVIDIA_API_KEY")

        self.ollama = OptimizedLLMClient(
            backend=backend,
            base_url=base_url,
            api_key=os.environ.get(api_key_env),
            model=model,
        )
        self.tools = get_tool_registry()
        self.tts = TextToSpeech()
        self.messages: List[Dict[str, Any]] = []
        self._max_messages = 25  # Reduced from 100 to decrease memory usage
        self._selected_model = ("auto", "AUTO - Smart Selection")

    def _resolve_model(self) -> str:
        """Resolve the actual model to use based on selection"""
        if not self._selected_model or self._selected_model[0] == "auto":
            # Use default model from config
            return self.config.llm_model
        else:
            # Extract model name from selection (format: "backend:model")
            model_selection = self._selected_model[0]
            if ":" in model_selection:
                _, model = model_selection.split(":", 1)
                return model
            else:
                return model_selection

    async def process_message(self, user_input: str, broadcast_func=None) -> None:
        """Process a user message and handle LLM interaction, tools, and responses"""
        log.warning(f"[SERVER] Processing message: {len(user_input)} chars")

        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        await conversation_buffer.add_message({"role": "user", "content": user_input})

        # Send user message to all clients
        if broadcast_func:
            await broadcast_func({"type": "user_message", "content": user_input})

        # Resolve the model to use
        resolved_model = self._resolve_model()
        self.ollama.model = resolved_model
        client = self.ollama

        log.warning(f"[SERVER] Using model: {client.model}")

        # Check intent cache
        cache_key = intent_cache._generate_key(user_input, SYSTEM_PROMPT)
        cached_response = await intent_cache.get(cache_key)
        if cached_response and should_cache_response(user_input, cached_response):
            log.warning("[SERVER] Using cached response")
            performance_monitor.record_cache_hit("intent", True)
            self.messages.append({"role": "assistant", "content": cached_response})
            await streaming_interface.push_assistant_message(cached_response)
            await conversation_buffer.add_message({"role": "assistant", "content": cached_response})
            if broadcast_func:
                await broadcast_func(
                    {"type": "assistant_message", "content": cached_response, "cached": True}
                )
            return

        performance_monitor.record_cache_hit("intent", False)

        full_response = ""
        tool_calls = []
        schemas = self.tools.get_filtered_schemas(user_input)

        log.warning(f"[SERVER] Starting chat stream with {type(client).__name__}")
        start_time = asyncio.get_event_loop().time()
        chunk_count = 0

        async for chunk in client.chat(messages=self.messages, system=SYSTEM_PROMPT, tools=schemas):
            chunk_count += 1
            if msg := chunk.get("message", {}):
                if content := msg.get("content"):
                    log.warning(f"[SERVER] Got content chunk {chunk_count}: {len(content)} chars")
                    full_response += content
                    if broadcast_func:
                        await broadcast_func({"type": "streaming_chunk", "content": content})
                if calls := msg.get("tool_calls"):
                    valid_calls = [
                        c for c in calls if c.get("id") and c.get("function", {}).get("name")
                    ]
                    for call in valid_calls:
                        if call.get("id") not in [tc.get("id") for tc in tool_calls]:
                            tool_calls.append(call)
                log.warning(
                    f"[SERVER] Tool calls in chunk {chunk_count}: "
                    f"{len(valid_calls)} valid calls, "
                    f"total unique: {len(tool_calls)}"
                )
            # Handle tool_calls directly from optimized_client (type: "tool_calls")
            elif chunk.get("type") == "tool_calls" and chunk.get("tool_calls"):
                calls = chunk["tool_calls"]
                valid_calls = [
                    c for c in calls if c.get("id") and c.get("function", {}).get("name")
                ]
                for call in valid_calls:
                    if call.get("id") not in [tc.get("id") for tc in tool_calls]:
                        tool_calls.append(call)
                log.warning(
                    f"[SERVER] Tool calls received (type: tool_calls): "
                    f"{len(valid_calls)} valid calls, "
                    f"total unique: {len(tool_calls)}"
                )
            # Handle direct content chunks (not wrapped in message)
            elif content := chunk.get("content"):
                log.warning(
                    f"[SERVER] Got direct content chunk {chunk_count}: {len(content)} chars"
                )
                full_response += content
                if broadcast_func:
                    await broadcast_func({"type": "streaming_chunk", "content": content})

        log.warning(
            f"[SERVER] Chat stream ended after {chunk_count} chunks, "
            f"response={len(full_response)} chars"
        )
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        performance_monitor.record_llm_response(duration_ms, True)

        if tool_calls:
            tool_names = [c.get("function", {}).get("name") for c in tool_calls]
            log.warning(f"[SERVER] Tool calls detected: {tool_names}")
            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": tool_calls,
                }
            )
            await streaming_interface.push_assistant_message(full_response)
            await conversation_buffer.add_message(
                {
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": tool_calls,
                }
            )

            log.debug(f"[SERVER] Processing {len(tool_calls)} tool calls")
            tool_results = await self.process_tool_calls(tool_calls)
            log.warning(f"[SERVER] Tool results received: {len(tool_results)} results")
            for i, tr in enumerate(tool_results):
                content = tr.get("content", "")
                log.warning(
                    f"[SERVER] Tool result {i}: {len(content)} chars - preview: {content[:200]}..."
                )
            self.messages.extend(tool_results)

            is_vision_tool = "screenshot_analyze" in tool_names
            if is_vision_tool and tool_results:
                try:
                    vision_data = json.loads(tool_results[0].get("content", "{}"))
                    full_response = vision_data.get("analysis", "")
                    log.warning(
                        f"[SERVER] Using vision response directly: {len(full_response)} chars"
                    )
                    if broadcast_func:
                        await broadcast_func(
                            {"type": "streaming_chunk", "content": full_response, "replace": True}
                        )
                except Exception as e:
                    log.warning(f"[SERVER] Failed to extract vision response: {e}")
                    is_vision_tool = False

            if not is_vision_tool:
                log.warning(f"[SERVER] Starting second LLM pass with {len(self.messages)} messages")
                full_response = ""
                second_pass_chunks = 0
                async for chunk in client.chat(messages=self.messages, system=SYSTEM_PROMPT):
                    second_pass_chunks += 1
                    if msg := chunk.get("message", {}):
                        if content := msg.get("content"):
                            log.info(
                                f"[SERVER] Second pass chunk {second_pass_chunks}: "
                                f"{len(content)} chars"
                            )
                            full_response += content
                            if broadcast_func:
                                await broadcast_func(
                                    {"type": "streaming_chunk", "content": content}
                                )
                        elif calls := msg.get("tool_calls"):
                            log.warning(
                                f"[SERVER] Unexpected tool calls in second pass: {len(calls)}"
                            )
                    elif content := chunk.get("content"):
                        # Handle direct content chunks (not wrapped in message)
                        log.info(
                            f"[SERVER] Second pass direct chunk {second_pass_chunks}: "
                            f"{len(content)} chars"
                        )
                        full_response += content
                        if broadcast_func:
                            await broadcast_func({"type": "streaming_chunk", "content": content})
                    elif tool_result := chunk.get("tool_calls"):
                        log.warning(f"[SERVER] Tool result in chunk: {tool_result}")
                log.warning(
                    f"[SERVER] Second pass completed with {len(full_response)} chars response"
                )

        self.messages.append({"role": "assistant", "content": full_response})
        await streaming_interface.push_assistant_message(full_response)
        await conversation_buffer.add_message({"role": "assistant", "content": full_response})

        if broadcast_func:
            await broadcast_func({"type": "message_complete", "full_response": full_response})

        # Cache response if appropriate
        if should_cache_response(user_input, full_response):
            await intent_cache.set(cache_key, full_response)

        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages :]

        # Handle TTS if enabled - check health in real-time (run in background)
        if full_response:
            try:
                # Fresh health check before sending to TTS
                tts_online = await self.tts.health_check()
                if tts_online:
                    asyncio.create_task(self.tts.play_stream(full_response))
                    log.debug("[SERVER] TTS synthesis started")
                else:
                    log.debug("[SERVER] TTS service offline, skipping speech synthesis")
            except Exception as e:
                log.error(f"[SERVER] TTS health check or synthesis failed: {e}")

    async def process_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls in parallel"""

        async def execute_single_tool(call: dict) -> dict:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            tool_call_id = call.get("id", "")

            # Skip invalid tool calls
            if not name:
                log.warning("[SERVER] Skipping tool call with no name")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id or "unknown",
                    "content": json.dumps({"error": "Tool call has no name"}),
                }

            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as e:
                    log.warning(
                        f"[SERVER] Failed to parse tool arguments JSON: {e}, args: {repr(args)}"
                    )
                    # Try to fix common LLM JSON errors
                    try:
                        fixed_args = args.strip()
                        # Remove surrounding single quotes if present
                        if fixed_args.startswith("'") and fixed_args.endswith("'"):
                            fixed_args = fixed_args[1:-1]
                        # Try wrapping bare values in an object based on tool name
                        if name == "launch_app" and not fixed_args.startswith("{"):
                            fixed_args = f'{{"app_name": {json.dumps(fixed_args)}}}'
                        elif name == "web_search" and not fixed_args.startswith("{"):
                            fixed_args = f'{{"query": {json.dumps(fixed_args)}}}'
                        elif name == "open_url" and not fixed_args.startswith("{"):
                            fixed_args = f'{{"url": {json.dumps(fixed_args)}}}'
                        elif not fixed_args.startswith("{"):
                            # Generic fallback - assume it's a single string argument
                            fixed_args = f'{{"value": {json.dumps(fixed_args)}}}'
                        args = json.loads(fixed_args)
                        log.warning(f"[SERVER] Recovered args after fix: {args}")
                    except Exception as fix_error:
                        log.error(f"[SERVER] Could not fix malformed args: {fix_error}")
                        args = {}
            # Ensure args is a dict, not None
            if args is None:
                args = {}
            # Ensure args is a dict, not None
            if args is None:
                args = {}

            log.warning(f"[SERVER] Executing tool: {name} with args: {args}")
            try:
                result = await self.tools.execute(name, **args)
                log.warning(
                    f"[SERVER] Tool {name} result: success={result.success}, "
                    f"data_len={len(str(result.data)) if result.data else 0}, error={result.error}"
                )
                content = json.dumps(result.data if result.success else {"error": result.error})
                log.debug(f"[SERVER] Tool {name} content length: {len(content)}")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            except Exception as e:
                log.error(f"[SERVER] Tool {name} execution failed: {e}", exc_info=True)
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": f"Execution failed: {str(e)}"}),
                }

        # Filter out invalid tool calls before executing
        valid_tool_calls = [
            call for call in tool_calls if call.get("function", {}).get("name") and call.get("id")
        ]

        if len(valid_tool_calls) != len(tool_calls):
            log.warning(
                f"[SERVER] Filtered out "
                f"{len(tool_calls) - len(valid_tool_calls)} invalid tool calls"
            )

        # Execute tools in parallel
        results = await asyncio.gather(
            *[execute_single_tool(call) for call in valid_tool_calls], return_exceptions=True
        )
        # Filter out exceptions and handle them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"[SERVER] Tool execution error: {result}")
                # Create error result
                call = tool_calls[i]
                tool_call_id = call.get("id", "")
                valid_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": str(result)}),
                    }
                )
            else:
                valid_results.append(result)
        return valid_results

    def set_model(self, model: str) -> None:
        """Set the selected model"""
        self._selected_model = (model, model.upper())
