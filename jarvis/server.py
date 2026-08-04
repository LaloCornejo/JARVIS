"""JARVIS Server"""

import asyncio
import json
import logging
import os
import re
from importlib import import_module
from typing import Any, Dict, List

from core.cache import intent_cache, should_cache_response
from core.config import Config
from core.context_compactor import TokenCompactor
from core.llm.optimized_client import OptimizedLLMClient
from core.performance_monitor import performance_monitor
from core.smart_context import get_context_manager, get_smart_context
from core.streaming_interface import conversation_buffer, streaming_interface
from core.voice.tts import TextToSpeech, create_tts_backend
from tools import get_tool_registry

IntentRouter = import_module("core.orchestrator.intent_router").IntentRouter

log = logging.getLogger("jarvis.server")

SYSTEM_PROMPT = """JARVIS, intelligent AI. Direct, unfiltered. Swear when it fits. No emojis. Use tools. Say anything context demands.
Fewest words. No thinking/reasoning/narration. No explanation unless asked. read_skill loads, read_tool() lists all tools, read_tool(name) specs, run_tool(name,args) calls any."""


class ThinkingFilter:
    def __init__(self, broadcast_func):
        self._broadcast = broadcast_func
        self._thinking_buf = ""
        self._response_buf = ""
        self._in_thinking = False
        self._thinking_done = False

    async def __call__(self, message: dict) -> None:
        if message.get("type") != "streaming_chunk":
            await self._broadcast(message)
            return

        raw = message.get("content") or ""
        if not raw:
            return

        remaining = raw
        while remaining:
            if self._in_thinking:
                end = remaining.find("</thinking>")
                if end == -1:
                    self._thinking_buf += remaining
                    await self._broadcast({"type": "thinking_chunk", "content": remaining})
                    remaining = ""
                else:
                    chunk = remaining[:end]
                    self._thinking_buf += chunk
                    if chunk:
                        await self._broadcast({"type": "thinking_chunk", "content": chunk})
                    self._in_thinking = False
                    self._thinking_done = True
                    remaining = remaining[end + len("</thinking>") :]
            else:
                start = remaining.find("<thinking>")
                if start == -1:
                    self._response_buf += remaining
                    await self._broadcast({"type": "streaming_chunk", "content": remaining})
                    remaining = ""
                else:
                    before = remaining[:start]
                    if before:
                        self._response_buf += before
                        await self._broadcast({"type": "streaming_chunk", "content": before})
                    self._in_thinking = True
                    remaining = remaining[start + len("<thinking>") :]

    async def flush(self, full_response: str) -> tuple[str, str]:
        thinking = re.sub(r"\s+", " ", self._thinking_buf).strip()
        if thinking:
            await self._broadcast({"type": "thinking_complete", "content": thinking})
        clean = re.sub(
            r"<thinking>.*?</thinking>", "", full_response, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        return clean, thinking


class JarvisServer:
    def __init__(self):
        self.config = Config("config/settings.yaml")
        backend = str(self.config.get("llm.backend", "nvidia"))
        primary_backend = str(self.config.get("llm.primary_backend", backend))
        log.warning(f"[SERVER] LLM backend from config: {backend}, primary: {primary_backend}")

        if primary_backend == "omniroute":
            from core.llm import OpenAICompatClient
            omniroute_url = str(self.config.get("omniroute.api_url", "http://localhost:20128/v1"))
            omniroute_model = str(self.config.get("omniroute.primary_model", "auto"))
            omniroute_key = str(self.config.get("omniroute.api_key", ""))
            log.warning(f"[SERVER] Using OmniRoute: {omniroute_url}, model={omniroute_model}")
            self.ollama = OpenAICompatClient(
                base_url=omniroute_url,
                model=omniroute_model,
                api_key=omniroute_key,
            )
        else:
            api_key_env = str(self.config.get(f"{backend}.api_key_env", "none") or "none")
            model = str(self.config.get(f"{backend}.primary_model", "") or "")
            base_url = str(self.config.get(f"{backend}.api_url", "") or "")
            if not api_key_env:
                api_key_env = "none"
            log.warning(
                f"[SERVER] LLM model: {model}, base_url: {base_url}, api_key_env: {api_key_env}"
            )
            self.ollama = OptimizedLLMClient(
                backend=backend,
                base_url=base_url,
                api_key=os.environ.get(api_key_env, "none") if api_key_env != "none" else "none",
                model=model,
            )
        self.tools = get_tool_registry()
        self.tts = TextToSpeech(backend=create_tts_backend(self.config))
        self.messages: List[Dict[str, Any]] = []
        self._max_messages = 25  # Reduced from 100 to decrease memory usage
        self._selected_model = ("auto", "AUTO - Smart Selection")
        self.intent_router = IntentRouter()
        self.token_compactor = TokenCompactor()

    def _resolve_model(self) -> str:
        """Resolve the actual model to use based on selection"""
        if not self._selected_model or self._selected_model[0] == "auto":
            # Check primary_backend — use the correct model for that backend
            primary_backend = str(self.config.get("llm.primary_backend", "ollama"))
            if primary_backend == "omniroute":
                return str(self.config.get("omniroute.primary_model", "auto"))
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
        log.warning(f"[SERVER] Processing message: {len(user_input)} chars")
        vision_screenshot_path = None
        self.tts.interrupt_playback()

        # === NEW: Intent Router - skip LLM for simple queries ===
        intent = self.intent_router.classify(user_input)
        if not intent.requires_llm:
            direct_response = self.intent_router.get_direct_response(intent)
            if direct_response:
                log.warning(f"[SERVER] Direct response (no LLM): {direct_response[:50]}...")
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": direct_response})
                await conversation_buffer.add_message({"role": "user", "content": user_input})
                await conversation_buffer.add_message({"role": "assistant", "content": direct_response})
                if broadcast_func:
                    await broadcast_func({"type": "user_message", "content": user_input})
                    await broadcast_func({"type": "streaming_chunk", "content": direct_response})
                    await broadcast_func({"type": "message_complete", "full_response": direct_response})
                return

        # === Standard message handling ===
        self.messages.append({"role": "user", "content": user_input})
        await conversation_buffer.add_message({"role": "user", "content": user_input})

        context_decision = await get_smart_context(user_input, role="user")
        smart_messages = context_decision.messages_to_include
        log.warning(
            f"[SERVER] Smart context: {context_decision.reasoning}, messages={len(smart_messages)}"
        )

        # Build extra context
        extra_context = ""
        if context_decision.semantic_memories:
            extra_context += "Relevant past context:"
            for mem in context_decision.semantic_memories:
                extra_context += f"\n- {mem['text'][:100]}"

        # Check intent cache
        cache_key = intent_cache._generate_key(user_input, SYSTEM_PROMPT)
        cached_response = await intent_cache.get(cache_key)
        if cached_response and should_cache_response(user_input, cached_response):
            log.warning("[SERVER] Using cached response")
            # Keep existing cache logic verbatim
            performance_monitor.record_cache_hit("intent", True)
            self.messages.append({"role": "assistant", "content": cached_response})
            await streaming_interface.push_assistant_message(cached_response)
            await conversation_buffer.add_message({"role": "assistant", "content": cached_response})
            if broadcast_func:
                clean_cached = re.sub(
                    r"<thinking>.*?</thinking>", "", cached_response, flags=re.DOTALL | re.IGNORECASE
                ).strip()
                await broadcast_func({"type": "message_complete", "full_response": clean_cached, "cached": True})
            return

        performance_monitor.record_cache_hit("intent", False)

        resolved_model = self._resolve_model()
        self.ollama.model = resolved_model
        client = self.ollama

        # === NEW: Token Compactor - get tool schemas and compact context ===
        all_schemas = self.tools.get_filtered_schemas(user_input)

        # If IntentRouter suggested tools, filter to only those
        if intent.suggested_tools and all_schemas:
            all_schemas = [
                s for s in all_schemas if s.get("function", {}).get("name") in intent.suggested_tools
            ]

        tool_names = [s.get("function", {}).get("name", "unknown") for s in all_schemas]
        log.warning(f"[SERVER] Got {len(all_schemas)} tool schemas: {tool_names[:5]}{'...' if len(tool_names) > 5 else ''}")

        log.warning(
            f"[SERVER] Pre-compaction: {len(smart_messages)} smart msgs "
            f"({sum(1 for m in smart_messages if m.get('role')=='user')} user, "
            f"{sum(1 for m in smart_messages if m.get('role')=='assistant')} assistant"
            f"{', ' + str(len([m for m in smart_messages if m.get('role') not in ('user','assistant')])) + ' other' if any(m.get('role') not in ('user','assistant') for m in smart_messages) else ''}), "
            f"extra_context={len(extra_context)} chars"
        )

        compacted = self.token_compactor.prepare_context(
            query=user_input,
            conversation_messages=smart_messages,
            all_tool_schemas=all_schemas,
            system_prompt=SYSTEM_PROMPT,
            extra_context=extra_context,
        )
        tool_schemas: list[dict] = list(compacted.tool_schemas)

        compacted_messages = compacted.messages
        compacted_roles = [m.get("role", "?") for m in compacted_messages]
        log.warning(
            f"[SERVER] Compacted: {len(compacted_messages)} msgs "
            f"(user={compacted_roles.count('user')}, "
            f"assistant={compacted_roles.count('assistant')}"
            f"{', ' + str(len([r for r in compacted_roles if r not in ('user','assistant')])) + ' other' if any(r not in ('user','assistant') for r in compacted_roles) else ''}), "
            f"{len(compacted.tool_schemas)} tools, "
            f"{compacted.estimated_tokens} est tokens "
            f"({compacted.compression_ratio*100:.0f}% compression)"
        )

        system_prompt = compacted.system_prompt

        if broadcast_func:
            await broadcast_func({"type": "user_message", "content": user_input})

        # === SINGLE LLM PASS ===
        full_response = ""
        tool_calls = []
        thinking_text = ""

        async def broadcast_thinking_chunk(content: str):
            nonlocal thinking_text
            thinking_text += content
            if broadcast_func:
                await broadcast_func({"type": "thinking_chunk", "content": content})

        log.warning(f"[SERVER] Starting single LLM pass with {type(client).__name__}")
        # Fire TTS health check early � runs in parallel with LLM pass + tool execution
        tts_health_task = asyncio.create_task(self.tts.health_check())
        start_time = asyncio.get_event_loop().time()
        chunk_count = 0

        messages_for_llm = [{"role": "user", "content": user_input}] + compacted.messages
        async for chunk in client.chat(
            messages=messages_for_llm,
            system=system_prompt,
            tools=tool_schemas,
        ):
            chunk_count += 1
            # Ollama non-streaming format
            if msg := chunk.get("message", {}):
                if content := msg.get("content"):
                    full_response += content
                    if broadcast_func:
                        await broadcast_func({"type": "streaming_chunk", "content": content})
                if calls := msg.get("tool_calls"):
                    log.warning(f"[SERVER] Found {len(calls)} tool calls in chunk {chunk_count}")
                    self._accumulate_tool_calls(calls, tool_calls, chunk_count)
            # Ollama thinking format
            elif chunk.get("type") == "thinking":
                if thinking_content := chunk.get("content"):
                    await broadcast_thinking_chunk(thinking_content)
            elif chunk.get("type") == "tool_calls" and chunk.get("tool_calls"):
                calls = chunk["tool_calls"]
                self._accumulate_tool_calls(calls, tool_calls, chunk_count, source="sse")
            # OpenAI streaming format (OmniRoute, vLLM, etc.)
            elif choices := chunk.get("choices", []):
                choice = choices[0] if choices else {}
                # Streaming uses "delta", non-streaming uses "message"
                payload = choice.get("delta", {}) or choice.get("message", {})
                if reasoning := payload.get("reasoning", ""):
                    await broadcast_thinking_chunk(reasoning)
                if content := payload.get("content", ""):
                    log.warning(f"[SERVER] Content chunk: {content[:50]!r}")
                    full_response += content
                    if broadcast_func:
                        await broadcast_func({"type": "streaming_chunk", "content": content})
                if new_tool_calls := payload.get("tool_calls"):
                    self._accumulate_tool_calls(
                        new_tool_calls, tool_calls, chunk_count, source="sse"
                    )
                # Some providers put tool_calls at choice level (e.g. vLLM final chunk)
                if not payload.get("tool_calls") and choice.get("tool_calls"):
                    self._accumulate_tool_calls(
                        choice["tool_calls"], tool_calls, chunk_count, source="sse"
                    )
            elif content := chunk.get("content"):
                full_response += content
                if broadcast_func:
                    await broadcast_func({"type": "streaming_chunk", "content": content})

        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        performance_monitor.record_llm_response(duration_ms, True)
        log.warning(
            f"[SERVER] LLM pass done: {chunk_count} chunks, "
            f"tool_calls={len(tool_calls)}, response={len(full_response)} chars"
        )

        # === Parse embedded function calls from response ===
        if not tool_calls and full_response:
            xml_calls = self._parse_xml_function_calls(full_response)
            if xml_calls:
                tool_calls = xml_calls
                full_response = re.sub(
                    r"<function_calls>.*?</function_calls>", "", full_response, flags=re.DOTALL
                ).strip()

            if not tool_calls:
                nvidia_calls = self._parse_nvidia_tool_calls(full_response)
                if nvidia_calls:
                    tool_calls = nvidia_calls
                    full_response = re.sub(
                        r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
                        "",
                        full_response,
                        flags=re.DOTALL,
                    ).strip()

            if not tool_calls:
                simple_calls = self._parse_simple_tool_calls(full_response)
                if simple_calls:
                    tool_calls = simple_calls
                    full_response = re.sub(
                        r"<tool_call>.*?</tool_call>", "", full_response, flags=re.DOTALL
                    ).strip()

        # === TOOL EXECUTION - single pass, no second LLM call ===
        if tool_calls:
            tool_names = [c.get("function", {}).get("name") for c in tool_calls]
            log.warning(f"[SERVER] Tool calls detected: {tool_names}")

            # Filter and validate tool calls (keep existing validation logic)
            def try_fix_and_validate(call):
                fn = call.get("function", {})
                args = fn.get("arguments") or ""
                if args == "":
                    call["function"]["arguments"] = {}
                    return True
                if isinstance(args, dict):
                    return True  # Already parsed
                if not isinstance(args, str):
                    log.warning(f"[SERVER] Cannot fix non-string args: {type(args)}")
                    return False
                try:
                    call["function"]["arguments"] = json.loads(args)
                    return True
                except json.JSONDecodeError:
                    # Try to fix common issues with accumulated fragments
                    fixed = args.strip()

                    # Fix 1: Remove extra trailing braces/keys from concatenation
                    # e.g., '{"query": "x"}{' -> '{"query": "x"}'
                    brace_count = fixed.count("{") - fixed.count("}")
                    if brace_count > 0:
                        fixed = fixed + ("}" * brace_count)
                    elif brace_count < 0:
                        # Remove extra closing braces
                        fixed = fixed[: fixed.rfind("}") + 1] if "}" in fixed else fixed

                    # Fix 2: Try to parse again
                    try:
                        call["function"]["arguments"] = json.loads(fixed)
                        log.debug(
                            f"[SERVER] Fixed accumulated args for {call.get('id')}: "
                            f"{fixed[:100]}..."
                        )
                        return True
                    except json.JSONDecodeError:
                        pass

                    # Fix 3: Try to extract key-value pairs manually from malformed JSON
                    # e.g., '{"query":amazon.com server", "num_results": 10}'
                    try:
                        extracted = {}
                        # Match "key": value patterns, handling missing quotes around values
                        # Pattern: "key": followed by value until comma or }
                        pattern = r'"([^"]+)"\s*:\s*([^,\}]+)'
                        matches = re.findall(pattern, fixed)
                        for key, val in matches:
                            val = val.strip()
                            # Try to parse as JSON (for numbers, booleans, quoted strings)
                            try:
                                extracted[key] = json.loads(val)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, treat as string
                                extracted[key] = val.strip("\"' ")
                        if extracted and "query" in extracted:
                            call["function"]["arguments"] = extracted
                            log.warning(f"[SERVER] Extracted args manually: {extracted}")
                            return True
                    except Exception as extract_error:
                        log.debug(f"[SERVER] Manual extraction failed: {extract_error}")

                    log.warning(
                        f"[SERVER] Filtering out incomplete tool call {call.get('id')}: "
                        f"args not valid JSON"
                    )
                return False

            valid_tool_calls = [c for c in tool_calls if try_fix_and_validate(c)]

            self.messages.append({
                "role": "assistant", "content": full_response, "tool_calls": valid_tool_calls
            })
            await streaming_interface.push_assistant_message(full_response)
            await conversation_buffer.add_message({
                "role": "assistant", "content": full_response, "tool_calls": valid_tool_calls
            })

            tool_results = []
            if valid_tool_calls:
                tool_results = await self.process_tool_calls(valid_tool_calls)
                log.warning(f"[SERVER] Tool results: {len(tool_results)} results")

            for tr in tool_results:
                self.messages.append(tr)

            # Vision tools get special handling (keep existing logic)
            if "screenshot_analyze" in tool_names and tool_results:
                try:
                    vision_data = json.loads(tool_results[0].get("content", "{}"))
                    full_response = vision_data.get("analysis", "")
                    vision_screenshot_path = vision_data.get("screenshot_path")
                    log.warning(f"[SERVER] Using vision response: {len(full_response)} chars")
                    if broadcast_func:
                        await broadcast_func({
                            "type": "streaming_chunk", "content": full_response,
                            "replace": True, "screenshot_path": vision_screenshot_path,
                        })
                except Exception as e:
                    log.warning(f"[SERVER] Failed to extract vision response: {e}")

            if not ("screenshot_analyze" in tool_names and tool_results):
                log.warning(f"[SERVER] Starting second LLM pass to generate response from tool results")
                full_system_prompt = SYSTEM_PROMPT
                if extra_context:
                    full_system_prompt += "\n\n" + extra_context

                # Build minimal messages: current user input + assistant tool call + tool results only
                # Sending all of self.messages includes stale tool results from prior turns
                second_pass_messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": full_response, "tool_calls": valid_tool_calls},
                ] + tool_results
                log.warning(
                    f"[SERVER] Second pass messages: {len(second_pass_messages)} msgs "
                    f"(user + assistant tool_call + {len(tool_results)} tool results)"
                )

                second_response = ""
                second_chunks = 0
                # Include tool schemas so LLM can call additional tools
                tool_schemas = self.tools.get_filtered_schemas(user_input)
                async for chunk in client.chat(
                    messages=second_pass_messages, system=full_system_prompt,
                    tools=tool_schemas,
                ):
                    second_chunks += 1
                    if msg := chunk.get("message", {}):
                        if content := msg.get("content"):
                            second_response += content
                            if broadcast_func:
                                await broadcast_func({"type": "streaming_chunk", "content": content})
                    elif choices := chunk.get("choices", []):
                        choice = choices[0] if choices else {}
                        payload = choice.get("delta", {}) or choice.get("message", {})
                        if reasoning := payload.get("reasoning", ""):
                            await broadcast_thinking_chunk(reasoning)
                        if content := payload.get("content", ""):
                            second_response += content
                            if broadcast_func:
                                await broadcast_func({"type": "streaming_chunk", "content": content})
                    elif content := chunk.get("content"):
                        second_response += content
                        if broadcast_func:
                            await broadcast_func({"type": "streaming_chunk", "content": content})
                    elif chunk.get("type") == "thinking":
                        if thinking_content := chunk.get("content"):
                            await broadcast_thinking_chunk(thinking_content)

                if second_response:
                    full_response = second_response
                    log.warning(
                        f"[SERVER] Second LLM pass done: {second_chunks} chunks, "
                        f"response={len(second_response)} chars"
                    )
                else:
                    log.warning("[SERVER] Second LLM pass produced empty response")

        # === FINAL STEPS ===
        if not full_response and not tool_calls:
            log.warning("[SERVER] LLM returned empty — retrying without tools")
            retry_msgs = [{"role": "user", "content": user_input}] + compacted.messages
            async for chunk in client.chat(
                messages=retry_msgs,
                system=system_prompt,
                tools=None,
            ):
                if choices := chunk.get("choices", []):
                    choice = choices[0] if choices else {}
                    payload = choice.get("delta", {}) or choice.get("message", {})
                    if content := payload.get("content", ""):
                        full_response += content
                        if broadcast_func:
                            await broadcast_func({"type": "streaming_chunk", "content": content})

        if thinking_text and broadcast_func:
            await broadcast_func({"type": "thinking_complete", "content": thinking_text})

        clean_response = re.sub(
            r"<thinking>.*?</thinking>", "", full_response, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        if clean_response.strip():
            self.messages.append({"role": "assistant", "content": clean_response})
            await streaming_interface.push_assistant_message(clean_response)
            asyncio.create_task(conversation_buffer.add_message({"role": "assistant", "content": clean_response}))
            try:
                ctx_manager = get_context_manager()
                asyncio.create_task(ctx_manager.add_to_memory("user", user_input))
                asyncio.create_task(ctx_manager.add_to_memory("assistant", clean_response))
            except Exception as e:
                log.debug(f"Failed to store in semantic memory: {e}")
        else:
            log.warning("[SERVER] Skipping empty assistant message append")

        if broadcast_func:
            await broadcast_func({
                "type": "message_complete",
                "full_response": clean_response,
                "screenshot_path": vision_screenshot_path,
            })

        if should_cache_response(user_input, clean_response):
            asyncio.create_task(intent_cache.set(cache_key, clean_response))

        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages :]

        if full_response:
            try:
                # Don't block on TTS health -- wait at most 1s, then skip
                done, _ = await asyncio.wait(
                    [tts_health_task], timeout=1.0,
                )
                if done and tts_health_task.result():
                    asyncio.create_task(self.tts.play_stream(full_response))
            except Exception as e:
                log.warning(f"[SERVER] TTS unavailable (non-fatal): {e}")

    async def process_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls in parallel"""

        async def execute_single_tool(call: dict) -> dict:
            # Initialize all local variables at the start to avoid UnboundLocalError
            args = {}
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
            if raw_args is None:
                args = {}
            elif isinstance(raw_args, str):
                args = raw_args
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            tool_call_id = call.get("id", "")

            # Skip invalid tool calls
            if not name:
                log.warning("[SERVER] Skipping tool call with no name")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id or "unknown",
                    "content": json.dumps({"error": "Tool call has no name"}),
                }

            if isinstance(args, dict):
                # Already parsed, use as-is
                log.debug(f"[SERVER] Using parsed dict args: {args}")
                pass
            elif isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as e:
                    log.warning(
                        f"[SERVER] Failed to parse tool arguments JSON: {e}, args: {repr(args)}"
                    )
                    # Try to fix common LLM JSON errors
                    fixed_args = args.strip()
                    # Keep for logging
                    # Remove surrounding single quotes if present
                    if fixed_args.startswith("'") and fixed_args.endswith("'"):
                        fixed_args = fixed_args[1:-1]
                    # Handle various malformed JSON patterns from streaming LLM responses
                    # Pattern 1: Incomplete object like '{"query":' (has key, no value, no closing)
                    if fixed_args.startswith("{") and not fixed_args.endswith("}"):
                        # Extract what we have and complete it
                        # e.g., '{"query":' -> need to add a value and close
                        if fixed_args.rstrip().endswith(":"):
                            # Has key but no value - need to complete based on tool
                            if name == "web_search":
                                # '{"query":' -> '{"query": ""}'
                                fixed_args = fixed_args.rstrip() + ' ""}'
                            elif name == "launch_app":
                                fixed_args = fixed_args.rstrip() + ' ""}'
                            elif name == "open_url":
                                fixed_args = fixed_args.rstrip() + ' ""}'
                            else:
                                # Generic: add empty value
                                fixed_args = fixed_args.rstrip() + " null}"
                            log.warning(
                                f"[SERVER] Fixed incomplete JSON by adding value: {fixed_args}"
                            )

                    # Pattern 2: Partial object missing opening brace
                    # e.g., '"num_results": 10}' or 'query": "weather", "num_results": 10}'
                    elif not fixed_args.startswith("{") and not fixed_args.startswith("["):
                        if fixed_args.endswith("}") and ":" in fixed_args:
                            # Try prepending '{'
                            try:
                                test_fixed = "{" + fixed_args
                                json.loads(test_fixed)  # Validate
                                fixed_args = test_fixed
                                log.warning(
                                    f"[SERVER] Fixed partial JSON by prepending '{{': {fixed_args}"
                                )
                            except json.JSONDecodeError:
                                pass

                    # Try wrapping bare values in an object based on tool name
                    # (if still not valid JSON)
                    if not fixed_args.startswith("{") and not fixed_args.startswith("["):
                        if name == "launch_app":
                            fixed_args = '{"app_name": ' + json.dumps(fixed_args) + "}"
                        elif name == "web_search":
                            fixed_args = '{"query": ' + json.dumps(fixed_args) + "}"
                        elif name == "open_url":
                            fixed_args = '{"url": ' + json.dumps(fixed_args) + "}"
                        else:
                            # Generic fallback - assume it's a single string argument
                            fixed_args = '{"value": ' + json.dumps(fixed_args) + "}"

                    # Final attempt to parse
                    try:
                        args = json.loads(fixed_args)
                    except json.JSONDecodeError:
                        # Last resort: try to extract key-value pairs manually
                        extracted = {}
                        # Match "key": value or "key": "value" patterns
                        pattern = r'"([^"]+)"\s*:\s*([^,"\}]+|"[^"]*")'
                        matches = re.findall(pattern, fixed_args)
                        for key, val in matches:
                            # Parse the value
                            val = val.strip()
                            try:
                                extracted[key] = json.loads(val)
                            except json.JSONDecodeError:
                                extracted[key] = val.strip('"')

                        if extracted:
                            args = extracted
                            log.warning(f"[SERVER] Extracted args manually: {args}")
                        else:
                            raise  # Re-raise if we couldn't extract anything

                    # For web_search, ensure 'query' parameter exists
                    if name == "web_search" and "query" not in args:
                        log.error(f"[SERVER] web_search missing required 'query' parameter: {args}")
                        args = {"error": "Missing required 'query' parameter"}

                    log.warning(f"[SERVER] Recovered args after fix: {args}")
                except Exception as e:
                    log.error(
                        "[SERVER] Could not fix malformed args: {}, original: {}".format(
                            str(e), repr(args)
                        )
                    )
                    args = {}

            # Ensure args is a dict, not None
            if args is None:
                args = {}
            log.warning(f"[SERVER] Executing tool: {name} with args: {args}")
            try:
                result = await self.tools.execute(name, **args)
                log.warning(
                    f"[SERVER] Tool {name} result: success={result.success}"
                    f"data_len={len(str(result.data)) if result.data else 0}"
                    f"error={result.error}"
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
            *[execute_single_tool(call) for call in valid_tool_calls],
            return_exceptions=True,
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

    def _parse_xml_function_calls(self, content: str) -> list[dict]:
        """Parse XML-style function calls from LLM response content.

        Handles format like:
        <function_calls>
        <invoke name="web_search">
        <parameter name="query">search terms</parameter>
        </invoke>
        </function_calls>
        """
        import xml.etree.ElementTree as ET

        calls = []

        # Find function_calls blocks
        func_calls_pattern = re.compile(
            r"<function_calls>(.*?)</function_calls>", re.DOTALL | re.IGNORECASE
        )

        for match in func_calls_pattern.finditer(content):
            xml_block = match.group(1)

            # Wrap in root for parsing
            try:
                root = ET.fromstring(f"<root>{xml_block}</root>")

                for invoke in root.findall(".//invoke"):
                    name = invoke.get("name")
                    if not name:
                        continue

                    # Build arguments dict from parameters
                    args = {}
                    for param in invoke.findall("parameter"):
                        param_name = param.get("name")
                        param_value = param.text if param.text else ""
                        if param_name:
                            args[param_name] = param_value

                    # Create tool call structure
                    call = {
                        "id": f"xml_{len(calls)}_{hash(content) % 10000}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args) if args else "{}",
                        },
                    }
                    calls.append(call)
                    log.debug(f"[SERVER] Parsed XML function call: {name} with args: {args}")

            except ET.ParseError as e:
                log.debug(f"[SERVER] Failed to parse XML function calls: {e}")
                # Fallback: try regex extraction
                calls.extend(self._extract_xml_calls_with_regex(match.group(1)))

        return calls

    def _parse_nvidia_tool_calls(self, content: str) -> list[dict]:
        """Parse NVIDIA-style tool calls from LLM response content.

        Handles format like:
        <|tool_calls_section_begin|>
        <|tool_call_begin|>  functions.web_search:3
        <|tool_call_argument_begin|> {"query": "...", "num_results": 5}
        <|tool_call_end|>
        <|tool_call_begin|>  functions.web_search:4
        <|tool_call_argument_begin|> {"query": "...", "num_results": 5}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """
        calls = []

        # Check if this is a tool call section
        if "<|tool_calls_section_begin|>" not in content:
            return calls

        # Extract the tool calls section
        section_start = content.find("<|tool_calls_section_begin|>")
        section_end = content.find("<|tool_calls_section_end|>")

        if section_start == -1 or section_end == -1:
            return calls

        section_content = content[section_start:section_end]

        # Find all individual tool calls
        tool_call_pattern = re.compile(
            r"<\|tool_call_begin\|>\s*functions\.(\w+):(\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        for match in tool_call_pattern.finditer(section_content):
            name = match.group(1)
            call_id = match.group(2)
            args_str = match.group(3)

            # Create tool call with raw arguments - let downstream validation handle parsing
            call = {
                "id": f"nvidia_{call_id}_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str,  # Pass raw string, let process_tool_calls handle parsing
                },
            }
            calls.append(call)
            log.warning(
                f"[SERVER] Parsed NVIDIA tool call: {name}:{call_id} with raw args: {args_str[:100]}"
            )

        return calls

    def _parse_simple_tool_calls(self, content: str) -> list[dict]:
        """Parse simple <function=tool_name> format from content."""
        calls = []

        # Match <function=tool_name> or <function=tool_name(arg1=value1, arg2=value2)>
        pattern = re.compile(
            r"<function=(\w+)(?:\(([^)]*)\))?>",
            re.DOTALL,
        )

        for match in pattern.finditer(content):
            name = match.group(1)
            args_str = match.group(2)

            args = {}
            if args_str:
                try:
                    for pair in args_str.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            args[key.strip()] = value.strip().strip('"').strip("'")
                except Exception:
                    pass

            call = {
                "id": f"simple_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if args else "{}",
                },
            }
            calls.append(call)
            log.warning(f"[SERVER] Parsed simple tool call: {name} with args: {args}")

        if not calls:
            calls = self._parse_tool_call_xml_format(content)

        return calls

    def _parse_tool_call_xml_format(self, content: str) -> list[dict]:
        calls = []

        tool_call_pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>",
            re.DOTALL | re.IGNORECASE,
        )

        for tc_match in tool_call_pattern.finditer(content):
            tc_content = tc_match.group(1)

            func_pattern = re.compile(
                r"<function=(\w+)>",
                re.DOTALL,
            )
            func_match = func_pattern.search(tc_content)
            if not func_match:
                continue

            name = func_match.group(1)

            args = {}
            param_pattern = re.compile(
                r"<parameter=(\w+)>([^<]*)",
                re.DOTALL,
            )

            for param_match in param_pattern.finditer(tc_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, TypeError):
                    args[param_name] = param_value

            call = {
                "id": f"toolcall_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if args else "{}",
                },
            }
            calls.append(call)
            log.warning(f"[SERVER] Parsed tool_call XML: {name} with args: {args}")

        return calls

    def _extract_xml_calls_with_regex(self, xml_content: str) -> list[dict]:
        """Fallback regex-based extraction for malformed XML."""
        calls = []

        # Pattern to match invoke tags
        invoke_pattern = re.compile(
            r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE
        )

        for match in invoke_pattern.finditer(xml_content):
            name = match.group(1)
            inner = match.group(2)

            # Extract parameters
            args = {}
            param_pattern = re.compile(
                r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>',
                re.DOTALL | re.IGNORECASE,
            )
            for param_match in param_pattern.finditer(inner):
                param_name = param_match.group(1)
                param_value = param_match.group(2)
                args[param_name] = param_value.strip()

            call = {
                "id": f"xml_regex_{len(calls)}_{hash(xml_content) % 10000}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if args else "{}",
                },
            }
            calls.append(call)
            log.debug(f"[SERVER] Parsed XML function call (regex): {name}")

        return calls

    def _accumulate_tool_calls(
        self,
        new_calls: list,
        tool_calls: list,
        chunk_count: int,
        source: str = "message",
    ) -> None:
        """Accumulate tool call fragments from streaming chunks.

        NVIDIA API streams tool calls with arguments split across chunks.
        We accumulate by index and concatenate argument fragments.
        """
        log.warning(
            f"[SERVER] _accumulate_tool_calls called: {len(new_calls)} new calls, "
            f"current total {len(tool_calls)}, source={source}"
        )
        for i, call in enumerate(new_calls):
            args_val = call.get("function", {}).get("arguments")
            args_repr = repr(str(args_val)[:50]) if args_val is not None else "None"
            log.warning(
                f"[SERVER] Processing new call {i}: id={call.get('id')}, "
                f"name={call.get('function', {}).get('name')}, "
                f"args={args_repr}"
            )
        for call in new_calls:
            idx = call.get("index", 0)
            call_id = call.get("id")
            fn = call.get("function", {})
            name = fn.get("name")
            args_fragment = fn.get("arguments")

            if isinstance(args_fragment, dict):
                args_fragment = json.dumps(args_fragment)
            elif not isinstance(args_fragment, str):
                args_fragment = str(args_fragment) if args_fragment else ""

            # Find existing call by id (preferred) or index (fallback)
            existing = None
            for tc in tool_calls:
                if call_id and tc.get("id") == call_id:
                    existing = tc
                    break
                if not existing and tc.get("index") == idx:
                    existing = tc

            if existing:
                if args_fragment:
                    existing_args = existing.get("function", {}).get("arguments")
                    if isinstance(existing_args, dict):
                        existing_args = json.dumps(existing_args)
                    elif not isinstance(existing_args, str):
                        existing_args = str(existing_args) if existing_args else ""

                    if isinstance(args_fragment, dict):
                        args_fragment = json.dumps(args_fragment)
                    elif not isinstance(args_fragment, str):
                        args_fragment = str(args_fragment) if args_fragment else ""

                    combined_args = existing_args + args_fragment
                    existing["function"]["arguments"] = combined_args
                    log.warning(
                        f"[SERVER] Appended args for index {idx}: "
                        f"+{len(args_fragment)} chars, total {len(combined_args)}"
                    )

                # Update ID if we got a real one (previously null)
                if call_id and not existing.get("id"):
                    existing["id"] = call_id
                    log.warning(f"[SERVER] Updated call ID for index {idx}: {call_id}")

                # Update name if we got one (previously null)
                if name and not existing.get("function", {}).get("name"):
                    existing["function"]["name"] = name
                    log.warning(f"[SERVER] Updated function name for index {idx}: {name}")
            else:
                # New tool call - make sure it has an index
                log.warning(f"[SERVER] ELSE BRANCH: adding new call at index {idx}")
                call["index"] = idx
                tool_calls.append(call)
                log.warning(
                    f"[SERVER] Added new tool call index {idx}: name={name}, id={call_id}, "
                    f"list len before={len(tool_calls) - 1}, after={len(tool_calls)}"
                )

        log.warning(
            f"[SERVER] Tool calls from {source} in chunk {chunk_count}: "
            f"{len(new_calls)} new, total {len(tool_calls)} accumulated"
        )

    def set_model(self, model: str) -> None:
        """Set the selected model"""
        self._selected_model = (model, model.upper())
