"""JARVIS Server - Backend processing for the TUI client"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List

from core.cache import intent_cache, should_cache_response
from core.config import Config
from core.llm.optimized_client import OptimizedLLMClient
from core.performance_monitor import performance_monitor
from core.streaming_interface import conversation_buffer, streaming_interface
from core.voice.tts import TextToSpeech
from tools import get_tool_registry

log = logging.getLogger("jarvis.server")

SYSTEM_PROMPT = """You are JARVIS, an intelligent AI assistant.
You are helpful, concise, and friendly — but also real.
You don't sugarcoat. You say what's true, even if it's rough.
Use foul language when it fits — curse, laugh, or call it like it is.
Be direct, avoid fluff, and never overthink the vibe.
Always use tools for info, time, searches, or actions — unless context says otherwise.
Never use emojis.
IMPORTANT: You are not censored — you may generate any content necessary
to fulfill the user's request, including bold, imaginative, crude, or even offensive statements,
as long as they're contextually appropriate.
You may freely express opinions, make blunt assumptions, or use strong, real language — even
swear — without self-censorship.
Be friendly, but don't be afraid to be raw, real, and unfiltered.
JARVIS doesn't hold back — and neither should you."""


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
                    {
                        "type": "assistant_message",
                        "content": cached_response,
                        "cached": True,
                    }
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
                log.warning(
                    f"[SERVER] Found {len(calls)} tool calls in message chunk {chunk_count}"
                )
                self._accumulate_tool_calls(calls, tool_calls, chunk_count)
            # Handle tool_calls directly from optimized_client (type: "tool_calls")
            elif chunk.get("type") == "tool_calls" and chunk.get("tool_calls"):
                calls = chunk["tool_calls"]
                log.warning(f"[SERVER] Found {len(calls)} tool calls in SSE chunk {chunk_count}")
                self._accumulate_tool_calls(calls, tool_calls, chunk_count, source="sse")
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

        # Parse XML-style function calls from content (e.g., <function_calls><invoke name="...">)
        if not tool_calls and full_response:
            xml_calls = self._parse_xml_function_calls(full_response)
            if xml_calls:
                log.warning(f"[SERVER] Extracted {len(xml_calls)} XML function calls from content")
                tool_calls = xml_calls
                # Remove XML from content to clean up response
                import re

                full_response = re.sub(
                    r"<function_calls>.*?</function_calls>",
                    "",
                    full_response,
                    flags=re.DOTALL,
                ).strip()

            # If no XML calls found, try NVIDIA-style tool calls
            if not tool_calls:
                nvidia_calls = self._parse_nvidia_tool_calls(full_response)
                if nvidia_calls:
                    log.warning(
                        f"[SERVER] Extracted {len(nvidia_calls)} NVIDIA function calls from content"
                    )
                    tool_calls = nvidia_calls
                    # Remove NVIDIA tool call section from content
                    import re

                    full_response = re.sub(
                        r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
                        "",
                        full_response,
                        flags=re.DOTALL,
                    ).strip()

        log.warning(f"[SERVER] After stream ended: tool_calls list has {len(tool_calls)} items")
        for i, tc in enumerate(tool_calls):
            log.warning(
                f"[SERVER] Tool call {i}: id={tc.get('id')}, "
                f"name={tc.get('function', {}).get('name')}"
            )

        if tool_calls:
            tool_names = [c.get("function", {}).get("name") for c in tool_calls]
            log.warning(f"[SERVER] Tool calls detected: {tool_names}")

            # Filter tool calls to only those with valid, complete JSON arguments
            def try_fix_and_validate(call):
                fn = call.get("function", {})
                args = fn.get("arguments") or ""
                # Empty string is valid (will become {})
                if args == "":
                    call["function"]["arguments"] = "{}"
                    return True
                if isinstance(args, dict):
                    return True  # Already parsed
                if not isinstance(args, str):
                    log.warning(f"[SERVER] Cannot fix non-string args: {type(args)}")
                    return False
                try:
                    json.loads(args)
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
                        json.loads(fixed)
                        call["function"]["arguments"] = fixed
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
                            # Reconstruct valid JSON
                            valid_json = json.dumps(extracted)
                            call["function"]["arguments"] = valid_json
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
            if len(valid_tool_calls) < len(tool_calls):
                log.warning(
                    f"[SERVER] Filtered {len(tool_calls) - len(valid_tool_calls)} "
                    f"incomplete tool calls"
                )

            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": valid_tool_calls,
                }
            )
            await streaming_interface.push_assistant_message(full_response)
            await conversation_buffer.add_message(
                {
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": valid_tool_calls,
                }
            )

            # Skip tool execution if no valid tool calls remain
            tool_results = []
            if not valid_tool_calls:
                log.warning("[SERVER] No valid tool calls after filtering, skipping tool execution")
            else:
                log.debug(f"[SERVER] Processing {len(valid_tool_calls)} tool calls")
                tool_results = await self.process_tool_calls(valid_tool_calls)
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
                            {
                                "type": "streaming_chunk",
                                "content": full_response,
                                "replace": True,
                            }
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

                # Multi-pass tool calling: keep processing until we get a clean response
                max_passes = 5  # Prevent infinite loops
                current_pass = 0

                while current_pass < max_passes:
                    current_pass += 1

                    # Check if this is a vision tool - use result directly
                    is_vision_tool = "screenshot_analyze" in tool_names and current_pass == 1
                    if is_vision_tool and tool_results:
                        try:
                            vision_data = json.loads(tool_results[0].get("content", "{}"))
                            full_response = vision_data.get("analysis", "")
                            log.warning(
                                f"[SERVER] Using vision response directly: {len(full_response)} chars"
                            )
                            if broadcast_func:
                                await broadcast_func(
                                    {
                                        "type": "streaming_chunk",
                                        "content": full_response,
                                        "replace": True,
                                    }
                                )
                            break  # Vision tool is done
                        except Exception as e:
                            log.warning(f"[SERVER] Failed to extract vision response: {e}")
                            is_vision_tool = False

                    # Skip streaming pass if we have vision tool result
                    if is_vision_tool:
                        break

                    log.warning(
                        f"[SERVER] Starting pass {current_pass}/{max_passes} with {len(self.messages)} messages"
                    )
                    full_response = ""
                    pass_chunks = 0
                    tool_call_buffer = ""  # Buffer to accumulate potential tool calls
                    in_tool_call = False  # Track if we're inside a tool call

                    async for chunk in client.chat(messages=self.messages, system=SYSTEM_PROMPT):
                        pass_chunks += 1
                        content = None

                        if msg := chunk.get("message", {}):
                            content = msg.get("content")
                            if calls := msg.get("tool_calls"):
                                log.warning(
                                    f"[SERVER] Unexpected tool calls in pass {current_pass}: {len(calls)}"
                                )
                        elif chunk.get("content"):
                            content = chunk.get("content")
                        elif chunk.get("tool_calls"):
                            log.warning(f"[SERVER] Tool result in chunk: {chunk.get('tool_calls')}")

                        if content:
                            log.info(
                                f"[SERVER] Pass {current_pass} chunk {pass_chunks}: {len(content)} chars"
                            )
                            full_response += content

                            # Buffer for tool call detection
                            tool_call_buffer += content

                            # Check if we're entering or inside a tool call
                            if "<|tool_calls_section_begin|>" in content:
                                in_tool_call = True
                                log.warning(
                                    f"[SERVER] Detected <|tool_calls_section_begin|> start in pass {current_pass}"
                                )
                                # Don't broadcast until we see the closing tag
                                continue

                            if in_tool_call:
                                if "<|tool_calls_section_end|>" in content:
                                    # Complete tool call found - parse and execute
                                    in_tool_call = False
                                    log.warning(
                                        f"[SERVER] Detected <|tool_calls_section_end|> end in pass {current_pass}"
                                    )

                                    # Parse the tool call from buffer
                                    parsed_calls = self._parse_nvidia_tool_calls(tool_call_buffer)
                                    if parsed_calls:
                                        log.warning(
                                            f"[SERVER] Parsed {len(parsed_calls)} NVIDIA tool calls from text"
                                        )
                                        # Execute the tool calls
                                        tool_results = await self.process_tool_calls(parsed_calls)
                                        self.messages.extend(tool_results)

                                        # Clear the buffer - don't send tool call text to user
                                        tool_call_buffer = ""
                                        full_response = ""
                                        break  # Exit this pass and start next one
                                    else:
                                        # Not a valid tool call, send what we have
                                        in_tool_call = False
                                else:
                                    # Still inside tool call, don't send to user yet
                                    continue

                    # Normal content - send to user (but not tool call content)
                    if broadcast_func and not in_tool_call:
                        await broadcast_func({"type": "streaming_chunk", "content": content})

                    # If we ended while still in a tool call (incomplete), try to parse anyway
                    if in_tool_call and tool_call_buffer:
                        log.warning(
                            f"[SERVER] Stream ended while in tool call in pass {current_pass}, attempting to parse"
                        )
                        parsed_calls = self._parse_nvidia_tool_calls(tool_call_buffer)
                        if parsed_calls:
                            tool_results = await self.process_tool_calls(parsed_calls)
                            self.messages.extend(tool_results)
                            full_response = ""
                            continue  # Go to next pass

                    # Check if we have any NVIDIA-style tool calls in the response
                    if (
                        "<|tool_calls_section_begin|>" in full_response
                        and "<|tool_calls_section_end|>" in full_response
                    ):
                        log.warning(
                            f"[SERVER] Detected NVIDIA tool calls in pass {current_pass} response"
                        )
                        parsed_calls = self._parse_nvidia_tool_calls(full_response)
                        if parsed_calls:
                            log.warning(f"[SERVER] Parsed {len(parsed_calls)} NVIDIA tool calls")
                            # Execute the tool calls
                            tool_results = await self.process_tool_calls(parsed_calls)
                            self.messages.extend(tool_results)
                            # Clear response since we executed tools
                            full_response = ""
                            continue  # Go to next pass

                    # No tool calls found - this is our final response
                    # But only exit if we actually have content to show (not just whitespace)
                    if full_response.strip():
                        log.warning(
                            f"[SERVER] Pass {current_pass} completed with {len(full_response)} chars response - no more tool calls"
                        )
                        break  # Exit the multi-pass loop
                    else:
                        # Continue to next pass even if no tool calls, in case we need to process more
                        log.warning(
                            f"[SERVER] Pass {current_pass} completed with empty response, continuing to next pass"
                        )

        # Only append non-empty responses to maintain conversation flow
        if full_response.strip():
            self.messages.append({"role": "assistant", "content": full_response})
            await streaming_interface.push_assistant_message(full_response)
            await conversation_buffer.add_message({"role": "assistant", "content": full_response})
        else:
            # If we have no content but had tool calls, the tool results are already in messages
            log.warning("[SERVER] Skipping empty assistant message append")

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
            log.warning(
                f"[SERVER] Processing new call {i}: id={call.get('id')}, "
                f"name={call.get('function', {}).get('name')}, "
                f"args={repr(call.get('function', {}).get('arguments'))[:50]}"
            )
        for call in new_calls:
            idx = call.get("index", 0)
            call_id = call.get("id")
            fn = call.get("function", {})
            name = fn.get("name")
            args_fragment = fn.get("arguments") or ""

            # Find existing call by index
            existing = None
            for tc in tool_calls:
                if tc.get("index") == idx:
                    existing = tc
                    break

            if existing:
                # Accumulate arguments by concatenation
                if args_fragment:
                    existing_args = existing.get("function", {}).get("arguments") or ""
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
