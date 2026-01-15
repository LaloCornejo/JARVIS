"""
Optimized LLM client with advanced streaming capabilities.

This module provides enhanced LLM integration with intelligent context
management, response streaming, and performance optimizations for real-time
conversational AI.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, AsyncIterator, Dict, List

import httpx

log = logging.getLogger(__name__)


class ContextSummarizer:
    """Intelligent context summarization for LLM optimization"""

    def __init__(self):
        self.summary_cache = {}
        self.max_summary_length = 500

    async def summarize(self, messages: List[Dict[str, Any]], max_length: int = None) -> str:
        """Summarize conversation history for context optimization"""
        if not messages:
            return ""

        if max_length is None:
            max_length = self.max_summary_length

        # Create cache key from message contents
        content_hash = hashlib.md5(
            json.dumps([msg.get("content", "") for msg in messages], sort_keys=True).encode()
        ).hexdigest()

        if content_hash in self.summary_cache:
            return self.summary_cache[content_hash]

        try:
            # Simple extractive summarization
            important_messages = []

            # Keep system messages
            for msg in messages:
                if msg.get("role") == "system":
                    important_messages.append(msg)

            # Keep recent user messages with questions or commands
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            for msg in user_messages[-3:]:  # Last 3 user messages
                content = msg.get("content", "").lower()
                if any(
                    keyword in content
                    for keyword in ["what", "how", "why", "when", "where", "can you", "please"]
                ):
                    important_messages.append(msg)

            # Keep assistant responses to important user messages
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            important_messages.extend(assistant_messages[-2:])  # Last 2 assistant responses

            # Create summary
            summary_parts = []
            for msg in important_messages[:5]:  # Limit to 5 key messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate long messages
                summary_parts.append(f"{role}: {content}")

            summary = " | ".join(summary_parts)

            # Cache the summary
            self.summary_cache[content_hash] = summary

            return summary

        except Exception as e:
            log.warning(f"Context summarization failed: {e}")
            return "Previous conversation context"


class OptimizedOllamaClient:
    """Optimized LLM client with advanced streaming capabilities"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:1.7b",
        timeout: float = 300.0,
        num_ctx: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx or 16384  # Default context window
        self._client: httpx.AsyncClient | None = None
        self._response_cache = {}
        self._context_summarizer = ContextSummarizer()
        self._cache_max_size = 100  # Maximum cached responses

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with optimizations"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.timeout, connect=5.0, read=self.timeout, write=10.0, pool=5.0
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0
                ),
                http2=True,  # Enable HTTP/2 for better streaming
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat_streaming(
        self,
        messages: list,
        system: str = None,
        tools: list = None,
        stream_callback: callable = None,
        context_window: int = None,
        enable_caching: bool = True,
    ) -> AsyncIterator[dict]:
        """Advanced streaming chat with intelligent chunking and context optimization

        Args:
            messages: List of conversation messages
            system: System prompt
            tools: Available tools for function calling
            stream_callback: Optional callback for real-time processing
            context_window: Maximum context window size
            enable_caching: Whether to use response caching

        Yields:
            dict: Streaming response chunks with metadata
        """
        # Optimize context window
        optimized_messages = await self._optimize_context(messages, context_window)

        # Check cache for similar queries if enabled
        if enable_caching:
            cache_key = self._generate_cache_key(optimized_messages, system, tools)
            if cached_response := self._response_cache.get(cache_key):
                async for chunk in self._stream_cached_response(cached_response, stream_callback):
                    yield chunk
                return

        # Prepare streaming request
        request_data = {
            "model": self.model,
            "messages": optimized_messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": -1,  # Unlimited
                "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                "num_ctx": context_window or self.num_ctx,
                "num_thread": -1,  # Use all available threads
                "repeat_penalty": 1.1,
                "repeat_last_n": 64,
            },
        }

        if system:
            request_data["system"] = system
        if tools:
            request_data["tools"] = tools

        client = await self._get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                full_response = ""
                buffer = ""
                chunk_count = 0

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        if "error" in data:
                            yield {"type": "error", "error": data["error"]}
                            return

                        if "message" in data:
                            message = data["message"]

                            # Handle content streaming
                            if "content" in message and message["content"]:
                                content = message["content"]
                                buffer += content
                                chunk_count += 1

                                # Intelligent chunking based on content analysis
                                should_yield = self._should_yield_chunk(
                                    buffer, chunk_count, content
                                )

                                if should_yield:
                                    chunk_data = {
                                        "type": "content",
                                        "content": buffer,
                                        "partial": True,
                                        "chunk_id": chunk_count,
                                        "timestamp": asyncio.get_event_loop().time(),
                                    }

                                    if stream_callback:
                                        await stream_callback(chunk_data)

                                    yield chunk_data
                                    full_response += buffer
                                    buffer = ""
                            elif "tool_calls" in message and message["tool_calls"]:
                                # Handle tool calls
                                yield {
                                    "type": "tool_calls",
                                    "tool_calls": message["tool_calls"],
                                    "partial": False,
                                    "timestamp": asyncio.get_event_loop().time(),
                                }

                    except json.JSONDecodeError as e:
                        log.debug(f"Failed to parse streaming response: {e}")
                        continue

                # Yield final content if any remains
                if buffer:
                    final_chunk = {
                        "type": "content",
                        "content": buffer,
                        "partial": False,
                        "final": True,
                        "timestamp": asyncio.get_event_loop().time(),
                    }

                    if stream_callback:
                        await stream_callback(final_chunk)

                    yield final_chunk
                    full_response += buffer

                # Cache the full response if caching is enabled
                if enable_caching:
                    self._cache_response(cache_key, full_response)

        except Exception as e:
            log.error(f"Streaming chat error: {e}")
            yield {"type": "error", "error": str(e)}

    def _should_yield_chunk(self, buffer: str, chunk_count: int, new_content: str) -> bool:
        """Intelligent chunking decision based on content analysis"""
        buffer_length = len(buffer)

        # Always yield at sentence boundaries
        if new_content.endswith((".", "!", "?", ":", ";")):
            return True

        # Yield based on buffer size with adaptive thresholds
        if chunk_count < 3:  # Early chunks
            return buffer_length >= 50
        elif chunk_count < 10:  # Middle chunks
            return buffer_length >= 100
        else:  # Later chunks
            return buffer_length >= 150

    async def _optimize_context(self, messages: list, max_context: int = None) -> list:
        """Intelligent context optimization with summarization"""
        if not max_context:
            max_context = self.num_ctx

        total_tokens = sum(len(msg.get("content", "")) for msg in messages)

        # If within context limit, return as-is
        if total_tokens <= max_context * 0.8:  # 80% threshold
            return messages

        try:
            # Separate message types
            system_msg = None
            user_messages = []
            assistant_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_msg = msg
                elif msg.get("role") == "user":
                    user_messages.append(msg)
                elif msg.get("role") == "assistant":
                    assistant_messages.append(msg)

            # Always keep system message
            optimized = [system_msg] if system_msg else []

            # Keep recent conversation (last 3 exchanges)
            recent_pairs = min(3, len(user_messages))
            recent_user = user_messages[-recent_pairs:]
            recent_assistant = assistant_messages[-recent_pairs:]

            # If we have too many messages, summarize older ones
            remaining_slots = max_context - sum(len(msg.get("content", "")) for msg in optimized)

            if len(recent_user) > 0:
                # Add recent messages
                for i in range(len(recent_user)):
                    if remaining_slots > 0:
                        optimized.append(recent_user[i])
                        remaining_slots -= len(recent_user[i].get("content", ""))

                        if i < len(recent_assistant):
                            optimized.append(recent_assistant[i])
                            remaining_slots -= len(recent_assistant[i].get("content", ""))

                # If still have space, add older summarized context
                if remaining_slots > 500 and len(user_messages) > recent_pairs:
                    older_messages = user_messages[:-recent_pairs]
                    if older_messages:
                        summary = await self._context_summarizer.summarize(
                            older_messages, remaining_slots // 2
                        )
                        if summary:
                            optimized.insert(
                                1, {"role": "system", "content": f"[Context Summary] {summary}"}
                            )

            return optimized

        except Exception as e:
            log.warning(f"Context optimization failed: {e}")
            return messages  # Return original if optimization fails

    def _generate_cache_key(self, messages: list, system: str = None, tools: list = None) -> str:
        """Generate cache key for response caching"""
        key_components = {
            "messages": [
                {"role": msg.get("role"), "content": msg.get("content", "")[:200]}
                for msg in messages
            ],
            "system": system[:200] if system else None,
            "tools": tools if tools else None,
            "model": self.model,
        }

        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_response(self, key: str, response: str):
        """Cache response with LRU eviction"""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]

        self._response_cache[key] = response

    async def _stream_cached_response(
        self, cached_response: str, callback: callable = None
    ) -> AsyncIterator[dict]:
        """Stream cached response with proper chunking"""
        # Simulate streaming by chunking the cached response
        words = cached_response.split()
        buffer = ""

        for i, word in enumerate(words):
            buffer += word + " "

            # Yield chunks at sentence boundaries or every few words
            if word.endswith((".", "!", "?", ":", ";")) or (i + 1) % 10 == 0:
                chunk_data = {
                    "type": "content",
                    "content": buffer,
                    "partial": True,
                    "cached": True,
                    "timestamp": asyncio.get_event_loop().time(),
                }

                if callback:
                    await callback(chunk_data)

                yield chunk_data
                buffer = ""

        # Yield final chunk
        if buffer:
            final_chunk = {
                "type": "content",
                "content": buffer,
                "partial": False,
                "final": True,
                "cached": True,
                "timestamp": asyncio.get_event_loop().time(),
            }

            if callback:
                await callback(final_chunk)

            yield final_chunk

    async def preload_model(self, model: str | None = None) -> bool:
        """Preload model for faster first inference"""
        model_to_load = model or self.model

        try:
            client = await self._get_client()

            # Send a simple generate request to load the model
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_to_load,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=30.0,
            )

            response.raise_for_status()
            log.info(f"Successfully preloaded model: {model_to_load}")
            return True

        except Exception as e:
            log.warning(f"Failed to preload model {model_to_load}: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about loaded models"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Failed to get model info: {e}")
            return {}

    def clear_cache(self):
        """Clear response cache"""
        self._response_cache.clear()
        log.info("Response cache cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_responses": len(self._response_cache),
            "max_cache_size": self._cache_max_size,
            "cache_hit_ratio": 0.0,  # Would need hit/miss tracking to calculate
        }
