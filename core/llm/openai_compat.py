from __future__ import annotations

import json
import logging
from typing import Any, Dict, AsyncIterator

import httpx

log = logging.getLogger("jarvis.vllm")


class OpenAICompatClient:
    """OpenAI-compatible client for vLLM, LM Studio, etc."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        api_key: str = "none",
        timeout: float = 300.0,
    ):
        if model is None:
            raise ValueError("model is required")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            log.warning(f"Health check failed: {e}")
            return False

    async def preload_model(self, model: str | None = None) -> bool:
        """vLLM loads models on startup, no preloading needed."""
        return await self.health_check()

    def _should_disable_thinking(self) -> bool:
        return "qwen3" in self.model.lower()

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        """Generate text (non-chat interface)."""
        client = await self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }

        if num_predict:
            payload["max_tokens"] = num_predict

        # Handle vision if images provided
        if images:
            # Convert base64 to data URLs for OpenAI format
            content: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
            messages[-1]["content"] = content  # type: ignore[assignment]

        endpoint = (
            f"{self.base_url}/completions" if not images else f"{self.base_url}/chat/completions"
        )

        if images:
            endpoint = f"{self.base_url}/chat/completions"

        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    # Handle SSE format
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("choices"):
                            delta = data["choices"][0].get("delta", {})
                            if text := delta.get("content", ""):
                                yield text
                            if thinking := delta.get("thinking", ""):
                                yield thinking
                        if data.get("done") or data.get("choices", [{}])[0].get("finish_reason"):
                            break

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        images: list[str] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        """Chat completion interface."""
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }

        # Handle system message
        if system:
            if messages and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system})

        # Handle images in last message
        if images and messages:
            # Convert base64 images to OpenAI vision format
            content = [{"type": "text", "text": messages[-1].get("content", "")}]
            for img in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
            messages[-1]["content"] = content

        # Handle tools
        if tools:
            payload["tools"] = tools

        log.debug(f"[VLLM CHAT] model={self.model}, messages={len(messages)}, tools={bool(tools)}")

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            chunk_count = 0
            sse_meta: dict[str, str] = {}
            body_lines: list[str] = []
            async for line in response.aiter_lines():
                if line:
                    # SSE comment lines (": ...") — extract metadata, skip body
                    if line.startswith(": "):
                        if line.startswith(": x-omniroute-"):
                            try:
                                k, v = line[2:].split("=", 1)
                                sse_meta[k] = v
                            except ValueError:
                                pass
                        continue
                    body_lines.append(line)

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        chunk_count += 1

                        if msg := data.get("choices", [{}])[0].get("delta", {}):
                            content = msg.get("content", "")
                            thinking = msg.get("thinking", "")
                            if content or thinking:
                                log.debug(
                                    f"[VLLM CHAT] chunk {chunk_count}: "
                                    f"content={len(content)}, thinking={len(thinking)}"
                                )

                        yield data

                        if data.get("choices", [{}])[0].get("finish_reason"):
                            log.debug(f"[VLLM CHAT] done after {chunk_count} chunks")
                            break

            # Non-streaming fallback: server ignored stream=True, or returned only SSE comments
            if chunk_count == 0 and sse_meta:
                model = sse_meta.get("x-omniroute-model", "?")
                tokens_in = sse_meta.get("x-omniroute-tokens-in", "?")
                tokens_out = sse_meta.get("x-omniroute-tokens-out", "?")
                log.warning(
                    f"[VLLM CHAT] Omniroute upstream ({model}) returned 0 data events "
                    f"({tokens_in} in / {tokens_out} out) — upstream request rejected"
                )
            elif chunk_count == 0 and body_lines:
                body = "".join(body_lines)
                log.warning(f"[VLLM CHAT] No SSE chunks — treating as non-streaming JSON ({len(body)} bytes)")
                try:
                    data = json.loads(body)
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message") or choices[0].get("delta", {})
                        if msg.get("content") or msg.get("tool_calls"):
                            yield data
                            return
                    log.warning(f"[VLLM CHAT] Non-streaming parse failed: {body[:500]}")
                except json.JSONDecodeError as e:
                    log.warning(f"[VLLM CHAT] Non-streaming body is not JSON ({e}): {body[:200]}")
            elif chunk_count == 0:
                log.warning("[VLLM CHAT] Empty response body from server")

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json().get("data", [])
