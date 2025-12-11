from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("jarvis.ollama")

MODEL_CONTEXT_WINDOWS = {
    "qwen3:1.7b": 32768,
    "qwen3-vl": 32768,
    "gpt-oss": 32768,
}
DEFAULT_NUM_CTX = 16384


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl",
        timeout: float = 300.0,
        num_ctx: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx or self._get_context_window(model)
        self._client: httpx.AsyncClient | None = None

    def _get_context_window(self, model: str) -> int:
        for name, ctx in MODEL_CONTEXT_WINDOWS.items():
            if name in model.lower():
                return ctx
        return DEFAULT_NUM_CTX

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def preload_model(self, model: str | None = None) -> bool:
        target_model = model or self.model
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": target_model, "prompt": "", "keep_alive": "10m"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def preload_models(self, models: list[str]) -> dict[str, bool]:
        results = await asyncio.gather(
            *[self.preload_model(m) for m in models],
            return_exceptions=True,
        )
        return {m: r is True for m, r in zip(models, results)}

    def _should_disable_thinking(self) -> bool:
        return "qwen3" in self.model.lower()

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        actual_prompt = prompt
        if self._should_disable_thinking():
            actual_prompt = f"/no_think\n{prompt}"
        payload: dict = {
            "model": self.model,
            "prompt": actual_prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        async with client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if text := data.get("response"):
                        yield text
                    if data.get("done"):
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
        client = await self._get_client()
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if self._should_disable_thinking() and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content", "")
                if isinstance(content, str) and not content.startswith("/no_think"):
                    last_msg["content"] = f"/no_think\n{content}"
        if system:
            if messages and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system})
        if images and messages:
            messages[-1]["images"] = images
        if tools:
            payload["tools"] = tools

        log.warning(
            f"[OLLAMA CHAT] model={self.model}, messages={len(messages)}, tools={bool(tools)}"
        )

        async with client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            chunk_count = 0
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    chunk_count += 1
                    if msg := data.get("message", {}):
                        content = msg.get("content", "")
                        thinking = msg.get("thinking", "")
                        if content or thinking:
                            log.warning(
                                f"[OLLAMA CHAT] chunk {chunk_count}: "
                                f"content={len(content)}, thinking={len(thinking)}"
                            )
                    yield data
                    if data.get("done"):
                        log.warning(f"[OLLAMA CHAT] done after {chunk_count} chunks")
                        break

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
