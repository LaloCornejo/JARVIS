from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .vector import VectorMemory, get_vector_memory

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extract key facts from this conversation that should be remembered about the user.
Focus on:
- Personal preferences (likes, dislikes, habits)
- Important information (name, location, work, family)
- Goals, projects, or tasks they're working on
- Technical preferences (tools, languages, workflows)

Return a JSON array of objects with "fact" and "category" fields.
Categories: "personal", "preference", "project", "technical", "location", "work"

If no meaningful facts to extract, return an empty array: []

Conversation:
{conversation}

Return ONLY valid JSON, no other text."""


class FactExtractor:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:1.7b",
        vector_memory: VectorMemory | None = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.memory = vector_memory or get_vector_memory()
        self._client: httpx.AsyncClient | None = None
        self._running_tasks: set[asyncio.Task] = set()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        for task in self._running_tasks:
            task.cancel()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _call_llm(self, prompt: str) -> str:
        client = await self._get_client()
        response = await client.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _parse_facts(self, response: str) -> list[dict[str, str]]:
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            facts = json.loads(response)
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict) and "fact" in f]
        except json.JSONDecodeError:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse facts: {response[:200]}")
        return []

    async def extract_facts(
        self,
        conversation: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        if len(conversation) < 2:
            return []

        conv_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in conversation[-10:]
        )

        prompt = EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            response = await self._call_llm(prompt)
            facts = self._parse_facts(response)

            stored = []
            for fact_data in facts:
                fact_text = fact_data.get("fact", "")
                category = fact_data.get("category", "general")

                if len(fact_text) < 10:
                    continue

                existing = self.memory.search(fact_text, limit=1, category=category)
                if existing and existing[0]["score"] > 0.85:
                    logger.debug(f"Skipping duplicate fact: {fact_text[:50]}")
                    continue

                memory_id = self.memory.add(
                    text=fact_text,
                    category=category,
                    metadata={"source": "conversation_extraction"},
                )
                stored.append(
                    {
                        "id": memory_id,
                        "fact": fact_text,
                        "category": category,
                    }
                )
                logger.info(f"Stored fact [{category}]: {fact_text[:50]}")

            return stored

        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    def extract_facts_background(
        self,
        conversation: list[dict[str, str]],
    ) -> None:
        async def _run():
            try:
                await self.extract_facts(conversation)
            except Exception as e:
                logger.error(f"Background extraction failed: {e}")
            finally:
                self._running_tasks.discard(asyncio.current_task())

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_run())
            self._running_tasks.add(task)
        except RuntimeError:
            pass


_extractor: FactExtractor | None = None


def get_fact_extractor(
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3:1.7b",
) -> FactExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FactExtractor(ollama_url=ollama_url, model=model)
    return _extractor
