from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("jarvis.compactor")


@dataclass
class CompactedContext:
    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    tool_schemas: list[dict] = field(default_factory=list)
    estimated_tokens: int = 0
    compression_ratio: float = 1.0


class TokenCompactor:
    """Compress conversation + tool context within token budget."""

    def __init__(self, max_context_tokens: int = 16_384) -> None:
        self.max_context_tokens = max_context_tokens

    def prepare_context(
        self,
        query: str,
        conversation_messages: list[dict],
        all_tool_schemas: list[dict],
        system_prompt: str,
        extra_context: str | None = None,
    ) -> CompactedContext:
        raw_tokens = self._estimate_tokens(conversation_messages)
        system_tokens = self._estimate_tokens(system_prompt)
        tool_tokens = self._estimate_tokens(all_tool_schemas)

        budget = self.max_context_tokens - system_tokens - tool_tokens - 1000

        filtered_messages = list(conversation_messages)
        log.warning(
            f"[COMPACTOR] Budget: {budget} tokens "
            f"(max={self.max_context_tokens}, system={system_tokens}, tools={tool_tokens}, reserve=1000)"
        )
        if budget <= 0:
            log.warning("Token budget exhausted before any messages — trimming tool schemas.")
            return CompactedContext(
                messages=[],
                system_prompt=system_prompt,
                tool_schemas=self._trim_tool_schemas(all_tool_schemas, self.max_context_tokens // 2),
                estimated_tokens=0,
                compression_ratio=0.0,
            )

        truncated_messages: list[dict] = []
        token_count = 0
        for msg in reversed(filtered_messages):
            msg_tokens = self._estimate_tokens(msg)
            if token_count + msg_tokens > budget:
                break
            truncated_messages.insert(0, msg)
            token_count += msg_tokens

        elapsed = raw_tokens - token_count
        ratio = (token_count / raw_tokens) if raw_tokens > 0 else 1.0

        log.warning(
            f"[COMPACTOR] Compacted: {len(filtered_messages)} → {len(truncated_messages)} msgs, "
            f"{token_count}/{raw_tokens} tokens ({ratio*100:.0f}%)"
        )

        return CompactedContext(
            messages=truncated_messages,
            system_prompt=system_prompt,
            tool_schemas=list(all_tool_schemas),
            estimated_tokens=token_count + system_tokens + tool_tokens,
            compression_ratio=ratio,
        )

    def _estimate_tokens(self, obj: object) -> int:
        text = str(obj)
        return (len(text) + 3) // 4

    def _trim_tool_schemas(self, schemas: list[dict], budget: int) -> list[dict]:
        trimmed: list[dict] = []
        count = 0
        for s in schemas:
            t = self._estimate_tokens(s)
            if count + t > budget:
                break
            trimmed.append(s)
            count += t
        return trimmed
