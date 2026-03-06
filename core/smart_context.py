"""
Smart Context Manager for JARVIS.
Provides intelligent context selection for LLM conversations.
"""

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from core.memory.vector import get_vector_memory

log = logging.getLogger(__name__)


class TopicContinuity(Enum):
    CONTINUATION = auto()
    NEW_TOPIC = auto()
    UNCLEAR = auto()


@dataclass
class MessageContext:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    topic_score: float = 0.0


@dataclass
class ContextDecision:
    continuity: TopicContinuity
    messages_to_include: list[dict]
    semantic_memories: list[dict]
    reasoning: str
    should_include_recent: bool
    recent_message_count: int


class SmartContextManager:
    TOPIC_SHIFTERS = {
        "new",
        "forget",
        "ignore",
        "reset",
        "clear",
        "actually",
        "by the way",
        "switching",
        "different",
        "hello",
        "hey",
        "hi",
        "good morning",
        "good evening",
        "what time",
        "what's the time",
        "search for",
        "look up",
    }

    TOPIC_KEEPERS = {
        "and",
        "also",
        "but",
        "however",
        "plus",
        "more",
        "what about",
        "how about",
        "and then",
        "after that",
        "what happened",
        "why",
        "how did",
        "did you",
        "it",
        "that",
        "them",
        "this",
        "those",
        "these",
    }

    def __init__(
        self,
        max_recent_messages: int = 6,
        max_context_messages: int = 10,
        semantic_limit: int = 3,
        similarity_threshold: float = 0.3,
        topic_shift_timeout: int = 300,
    ):
        self.max_recent_messages = max_recent_messages
        self.max_context_messages = max_context_messages
        self.semantic_limit = semantic_limit
        self.similarity_threshold = similarity_threshold
        self.topic_shift_timeout = topic_shift_timeout

        self._recent_messages: deque[MessageContext] = deque(maxlen=max_recent_messages * 2)
        self._last_message_time: float = 0
        self._current_session_id: str | None = None
        self._session_message_count: int = 0
        self._current_topics: set[str] = set()
        self._last_user_message: str = ""
        self._vector_memory = None
        self._vector_memory_lock = asyncio.Lock()
        self._enable_semantic_search = True

    async def _get_vector_memory(self):
        if self._vector_memory is None:
            async with self._vector_memory_lock:
                if self._vector_memory is None:
                    self._vector_memory = get_vector_memory()
        return self._vector_memory

    async def get_context(
        self,
        current_message: str,
        role: str = "user",
    ) -> ContextDecision:
        current_time = time.time()

        self._recent_messages.append(
            MessageContext(
                role=role,
                content=current_message,
                timestamp=current_time,
            )
        )

        time_since_last = current_time - self._last_message_time
        is_new_session = (
            time_since_last > self.topic_shift_timeout or self._session_message_count == 0
        )

        continuity = self._analyze_continuity(current_message, is_new_session)
        should_include_recent = continuity in (
            TopicContinuity.CONTINUATION,
            TopicContinuity.UNCLEAR,
        )
        recent_messages = self._get_recent_messages(include=should_include_recent)

        semantic_memories = []
        if self._enable_semantic_search and role == "user":
            try:
                semantic_memories = await self._get_semantic_memories(current_message)
            except Exception as e:
                log.debug(f"Semantic search failed: {e}")

        self._last_message_time = current_time
        self._session_message_count += 1
        self._extract_topics(current_message)
        self._last_user_message = current_message

        reasoning = self._build_reasoning(
            continuity=continuity,
            time_since_last=time_since_last,
            is_new_session=is_new_session,
            semantic_count=len(semantic_memories),
            recent_count=len(recent_messages),
        )

        return ContextDecision(
            continuity=continuity,
            messages_to_include=recent_messages,
            semantic_memories=semantic_memories,
            reasoning=reasoning,
            should_include_recent=should_include_recent,
            recent_message_count=len(recent_messages),
        )

    def _analyze_continuity(self, message: str, is_new_session: bool) -> TopicContinuity:
        msg_lower = message.lower()

        for shifter in self.TOPIC_SHIFTERS:
            if msg_lower.startswith(shifter) or f" {shifter} " in msg_lower:
                log.debug(f"Topic shift detected: '{shifter}' found in message")
                return TopicContinuity.NEW_TOPIC

        keeper_count = sum(1 for keeper in self.TOPIC_KEEPERS if f" {keeper} " in msg_lower)

        if keeper_count >= 2 and len(self._recent_messages) > 1:
            return TopicContinuity.CONTINUATION

        if "?" in message and len(self._recent_messages) > 2:
            if any(word in msg_lower for word in ["this", "that", "it", "these", "those"]):
                return TopicContinuity.CONTINUATION

        if self._last_user_message:
            similarity = self._calculate_text_similarity(message, self._last_user_message)
            if similarity > 0.5:
                return TopicContinuity.CONTINUATION
            elif similarity > 0.2:
                return TopicContinuity.UNCLEAR

        return TopicContinuity.NEW_TOPIC

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

        if not words1 or not words2:
            return 0.0

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "your",
        }
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_topics(self, message: str) -> None:
        words = re.findall(r"\w+", message.lower())

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "your",
            "my",
            "me",
            "and",
            "or",
            "not",
            "no",
            "but",
            "if",
            "so",
            "just",
            "get",
            "got",
            "need",
            "needs",
            "want",
            "wants",
        }

        meaningful_words = {w for w in words if w not in stop_words and len(w) > 2}

        if meaningful_words:
            self._current_topics = meaningful_words

    def _get_recent_messages(self, include: bool = True) -> list[dict]:
        if not include:
            return []

        recent = []
        for msg_ctx in list(self._recent_messages)[-self.max_context_messages :]:
            recent.append(
                {
                    "role": msg_ctx.role,
                    "content": msg_ctx.content,
                }
            )

        return recent

    async def _get_semantic_memories(self, query: str) -> list[dict]:
        try:
            vector_mem = await self._get_vector_memory()

            results = await vector_mem.search_async(
                query=query,
                limit=self.semantic_limit,
                category="conversation",
            )

            relevant = [r for r in results if r.get("score", 0) >= self.similarity_threshold]

            log.debug(f"Found {len(relevant)} relevant semantic memories")
            return relevant

        except Exception as e:
            log.debug(f"Error getting semantic memories: {e}")
            return []

    def _build_reasoning(
        self,
        continuity: TopicContinuity,
        time_since_last: float,
        is_new_session: bool,
        semantic_count: int,
        recent_count: int,
    ) -> str:
        parts = []

        if is_new_session:
            parts.append("new_session")
        elif continuity == TopicContinuity.NEW_TOPIC:
            parts.append("topic_shift")
        elif continuity == TopicContinuity.CONTINUATION:
            parts.append("continuation")

        parts.append(f"recent_msgs={recent_count}")
        parts.append(f"semantic={semantic_count}")

        return "; ".join(parts)

    async def add_to_memory(
        self,
        role: str,
        content: str,
        category: str = "conversation",
    ) -> None:
        if role == "system":
            return

        try:
            vector_mem = await self._get_vector_memory()
            memory_text = f"[{role.upper()}] {content}"

            await vector_mem.add_async(
                text=memory_text,
                category=category,
                metadata={"role": role},
            )
            log.debug(f"Added message to vector memory: {role}")

        except Exception as e:
            log.debug(f"Failed to add to vector memory: {e}")

    def reset_session(self) -> None:
        self._current_session_id = None
        self._session_message_count = 0
        self._current_topics.clear()
        log.info("Context manager session reset")

    def get_stats(self) -> dict:
        return {
            "recent_message_count": len(self._recent_messages),
            "session_message_count": self._session_message_count,
            "current_topics": list(self._current_topics),
            "last_message_time": self._last_message_time,
        }


_context_manager: SmartContextManager | None = None


def get_context_manager() -> SmartContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = SmartContextManager()
    return _context_manager


async def get_smart_context(current_message: str, role: str = "user") -> ContextDecision:
    manager = get_context_manager()
    return await manager.get_context(current_message, role)


__all__ = [
    "SmartContextManager",
    "ContextDecision",
    "TopicContinuity",
    "MessageContext",
    "get_context_manager",
    "get_smart_context",
]
