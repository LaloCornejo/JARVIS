"""
Advanced buffering and caching system for streaming optimization.

This module provides intelligent buffering strategies, predictive caching,
and adaptive chunking for optimal streaming performance.
"""

import asyncio
import hashlib
import logging
import re
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Dict, List, Optional


log = logging.getLogger(__name__)


class StreamingBufferManager:
    """Advanced buffering for streaming optimization"""

    def __init__(self):
        self.content_buffer = deque(maxlen=1000)
        self.tool_result_cache = {}
        self.response_predictor = ResponsePredictor()
        self.adaptive_chunker = AdaptiveChunker()
        self.buffer_stats = {
            "chunks_processed": 0,
            "average_chunk_size": 0,
            "content_types": defaultdict(int),
        }

    async def optimize_streaming_chunks(
        self, content: str, context: dict = None
    ) -> AsyncIterator[str]:
        """Intelligent chunking based on content type and context"""
        if not content:
            return

        # Analyze content type for optimal chunking
        content_type = self._analyze_content_type(content)
        self.buffer_stats["content_types"][content_type] += 1

        # Get adaptive chunker for this content type
        chunker = self.adaptive_chunker.get_chunker(content_type)

        # Apply context-aware chunking
        async for chunk in chunker.chunk(content, context):
            self.buffer_stats["chunks_processed"] += 1
            chunk_size = len(chunk)
            self.buffer_stats["average_chunk_size"] = (
                (
                    self.buffer_stats["average_chunk_size"]
                    * (self.buffer_stats["chunks_processed"] - 1)
                )
                + chunk_size
            ) / self.buffer_stats["chunks_processed"]

            yield chunk

    def _analyze_content_type(self, content: str) -> str:
        """Analyze content for optimal chunking strategy"""
        if not content:
            return "empty"

        content_lower = content.lower()

        # Technical content (code, data, technical writing)
        if any(
            keyword in content_lower
            for keyword in [
                "function",
                "class",
                "import",
                "def ",
                "return",
                "if ",
                "for ",
                "api",
                "json",
                "http",
                "database",
                "algorithm",
            ]
        ):
            return "technical"

        # Conversational content
        sentence_count = len(re.findall(r"[.!?]+", content))
        word_count = len(content.split())

        if sentence_count > word_count * 0.1:  # High sentence density
            return "conversational"

        # Structured content (lists, tables, formatted text)
        if any(char in content for char in ["•", "-", "1.", "2.", "|", "\t"]):
            return "structured"

        # Code or data
        if any(char in content for char in ["{", "}", "[", "]", "(", ")", "=", ":"]):
            return "code"

        # Default to general content
        return "general"

    def get_buffer_stats(self) -> dict:
        """Get buffer performance statistics"""
        return dict(self.buffer_stats)


class AdaptiveChunker:
    """Adaptive chunking based on content type and context"""

    def __init__(self):
        self.chunkers = {
            "conversational": ConversationalChunker(),
            "technical": TechnicalChunker(),
            "structured": StructuredChunker(),
            "code": CodeChunker(),
            "general": GeneralChunker(),
            "empty": EmptyChunker(),
        }

    def get_chunker(self, content_type: str):
        """Get appropriate chunker for content type"""
        return self.chunkers.get(content_type, self.chunkers["general"])


class BaseChunker:
    """Base class for content-aware chunking"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk content based on specific strategy"""
        raise NotImplementedError


class ConversationalChunker(BaseChunker):
    """Chunking optimized for conversational content"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk at sentence boundaries with smart sizing"""
        sentences = re.split(r"([.!?]+)", content)

        buffer = ""
        for i in range(0, len(sentences) - 1, 2):  # Handle sentence + punctuation
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            buffer += sentence

            # Yield at optimal points
            if len(buffer) >= 100 or sentence.endswith((".", "!", "?")):
                yield buffer.strip()
                buffer = ""

        if buffer:
            yield buffer.strip()


class TechnicalChunker(BaseChunker):
    """Chunking optimized for technical content"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk at logical boundaries in technical content"""
        # Split on code blocks, paragraphs, and sentences
        parts = re.split(r"(\n\s*\n|```[\s\S]*?```|def |class )", content)

        buffer = ""
        for part in parts:
            buffer += part

            # Yield at logical boundaries
            if len(buffer) >= 200 or part.endswith(("\n\n", "```", "def ", "class ")):
                yield buffer.strip()
                buffer = ""

        if buffer:
            yield buffer.strip()


class StructuredChunker(BaseChunker):
    """Chunking optimized for structured content (lists, tables)"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk at structural boundaries"""
        # Split on list items, table rows, headers
        parts = re.split(r"(\n\s*[-•*]\s*|\n\s*\d+\.\s*|\n\s*#+\s*|^\s*#+\s*)", content)

        buffer = ""
        for part in parts:
            buffer += part

            # Yield at structural boundaries
            if len(buffer) >= 150 or re.search(r"\n\s*[-•*\d]+\.\s*|\n\s*#+\s*", part):
                yield buffer.strip()
                buffer = ""

        if buffer:
            yield buffer.strip()


class CodeChunker(BaseChunker):
    """Chunking optimized for code content"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk at function/class boundaries"""
        # Split on function definitions, class definitions, imports
        parts = re.split(r"(\n\s*(?:def |class |import |from )\s*|\n\s*#.*\n)", content)

        buffer = ""
        for part in parts:
            buffer += part

            # Yield at code boundaries
            if len(buffer) >= 300 or re.search(r"\n\s*(?:def |class |import |from )\s*", part):
                yield buffer.strip()
                buffer = ""

        if buffer:
            yield buffer.strip()


class GeneralChunker(BaseChunker):
    """General-purpose chunking strategy"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Chunk with adaptive sizing"""
        words = content.split()
        buffer = ""

        for word in words:
            buffer += word + " "

            # Adaptive chunk size based on context
            max_chunk_size = 120 if context and context.get("is_streaming") else 200

            if len(buffer) >= max_chunk_size:
                yield buffer.strip()
                buffer = ""

        if buffer:
            yield buffer.strip()


class EmptyChunker(BaseChunker):
    """Chunker for empty content"""

    async def chunk(self, content: str, context: dict = None) -> AsyncIterator[str]:
        """Handle empty content"""
        return
        yield  # Make this an async generator


class ResponsePredictor:
    """Predictive caching for response optimization"""

    def __init__(self):
        self.pattern_cache = {}
        self.frequency_cache = defaultdict(int)
        self.prediction_threshold = 0.3

    def predict_response_pattern(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Predict likely response pattern for caching"""
        if not messages:
            return None

        # Extract key patterns from recent messages
        recent_messages = messages[-3:]  # Last 3 messages
        pattern_key = self._extract_pattern_key(recent_messages)

        # Check if we have cached predictions
        if pattern_key in self.pattern_cache:
            prediction = self.pattern_cache[pattern_key]
            if prediction["confidence"] > self.prediction_threshold:
                return prediction["response_pattern"]

        return None

    def _extract_pattern_key(self, messages: List[Dict[str, Any]]) -> str:
        """Extract pattern key from messages"""
        patterns = []
        for msg in messages:
            content = msg.get("content", "").lower()
            # Extract key question words and topics
            if any(word in content for word in ["what", "how", "why", "when", "where"]):
                patterns.append("question")
            elif any(word in content for word in ["help", "assist", "can you"]):
                patterns.append("request")
            elif any(word in content for word in ["yes", "no", "okay", "sure"]):
                patterns.append("confirmation")

        return "_".join(patterns) if patterns else "general"

    def cache_response_pattern(self, messages: List[Dict[str, Any]], response: str):
        """Cache successful response pattern"""
        pattern_key = self._extract_pattern_key(messages)

        # Update frequency
        self.frequency_cache[pattern_key] += 1

        # Cache pattern with confidence
        confidence = min(
            1.0, self.frequency_cache[pattern_key] / 10.0
        )  # Max confidence after 10 uses

        self.pattern_cache[pattern_key] = {
            "response_pattern": self._extract_response_pattern(response),
            "confidence": confidence,
            "last_updated": asyncio.get_event_loop().time(),
        }

    def _extract_response_pattern(self, response: str) -> str:
        """Extract pattern from response for future predictions"""
        # Simple pattern extraction - could be enhanced with ML
        response_lower = response.lower()

        if "sure" in response_lower or "happy to help" in response_lower:
            return "helpful_response"
        elif "according to" in response_lower or "based on" in response_lower:
            return "informational_response"
        elif any(word in response_lower for word in ["yes", "correct", "right"]):
            return "confirmatory_response"
        else:
            return "general_response"


class IntelligentCache:
    """Intelligent caching with predictive prefetching"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.frequencies = defaultdict(int)
        self.max_size = max_size
        self.prefetch_queue = asyncio.Queue()

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking"""
        if key in self.cache:
            self.access_times[key] = asyncio.get_event_loop().time()
            self.frequencies[key] += 1
            return self.cache[key]
        return None

    async def put(self, key: str, value: Any, priority: int = 1):
        """Put item in cache with priority"""
        if len(self.cache) >= self.max_size:
            await self._evict_low_priority()

        self.cache[key] = value
        self.access_times[key] = asyncio.get_event_loop().time()
        self.frequencies[key] = priority

    async def _evict_low_priority(self):
        """Evict low-priority items using intelligent algorithm"""
        if not self.cache:
            return

        # Calculate priority scores
        current_time = asyncio.get_event_loop().time()
        scores = {}

        for key in self.cache:
            # Score based on frequency, recency, and size
            time_since_access = current_time - self.access_times[key]
            recency_score = 1.0 / (1.0 + time_since_access / 3600)  # Decay over hours
            frequency_score = min(1.0, self.frequencies[key] / 10.0)  # Cap at 10 accesses

            scores[key] = recency_score * frequency_score

        # Remove lowest scoring items
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        items_to_remove = max(1, len(self.cache) - self.max_size + 10)  # Remove 10% or 1 item

        for key in sorted_keys[:items_to_remove]:
            del self.cache[key]
            del self.access_times[key]
            del self.frequencies[key]

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        if not self.cache:
            return {"hit_rate": 0.0, "size": 0, "max_size": self.max_size}

        total_accesses = sum(self.frequencies.values())
        avg_frequency = total_accesses / len(self.cache) if self.cache else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
            "avg_frequency": avg_frequency,
            "total_accesses": total_accesses,
        }


class StreamingOptimizer:
    """Main streaming optimization coordinator"""

    def __init__(self):
        self.buffer_manager = StreamingBufferManager()
        self.cache = IntelligentCache()
        self.response_predictor = ResponsePredictor()

        # Performance tracking
        self.performance_stats = {
            "total_chunks": 0,
            "average_latency": 0.0,
            "cache_hit_rate": 0.0,
            "prediction_accuracy": 0.0,
        }

    async def optimize_response_stream(
        self, response_generator: AsyncIterator[str], context: dict = None
    ) -> AsyncIterator[str]:
        """Optimize response streaming with intelligent chunking"""
        async for chunk in response_generator:
            # Apply intelligent chunking
            async for optimized_chunk in self.buffer_manager.optimize_streaming_chunks(
                chunk, context
            ):
                self.performance_stats["total_chunks"] += 1
                yield optimized_chunk

    def update_performance_stats(
        self, latency: float, cache_hit: bool = False, prediction_correct: bool = False
    ):
        """Update performance tracking statistics"""
        # Update average latency
        total_chunks = self.performance_stats["total_chunks"]
        current_avg = self.performance_stats["average_latency"]
        self.performance_stats["average_latency"] = (
            current_avg * (total_chunks - 1) + latency
        ) / total_chunks

        # Update cache hit rate (simplified tracking)
        if cache_hit:
            current_rate = self.performance_stats["cache_hit_rate"]
            self.performance_stats["cache_hit_rate"] = (
                current_rate * 0.9
            ) + 0.1  # Exponential moving average

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics"""
        buffer_stats = self.buffer_manager.get_buffer_stats()
        cache_stats = self.cache.get_cache_stats()

        return {
            **self.performance_stats,
            "buffer_stats": buffer_stats,
            "cache_stats": cache_stats,
            "optimization_efficiency": min(1.0, self.performance_stats["cache_hit_rate"] * 1.5),
        }


# Global streaming optimizer instance
streaming_optimizer = StreamingOptimizer()
