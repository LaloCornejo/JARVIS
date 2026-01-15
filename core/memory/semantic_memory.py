"""
Advanced memory system with semantic search and enhanced retrieval.

This module extends the existing vector memory with:
- Semantic search capabilities
- Long-term memory consolidation
- Contextual memory retrieval
- Memory importance scoring
- Memory clustering and organization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .vector import VectorMemory

log = logging.getLogger(__name__)


class SemanticMemory:
    """Enhanced memory system with semantic search and intelligent retrieval"""

    def __init__(self, vector_memory: VectorMemory):
        self.vector_memory = vector_memory
        self.model: SentenceTransformer = vector_memory._get_model()
        self._memory_importance_scores: Dict[str, float] = {}
        self._memory_clusters: Dict[str, List[str]] = {}
        self._consolidation_threshold = 50  # Consolidate after 50 memories

    async def store_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None, importance: float = 1.0
    ) -> str:
        """Store memory with semantic embedding and importance scoring"""
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Calculate embedding
        embedding = await self._calculate_embedding(content)

        # Store in vector memory
        await self.vector_memory.store_memory(
            content=content,
            embedding=embedding,
            metadata={
                **(metadata or {}),
                "memory_id": memory_id,
                "importance": importance,
                "created_at": datetime.now().isoformat(),
                "content_length": len(content),
            },
        )

        # Track importance
        self._memory_importance_scores[memory_id] = importance

        # Check if consolidation is needed
        total_memories = len(self._memory_importance_scores)
        if total_memories >= self._consolidation_threshold:
            asyncio.create_task(self._consolidate_memories())

        return memory_id

    async def semantic_search(
        self, query: str, limit: int = 10, threshold: float = 0.3, context_window: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with intelligent ranking and context"""
        # Calculate query embedding
        query_embedding = await self._calculate_embedding(query)

        # Search vector memory using text-based search
        # Note: For now, we'll use a simplified approach
        # In a full implementation, this would use the embedding directly
        results = self.vector_memory.search(query=query, limit=limit * 2)

        # Convert to expected format
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.5),
                    "metadata": result,
                }
            )

        results = formatted_results

        # Enhanced ranking with importance and recency
        ranked_results = await self._rank_results(query, results)

        # Add context around top results
        contextual_results = await self._add_context(ranked_results[:limit], context_window)

        return contextual_results

    async def retrieve_contextual_memories(
        self, current_topic: str, conversation_history: List[Dict[str, str]], max_memories: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to current conversation context"""
        # Analyze conversation context
        context_summary = await self._analyze_conversation_context(conversation_history)

        # Search for relevant memories
        relevant_memories = await self.semantic_search(
            query=context_summary, limit=max_memories * 2
        )

        # Filter and rank by conversation relevance
        conversation_relevant = await self._filter_conversation_relevant(
            relevant_memories, conversation_history
        )

        return conversation_relevant[:max_memories]

    async def get_memory_insights(self, topic: str) -> Dict[str, Any]:
        """Get insights about memories related to a topic"""
        memories = await self.semantic_search(topic, limit=50)

        if not memories:
            return {"insights": "No memories found for this topic"}

        # Analyze patterns
        insights = {
            "total_memories": len(memories),
            "average_importance": np.mean([m.get("importance", 1.0) for m in memories]),
            "time_distribution": self._analyze_time_distribution(memories),
            "content_patterns": await self._analyze_content_patterns(memories),
            "key_memories": memories[:5],  # Most relevant
            "insights": await self._generate_insights(memories, topic),
        }

        return insights

    async def _calculate_embedding(self, text: str) -> np.ndarray:
        """Calculate semantic embedding for text"""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, convert_to_numpy=True)
        )
        return embedding

    async def _rank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank search results using multiple criteria"""
        if not results:
            return results

        ranked = []
        for result in results:
            # Base similarity score from vector search
            base_score = result.get("score", 0.5)

            # Importance multiplier
            memory_id = result.get("metadata", {}).get("memory_id")
            importance = self._memory_importance_scores.get(memory_id, 1.0)

            # Recency boost (newer memories get slight preference)
            created_at = result.get("metadata", {}).get("created_at")
            recency_score = self._calculate_recency_score(created_at)

            # Content relevance (check for exact matches, key terms)
            content_relevance = self._calculate_content_relevance(query, result.get("content", ""))

            # Combined score
            final_score = (
                base_score * 0.4  # Vector similarity
                + importance * 0.3  # Importance
                + recency_score * 0.2  # Recency
                + content_relevance * 0.1  # Content matching
            )

            result_copy = result.copy()
            result_copy["enhanced_score"] = final_score
            ranked.append(result_copy)

        # Sort by enhanced score
        ranked.sort(key=lambda x: x["enhanced_score"], reverse=True)
        return ranked

    async def _add_context(
        self, results: List[Dict[str, Any]], context_window: int
    ) -> List[Dict[str, Any]]:
        """Add contextual memories around the main results"""
        if not results or context_window <= 0:
            return results

        enhanced_results = []

        for result in results:
            enhanced_result = result.copy()

            # Find memories created around the same time
            created_at = result.get("metadata", {}).get("created_at")
            if created_at:
                try:
                    base_time = datetime.fromisoformat(created_at)
                    time_window = timedelta(hours=1)  # 1 hour window

                    # Search for contextual memories
                    context_query = f"context around {base_time.strftime('%Y-%m-%d %H:%M')}"
                    context_memories = self.vector_memory.search(
                        query=context_query, limit=context_window
                    )

                    # Convert to expected format
                    formatted_context = []
                    for cm in context_memories:
                        formatted_context.append(
                            {
                                "content": cm.get("content", ""),
                                "score": cm.get("score", 0.5),
                                "metadata": cm,
                            }
                        )
                    context_memories = formatted_context

                    # Filter to time window and exclude the main result
                    main_id = result.get("metadata", {}).get("memory_id")
                    filtered_context = [
                        ctx
                        for ctx in context_memories
                        if ctx.get("metadata", {}).get("memory_id") != main_id
                        and self._is_within_time_window(
                            ctx.get("metadata", {}).get("created_at"), base_time, time_window
                        )
                    ]

                    enhanced_result["contextual_memories"] = filtered_context[:context_window]

                except Exception as e:
                    log.debug(f"Failed to add context: {e}")
                    enhanced_result["contextual_memories"] = []

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _consolidate_memories(self):
        """Consolidate old, low-importance memories to prevent database bloat"""
        try:
            log.info("Starting memory consolidation...")

            # Get all memories
            all_memories = await self.vector_memory.get_all_memories()

            # Identify memories to consolidate
            to_consolidate = []
            to_keep = []

            for memory in all_memories:
                importance = memory.get("metadata", {}).get("importance", 1.0)
                created_at = memory.get("metadata", {}).get("created_at")

                # Consolidate old, low-importance memories
                if importance < 0.5 and self._is_old_memory(created_at):
                    to_consolidate.append(memory)
                else:
                    to_keep.append(memory)

            if to_consolidate:
                # Create consolidated memory
                consolidated_content = await self._create_consolidated_memory(to_consolidate)

                if consolidated_content:
                    # Store consolidated memory
                    await self.store_memory(
                        content=f"Consolidated memories: {consolidated_content}",
                        metadata={"type": "consolidated", "original_count": len(to_consolidate)},
                        importance=0.3,
                    )

                    # Remove original memories
                    for memory in to_consolidate:
                        memory_id = memory.get("metadata", {}).get("memory_id")
                        if memory_id:
                            await self.vector_memory.delete_memory(memory_id)

                log.info(f"Consolidated {len(to_consolidate)} memories")

        except Exception as e:
            log.error(f"Memory consolidation failed: {e}")

    async def _create_consolidated_memory(self, memories: List[Dict[str, Any]]) -> str:
        """Create a consolidated summary of multiple memories"""
        if not memories:
            return ""

        # Extract key information
        contents = [m.get("content", "") for m in memories]
        combined_text = " ".join(contents)

        # Simple extractive summarization (can be enhanced with LLM)
        words = combined_text.split()
        if len(words) > 100:
            # Take first and last parts, plus some middle
            summary_words = (
                words[:30]
                + ["..."]
                + words[len(words) // 2 - 10 : len(words) // 2 + 10]
                + ["..."]
                + words[-30:]
            )
            return " ".join(summary_words)

        return combined_text[:500] + "..." if len(combined_text) > 500 else combined_text

    async def _analyze_conversation_context(
        self, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Analyze conversation context for memory retrieval"""
        if not conversation_history:
            return ""

        # Extract recent messages and combine
        recent_messages = conversation_history[-5:]  # Last 5 messages
        context_parts = []

        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content[:100]}")  # Truncate long messages

        return " ".join(context_parts)

    async def _filter_conversation_relevant(
        self, memories: List[Dict[str, Any]], conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Filter memories for conversation relevance"""
        if not memories or not conversation_history:
            return memories

        # Simple relevance scoring based on keyword overlap
        conversation_text = " ".join([msg.get("content", "") for msg in conversation_history])

        relevant_memories = []
        for memory in memories:
            memory_content = memory.get("content", "")
            relevance_score = self._calculate_text_overlap(conversation_text, memory_content)

            if relevance_score > 0.1:  # 10% overlap threshold
                memory_copy = memory.copy()
                memory_copy["conversation_relevance"] = relevance_score
                relevant_memories.append(memory_copy)

        # Sort by relevance
        relevant_memories.sort(key=lambda x: x.get("conversation_relevance", 0), reverse=True)
        return relevant_memories

    async def _analyze_content_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in memory content"""
        if not memories:
            return {}

        contents = [m.get("content", "") for m in memories]

        # Simple pattern analysis
        avg_length = np.mean([len(c) for c in contents])
        total_words = sum(len(c.split()) for c in contents)

        # Common themes (simplified - could use topic modeling)
        themes = []
        theme_keywords = {
            "work": ["work", "project", "task", "meeting", "email"],
            "personal": ["family", "friend", "home", "personal", "health"],
            "technical": ["code", "programming", "software", "computer", "system"],
            "learning": ["learn", "study", "course", "book", "research"],
        }

        for theme, keywords in theme_keywords.items():
            theme_count = sum(
                1 for content in contents if any(keyword in content.lower() for keyword in keywords)
            )
            if theme_count > len(contents) * 0.3:  # 30% threshold
                themes.append(theme)

        return {
            "average_length": avg_length,
            "total_words": total_words,
            "common_themes": themes,
            "memory_count": len(memories),
        }

    async def _generate_insights(self, memories: List[Dict[str, Any]], topic: str) -> str:
        """Generate human-readable insights about memories"""
        if not memories:
            return f"No memories found related to '{topic}'"

        patterns = await self._analyze_content_patterns(memories)

        insights = [
            f"Found {len(memories)} memories related to '{topic}'",
            f"Average memory length: {patterns['average_length']:.0f} characters",
            f"Total content: {patterns['total_words']} words",
        ]

        if patterns["common_themes"]:
            insights.append(f"Common themes: {', '.join(patterns['common_themes'])}")

        # Add temporal insights
        time_dist = self._analyze_time_distribution(memories)
        if time_dist.get("oldest") and time_dist.get("newest"):
            insights.append(f"Time span: {time_dist['oldest']} to {time_dist['newest']}")

        return ". ".join(insights)

    def _calculate_recency_score(self, created_at: Optional[str]) -> float:
        """Calculate recency score (0-1, higher is more recent)"""
        if not created_at:
            return 0.5

        try:
            creation_time = datetime.fromisoformat(created_at)
            now = datetime.now()
            days_old = (now - creation_time).days

            # Exponential decay: 1.0 for today, 0.5 for 30 days ago, etc.
            return max(0.1, np.exp(-days_old / 30))
        except Exception:
            return 0.5

    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """Calculate content relevance based on keyword overlap"""
        if not query or not content:
            return 0.0

        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        overlap = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return overlap / union if union > 0 else 0.0

    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _is_within_time_window(
        self, timestamp_str: Optional[str], base_time: datetime, window: timedelta
    ) -> bool:
        """Check if timestamp is within time window of base time"""
        if not timestamp_str:
            return False

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return abs(timestamp - base_time) <= window
        except Exception:
            return False

    def _is_old_memory(self, created_at: Optional[str], days_threshold: int = 90) -> bool:
        """Check if memory is old enough for consolidation"""
        if not created_at:
            return False

        try:
            creation_time = datetime.fromisoformat(created_at)
            days_old = (datetime.now() - creation_time).days
            return days_old > days_threshold
        except Exception:
            return False

    def _analyze_time_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal distribution of memories"""
        if not memories:
            return {}

        timestamps = []
        for memory in memories:
            created_at = memory.get("metadata", {}).get("created_at")
            if created_at:
                try:
                    timestamps.append(datetime.fromisoformat(created_at))
                except Exception:
                    continue

        if not timestamps:
            return {}

        timestamps.sort()
        return {
            "oldest": timestamps[0].strftime("%Y-%m-%d") if timestamps else None,
            "newest": timestamps[-1].strftime("%Y-%m-%d") if timestamps else None,
            "span_days": (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0,
            "count": len(timestamps),
        }


class EnhancedMemorySystem:
    """Complete enhanced memory system combining vector and semantic capabilities"""

    def __init__(self, db_path: str = "data/enhanced_memory"):
        self.vector_memory = VectorMemory(db_path=db_path)
        self.semantic_memory = SemanticMemory(self.vector_memory)
        self.conversation_memory = None  # Will be set externally

    async def initialize(self):
        """Initialize the enhanced memory system"""
        # Vector memory is initialized lazily
        log.info("Enhanced memory system initialized")

    async def store_conversation_memory(
        self, conversation_id: int, content: str, importance: float = 1.0
    ) -> str:
        """Store conversation-related memory with semantic embedding"""
        metadata = {
            "type": "conversation",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
        }

        return await self.semantic_memory.store_memory(content, metadata, importance)

    async def retrieve_relevant_memories(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve relevant memories using semantic search"""
        # Semantic search
        semantic_results = await self.semantic_memory.semantic_search(query, limit=limit)

        # Add conversation context if available
        contextual_results = []
        if conversation_context:
            contextual_results = await self.semantic_memory.retrieve_contextual_memories(
                query, conversation_context, max_memories=limit
            )

        return {
            "semantic_results": semantic_results,
            "contextual_results": contextual_results,
            "query": query,
            "total_results": len(semantic_results) + len(contextual_results),
        }

    async def get_memory_insights(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive insights about memories"""
        return await self.semantic_memory.get_memory_insights(topic)

    async def consolidate_old_memories(self):
        """Manually trigger memory consolidation"""
        await self.semantic_memory._consolidate_memories()


# Global enhanced memory system instance
enhanced_memory = EnhancedMemorySystem()


async def get_enhanced_memory() -> EnhancedMemorySystem:
    """Get the global enhanced memory system instance"""
    await enhanced_memory.initialize()
    return enhanced_memory


__all__ = [
    "SemanticMemory",
    "EnhancedMemorySystem",
    "enhanced_memory",
    "get_enhanced_memory",
]
