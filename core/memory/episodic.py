"""
Episodic Memory System for JARVIS.

Stores temporal experiences, events, and autobiographical memories.
Enables the assistant to recall specific moments and experiences.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class EpisodeType(Enum):
    """Types of episodic memories"""

    CONVERSATION = "conversation"
    ACTION = "action"
    OBSERVATION = "observation"
    DECISION = "decision"
    ACHIEVEMENT = "achievement"
    ERROR = "error"
    LEARNING = "learning"


@dataclass
class Episode:
    """A single episodic memory"""

    id: str
    timestamp: datetime
    episode_type: EpisodeType
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    emotions: List[str] = field(default_factory=list)
    importance: float = 1.0
    location: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    duration: Optional[timedelta] = None
    tags: List[str] = field(default_factory=list)
    related_episodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "episode_type": self.episode_type.value,
            "content": self.content,
            "context": self.context,
            "emotions": self.emotions,
            "importance": self.importance,
            "location": self.location,
            "participants": self.participants,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "tags": self.tags,
            "related_episodes": self.related_episodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            episode_type=EpisodeType(data["episode_type"]),
            content=data["content"],
            context=data.get("context", {}),
            emotions=data.get("emotions", []),
            importance=data.get("importance", 1.0),
            location=data.get("location"),
            participants=data.get("participants", []),
            duration=timedelta(seconds=data["duration_seconds"])
            if data.get("duration_seconds")
            else None,
            tags=data.get("tags", []),
            related_episodes=data.get("related_episodes", []),
        )


@dataclass
class TemporalSequence:
    """A sequence of related episodes"""

    id: str
    name: str
    description: str
    episode_ids: List[str]
    start_time: datetime
    end_time: datetime
    tags: List[str] = field(default_factory=list)

    def duration(self) -> timedelta:
        return self.end_time - self.start_time


@dataclass
class EpisodeQuery:
    """Query parameters for episode retrieval"""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    episode_types: Optional[List[EpisodeType]] = None
    participants: Optional[List[str]] = None
    location: Optional[str] = None
    tags: Optional[List[str]] = None
    emotions: Optional[List[str]] = None
    min_importance: float = 0.0
    keywords: Optional[List[str]] = None
    limit: int = 10


class EpisodicMemory:
    """
    Episodic memory system for storing and retrieving temporal experiences.

    Features:
    - Store conversations, actions, observations as episodes
    - Temporal querying and retrieval
    - Importance-based retention
    - Sequence detection and reconstruction
    - Emotional context tracking
    """

    def __init__(self, storage_path: str = "data/episodic_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.episodes: Dict[str, Episode] = {}
        self.sequences: Dict[str, TemporalSequence] = {}
        self._index_by_time: List[Tuple[datetime, str]] = []
        self._index_by_type: Dict[EpisodeType, Set[str]] = {t: set() for t in EpisodeType}
        self._index_by_tag: Dict[str, Set[str]] = {}
        self._index_by_participant: Dict[str, Set[str]] = {}

        self._lock = asyncio.Lock()
        self._loaded = False

    async def initialize(self):
        """Initialize the episodic memory system"""
        if self._loaded:
            return

        await self._load_episodes()
        self._loaded = True
        log.info(f"Episodic memory initialized with {len(self.episodes)} episodes")

    async def _load_episodes(self):
        """Load episodes from storage"""
        episodes_file = self.storage_path / "episodes.json"

        if episodes_file.exists():
            try:
                with open(episodes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for episode_data in data.get("episodes", []):
                    episode = Episode.from_dict(episode_data)
                    self.episodes[episode.id] = episode
                    self._update_indexes(episode)

                for seq_data in data.get("sequences", []):
                    sequence = TemporalSequence(
                        id=seq_data["id"],
                        name=seq_data["name"],
                        description=seq_data["description"],
                        episode_ids=seq_data["episode_ids"],
                        start_time=datetime.fromisoformat(seq_data["start_time"]),
                        end_time=datetime.fromisoformat(seq_data["end_time"]),
                        tags=seq_data.get("tags", []),
                    )
                    self.sequences[sequence.id] = sequence

            except Exception as e:
                log.error(f"Error loading episodes: {e}")

    async def _save_episodes(self):
        """Save episodes to storage"""
        episodes_file = self.storage_path / "episodes.json"

        data = {
            "episodes": [ep.to_dict() for ep in self.episodes.values()],
            "sequences": [
                {
                    "id": seq.id,
                    "name": seq.name,
                    "description": seq.description,
                    "episode_ids": seq.episode_ids,
                    "start_time": seq.start_time.isoformat(),
                    "end_time": seq.end_time.isoformat(),
                    "tags": seq.tags,
                }
                for seq in self.sequences.values()
            ],
        }

        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _update_indexes(self, episode: Episode):
        """Update all indexes for an episode"""
        # Time index
        self._index_by_time.append((episode.timestamp, episode.id))
        self._index_by_time.sort(key=lambda x: x[0])

        # Type index
        self._index_by_type[episode.episode_type].add(episode.id)

        # Tag index
        for tag in episode.tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = set()
            self._index_by_tag[tag].add(episode.id)

        # Participant index
        for participant in episode.participants:
            if participant not in self._index_by_participant:
                self._index_by_participant[participant] = set()
            self._index_by_participant[participant].add(episode.id)

    async def record_episode(
        self,
        episode_type: EpisodeType,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        emotions: Optional[List[str]] = None,
        location: Optional[str] = None,
        participants: Optional[List[str]] = None,
        duration: Optional[timedelta] = None,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """Record a new episodic memory"""
        async with self._lock:
            episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            episode = Episode(
                id=episode_id,
                timestamp=datetime.now(),
                episode_type=episode_type,
                content=content,
                context=context or {},
                emotions=emotions or [],
                importance=importance,
                location=location,
                participants=participants or [],
                duration=duration,
                tags=tags or [],
            )

            self.episodes[episode_id] = episode
            self._update_indexes(episode)

            # Auto-save periodically
            if len(self.episodes) % 10 == 0:
                await self._save_episodes()

            log.debug(f"Recorded episode: {episode_id}")
            return episode

    async def record_conversation(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> Episode:
        """Record a conversation episode"""
        content = f"User: {user_message}\nAssistant: {assistant_response}"

        context = {"conversation_id": conversation_id} if conversation_id else {}

        return await self.record_episode(
            episode_type=EpisodeType.CONVERSATION,
            content=content,
            context=context,
            importance=importance,
            participants=participants or ["user", "assistant"],
            tags=["conversation", "dialogue"],
        )

    async def record_action(
        self,
        action: str,
        result: str,
        success: bool = True,
        duration: Optional[timedelta] = None,
        importance: float = 1.0,
    ) -> Episode:
        """Record an action episode"""
        content = f"Action: {action}\nResult: {result}"

        episode_type = EpisodeType.ACTION if success else EpisodeType.ERROR

        return await self.record_episode(
            episode_type=episode_type,
            content=content,
            context={"success": success},
            importance=importance,
            duration=duration,
            tags=["action", "result"],
        )

    async def record_observation(
        self,
        observation: str,
        details: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Episode:
        """Record an observation episode"""
        return await self.record_episode(
            episode_type=EpisodeType.OBSERVATION,
            content=observation,
            context=details or {},
            importance=importance,
            tags=["observation"],
        )

    async def record_learning(
        self,
        topic: str,
        what_learned: str,
        source: Optional[str] = None,
        importance: float = 1.5,
    ) -> Episode:
        """Record a learning episode"""
        content = f"Learned about {topic}: {what_learned}"

        context = {"source": source} if source else {}

        return await self.record_episode(
            episode_type=EpisodeType.LEARNING,
            content=content,
            context=context,
            importance=importance,
            tags=["learning", topic.lower().replace(" ", "_")],
        )

    async def retrieve_episodes(
        self,
        query: EpisodeQuery,
    ) -> List[Episode]:
        """Retrieve episodes matching query criteria"""
        # Start with all episodes
        candidates: Set[str] = set(self.episodes.keys())

        # Apply time filter
        if query.start_time or query.end_time:
            time_candidates = set()
            for timestamp, episode_id in self._index_by_time:
                if query.start_time and timestamp < query.start_time:
                    continue
                if query.end_time and timestamp > query.end_time:
                    continue
                time_candidates.add(episode_id)
            candidates &= time_candidates

        # Apply type filter
        if query.episode_types:
            type_candidates = set()
            for etype in query.episode_types:
                type_candidates |= self._index_by_type.get(etype, set())
            candidates &= type_candidates

        # Apply participant filter
        if query.participants:
            participant_candidates = set()
            for p in query.participants:
                participant_candidates |= self._index_by_participant.get(p, set())
            candidates &= participant_candidates

        # Apply tag filter
        if query.tags:
            tag_candidates = set()
            for tag in query.tags:
                tag_candidates |= self._index_by_tag.get(tag, set())
            candidates &= tag_candidates

        # Filter remaining candidates
        results = []
        for episode_id in candidates:
            episode = self.episodes.get(episode_id)
            if not episode:
                continue

            # Check importance
            if episode.importance < query.min_importance:
                continue

            # Check location
            if query.location and episode.location != query.location:
                continue

            # Check emotions
            if query.emotions and not any(e in episode.emotions for e in query.emotions):
                continue

            # Check keywords
            if query.keywords:
                content_lower = episode.content.lower()
                if not any(kw.lower() in content_lower for kw in query.keywords):
                    continue

            results.append(episode)

        # Sort by timestamp (most recent first)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[: query.limit]

    async def recall_recent(
        self,
        hours: int = 24,
        episode_types: Optional[List[EpisodeType]] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Recall recent episodes"""
        query = EpisodeQuery(
            start_time=datetime.now() - timedelta(hours=hours),
            episode_types=episode_types,
            limit=limit,
        )
        return await self.retrieve_episodes(query)

    async def recall_conversations(
        self,
        about: Optional[str] = None,
        with_participant: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Recall conversations"""
        query = EpisodeQuery(
            episode_types=[EpisodeType.CONVERSATION],
            keywords=[about] if about else None,
            participants=[with_participant] if with_participant else None,
            limit=limit,
        )
        return await self.retrieve_episodes(query)

    async def recall_actions(
        self,
        action_type: Optional[str] = None,
        success_only: bool = True,
        limit: int = 10,
    ) -> List[Episode]:
        """Recall actions taken"""
        types = [EpisodeType.ACTION]
        if not success_only:
            types.append(EpisodeType.ERROR)

        query = EpisodeQuery(
            episode_types=types,
            keywords=[action_type] if action_type else None,
            limit=limit,
        )
        return await self.retrieve_episodes(query)

    async def recall_by_time(
        self,
        time_description: str,
        limit: int = 10,
    ) -> List[Episode]:
        """Recall episodes by time description (e.g., 'yesterday', 'last week')"""
        start_time, end_time = self._parse_time_description(time_description)

        if not start_time:
            return []

        query = EpisodeQuery(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return await self.retrieve_episodes(query)

    async def get_episode_details(self, episode_id: str) -> Optional[Episode]:
        """Get detailed information about a specific episode"""
        return self.episodes.get(episode_id)

    async def find_related_episodes(
        self,
        episode_id: str,
        limit: int = 5,
    ) -> List[Episode]:
        """Find episodes related to a given episode"""
        episode = self.episodes.get(episode_id)
        if not episode:
            return []

        # Direct relationships
        related_ids = set(episode.related_episodes)

        # Find by shared context
        for ep_id, ep in self.episodes.items():
            if ep_id == episode_id:
                continue

            # Shared participants
            if set(ep.participants) & set(episode.participants):
                related_ids.add(ep_id)

            # Shared tags
            if set(ep.tags) & set(episode.tags):
                related_ids.add(ep_id)

            # Shared context keys
            if set(ep.context.keys()) & set(episode.context.keys()):
                related_ids.add(ep_id)

        # Score by temporal proximity
        scored = []
        for rel_id in related_ids:
            rel_ep = self.episodes.get(rel_id)
            if not rel_ep:
                continue

            time_diff = abs((rel_ep.timestamp - episode.timestamp).total_seconds())
            score = 1.0 / (1 + time_diff / 3600)  # Decay with time

            scored.append((score, rel_ep))

        scored.sort(reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def create_sequence(
        self,
        name: str,
        description: str,
        episode_ids: List[str],
        tags: Optional[List[str]] = None,
    ) -> Optional[TemporalSequence]:
        """Create a sequence from a list of episodes"""
        # Validate all episodes exist
        episodes = []
        for ep_id in episode_ids:
            ep = self.episodes.get(ep_id)
            if ep:
                episodes.append(ep)

        if len(episodes) < 2:
            return None

        # Sort by timestamp
        episodes.sort(key=lambda e: e.timestamp)

        sequence_id = f"seq_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        sequence = TemporalSequence(
            id=sequence_id,
            name=name,
            description=description,
            episode_ids=[ep.id for ep in episodes],
            start_time=episodes[0].timestamp,
            end_time=episodes[-1].timestamp,
            tags=tags or [],
        )

        async with self._lock:
            self.sequences[sequence_id] = sequence
            await self._save_episodes()

        return sequence

    async def get_sequence(self, sequence_id: str) -> Optional[TemporalSequence]:
        """Get a sequence by ID"""
        return self.sequences.get(sequence_id)

    async def summarize_period(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Generate a summary of a time period"""
        query = EpisodeQuery(
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        episodes = await self.retrieve_episodes(query)

        if not episodes:
            return {"summary": "No memories from this period", "episode_count": 0}

        # Count by type
        type_counts = {}
        for ep in episodes:
            etype = ep.episode_type.value
            type_counts[etype] = type_counts.get(etype, 0) + 1

        # Get top participants
        participant_counts = {}
        for ep in episodes:
            for p in ep.participants:
                participant_counts[p] = participant_counts.get(p, 0) + 1
        top_participants = sorted(participant_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get top tags
        tag_counts = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get important episodes
        important = [ep for ep in episodes if ep.importance > 1.5]
        important.sort(key=lambda e: e.importance, reverse=True)

        return {
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "episode_count": len(episodes),
            "by_type": type_counts,
            "top_participants": top_participants,
            "top_tags": top_tags,
            "important_episodes": [ep.to_dict() for ep in important[:5]],
            "sequences": len(
                [
                    s
                    for s in self.sequences.values()
                    if s.start_time >= start_time and s.end_time <= end_time
                ]
            ),
        }

    async def forget_old_episodes(
        self,
        older_than_days: int = 365,
        min_importance: float = 1.0,
    ) -> int:
        """Remove old, low-importance episodes"""
        cutoff = datetime.now() - timedelta(days=older_than_days)

        to_remove = []
        for episode_id, episode in self.episodes.items():
            if episode.timestamp < cutoff and episode.importance < min_importance:
                to_remove.append(episode_id)

        async with self._lock:
            for episode_id in to_remove:
                del self.episodes[episode_id]

            # Rebuild indexes
            self._rebuild_indexes()
            await self._save_episodes()

        log.info(f"Forgot {len(to_remove)} old episodes")
        return len(to_remove)

    def _rebuild_indexes(self):
        """Rebuild all indexes"""
        self._index_by_time = []
        self._index_by_type = {t: set() for t in EpisodeType}
        self._index_by_tag = {}
        self._index_by_participant = {}

        for episode in self.episodes.values():
            self._update_indexes(episode)

    def _parse_time_description(
        self, description: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse natural time descriptions"""
        now = datetime.now()
        desc_lower = description.lower()

        if "today" in desc_lower:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now

        elif "yesterday" in desc_lower:
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59)
            return start, end

        elif "last week" in desc_lower:
            start = now - timedelta(days=now.weekday() + 7)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59)
            return start, end

        elif "this week" in desc_lower:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now

        elif "last month" in desc_lower:
            last_month = now.replace(day=1) - timedelta(days=1)
            start = last_month.replace(day=1, hour=0, minute=0, second=0)
            end = last_month.replace(day=last_month.day, hour=23, minute=59, second=59)
            return start, end

        elif "this month" in desc_lower:
            start = now.replace(day=1, hour=0, minute=0, second=0)
            return start, now

        elif "last hour" in desc_lower:
            start = now - timedelta(hours=1)
            return start, now

        elif re.match(r"last (\d+) hours?", desc_lower):
            match = re.match(r"last (\d+) hours?", desc_lower)
            hours = int(match.group(1))
            start = now - timedelta(hours=hours)
            return start, now

        elif re.match(r"last (\d+) days?", desc_lower):
            match = re.match(r"last (\d+) days?", desc_lower)
            days = int(match.group(1))
            start = now - timedelta(days=days)
            return start, now

        return None, None

    async def close(self):
        """Close the episodic memory system"""
        await self._save_episodes()
        log.info("Episodic memory system closed")


# Global instance
_episodic_memory: Optional[EpisodicMemory] = None


async def get_episodic_memory() -> EpisodicMemory:
    """Get the global episodic memory instance"""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
        await _episodic_memory.initialize()
    return _episodic_memory


__all__ = [
    "EpisodicMemory",
    "Episode",
    "EpisodeType",
    "EpisodeQuery",
    "TemporalSequence",
    "get_episodic_memory",
]
