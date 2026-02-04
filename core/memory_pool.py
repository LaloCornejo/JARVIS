"""
Memory pool and object pooling utilities for JARVIS performance optimization.

This module provides memory-efficient object pooling, circular buffers,
and zero-allocation strategies to reduce GC pressure and memory fragmentation.
"""

import asyncio
import logging
import threading
from collections import deque
from typing import Any, Generic, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class ObjectPool(Generic[T]):
    """Generic object pool for reusable objects"""

    def __init__(self, factory: callable, max_size: int = 100, cleanup_func: callable = None):
        self.factory = factory
        self.max_size = max_size
        self.cleanup_func = cleanup_func
        self.pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0

    def acquire(self) -> T:
        """Acquire an object from the pool or create new one"""
        with self._lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                return obj
            else:
                self.created_count += 1
                return self.factory()

    def release(self, obj: T):
        """Return an object to the pool"""
        if obj is None:
            return

        with self._lock:
            if len(self.pool) < self.max_size:
                # Optional cleanup before returning to pool
                if self.cleanup_func:
                    try:
                        self.cleanup_func(obj)
                    except Exception as e:
                        log.warning(f"Error in cleanup function: {e}")

                self.pool.append(obj)

    def clear(self):
        """Clear all objects from the pool"""
        with self._lock:
            self.pool.clear()

    def stats(self) -> dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self.pool),
                "max_size": self.max_size,
                "created": self.created_count,
                "reused": self.reused_count,
                "hit_rate": self.reused_count / max(1, self.created_count + self.reused_count),
            }


class AudioBufferPool:
    """Specialized pool for audio buffers with zero-copy operations"""

    def __init__(
        self, sample_rate: int = 16000, chunk_duration_ms: int = 100, max_buffers: int = 20
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = (sample_rate * chunk_duration_ms) // 1000
        self.max_buffers = max_buffers

        # Create buffer pools for different sizes
        self.small_pool = ObjectPool(
            factory=lambda: self._create_buffer(self.chunk_samples),
            max_size=max_buffers,
            cleanup_func=self._clear_buffer,
        )

        self.medium_pool = ObjectPool(
            factory=lambda: self._create_buffer(self.chunk_samples * 3),  # 300ms
            max_size=max_buffers // 2,
            cleanup_func=self._clear_buffer,
        )

        self.large_pool = ObjectPool(
            factory=lambda: self._create_buffer(self.chunk_samples * 10),  # 1 second
            max_size=max_buffers // 4,
            cleanup_func=self._clear_buffer,
        )

        self.circular_pool = ObjectPool(
            factory=lambda: CircularAudioBuffer(max_samples=self.chunk_samples * 30),  # 3 seconds
            max_size=5,
            cleanup_func=lambda x: x.clear(),
        )

    def _create_buffer(self, size: int):
        """Create a new audio buffer"""
        import numpy as np

        return np.zeros(size, dtype=np.float32)

    def _clear_buffer(self, buffer):
        """Clear buffer contents"""
        buffer.fill(0)

    def get_buffer(self, min_samples: int = None):
        """Get appropriate buffer for the requested size"""
        if min_samples is None:
            min_samples = self.chunk_samples

        if min_samples <= self.chunk_samples:
            return self.small_pool.acquire()
        elif min_samples <= self.chunk_samples * 3:
            return self.medium_pool.acquire()
        elif min_samples <= self.chunk_samples * 10:
            return self.large_pool.acquire()
        else:
            # Create on-demand for very large buffers
            import numpy as np

            return np.zeros(min_samples, dtype=np.float32)

    def return_buffer(self, buffer):
        """Return buffer to appropriate pool"""
        if buffer is None:
            return

        size = len(buffer)
        if size == self.chunk_samples:
            self.small_pool.release(buffer)
        elif size == self.chunk_samples * 3:
            self.medium_pool.release(buffer)
        elif size == self.chunk_samples * 10:
            self.large_pool.release(buffer)
        # Very large buffers are not pooled to prevent memory bloat

    def get_circular_buffer(self):
        """Get a circular audio buffer"""
        return self.circular_pool.acquire()

    def return_circular_buffer(self, buffer):
        """Return circular buffer to pool"""
        self.circular_pool.release(buffer)

    def stats(self) -> dict[str, Any]:
        """Get comprehensive buffer pool statistics"""
        return {
            "small_pool": self.small_pool.stats(),
            "medium_pool": self.medium_pool.stats(),
            "large_pool": self.large_pool.stats(),
            "circular_pool": self.circular_pool.stats(),
            "sample_rate": self.sample_rate,
            "chunk_samples": self.chunk_samples,
        }


class CircularAudioBuffer:
    """Circular buffer for streaming audio with zero-copy operations"""

    def __init__(self, max_samples: int, dtype=None):
        import numpy as np

        if dtype is None:
            dtype = np.float32

        self.max_samples = max_samples
        self.dtype = dtype
        self.buffer = np.zeros(max_samples, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self._lock = threading.Lock()

    def write(self, data):
        """Write data to circular buffer"""
        with self._lock:
            data_len = len(data)

            if data_len > self.max_samples:
                # If data is larger than buffer, only keep the most recent samples
                data = data[-self.max_samples :]
                data_len = self.max_samples

            # Calculate available space
            available = self.max_samples - self.size

            if data_len > available:
                # Need to overwrite old data
                overwrite = data_len - available
                self.read_pos = (self.read_pos + overwrite) % self.max_samples
                self.size = self.max_samples
            else:
                self.size += data_len

            # Write data
            end_pos = (self.write_pos + data_len) % self.max_samples

            if end_pos > self.write_pos:
                # Single contiguous write
                self.buffer[self.write_pos : end_pos] = data
            else:
                # Wrap around write
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos :] = data[:first_part]
                self.buffer[:end_pos] = data[first_part:]

            self.write_pos = end_pos

    def read(self, samples: int = None) -> Optional[Any]:
        """Read data from circular buffer"""
        with self._lock:
            if self.size == 0:
                return None

            if samples is None:
                samples = self.size
            else:
                samples = min(samples, self.size)

            import numpy as np

            result = np.zeros(samples, dtype=self.dtype)

            # Calculate read positions
            end_pos = (self.read_pos + samples) % self.max_samples

            if end_pos > self.read_pos:
                # Single contiguous read
                result[:] = self.buffer[self.read_pos : end_pos]
            else:
                # Wrap around read
                first_part = self.max_samples - self.read_pos
                result[:first_part] = self.buffer[self.read_pos :]
                result[first_part:] = self.buffer[:end_pos]

            self.read_pos = end_pos
            self.size -= samples

            return result

    def peek(self, samples: int = None) -> Optional[Any]:
        """Peek at data without removing it"""
        with self._lock:
            if self.size == 0:
                return None

            if samples is None:
                samples = self.size
            else:
                samples = min(samples, self.size)

            import numpy as np

            result = np.zeros(samples, dtype=self.dtype)

            # Calculate peek positions (same as read but don't update positions)
            end_pos = (self.read_pos + samples) % self.max_samples

            if end_pos > self.read_pos:
                result[:] = self.buffer[self.read_pos : end_pos]
            else:
                first_part = self.max_samples - self.read_pos
                result[:first_part] = self.buffer[self.read_pos :]
                result[first_part:] = self.buffer[:end_pos]

            return result

    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self.write_pos = 0
            self.read_pos = 0
            self.size = 0
            self.buffer.fill(0)

    def available(self) -> int:
        """Get number of available samples"""
        with self._lock:
            return self.size

    def space_available(self) -> int:
        """Get available space for writing"""
        with self._lock:
            return self.max_samples - self.size


class MemoryOptimizedMessageHistory:
    """Memory-optimized message history with generational GC"""

    def __init__(self, max_messages: int = 1000, generations: int = 3):
        self.generations = [[] for _ in range(generations)]
        self.current_gen = 0
        self.max_per_gen = max_messages // generations
        self.access_counts = {}
        self._lock = asyncio.Lock()
        self.total_messages = 0

    async def add_message(self, message: dict, message_id: str = None):
        """Add message with generational promotion"""
        async with self._lock:
            if message_id is None:
                import uuid

                message_id = str(uuid.uuid4())

            # Add access count tracking
            self.access_counts[message_id] = 0

            self.generations[self.current_gen].append(
                {"id": message_id, "content": message, "timestamp": asyncio.get_event_loop().time()}
            )

            self.total_messages += 1

            # Check if current generation is full
            if len(self.generations[self.current_gen]) > self.max_per_gen:
                await self._promote_frequent_messages()
                self._rotate_generation()

    async def get_recent_messages(self, count: int, mark_accessed: bool = True) -> list[dict]:
        """Get recent messages with generational scanning"""
        async with self._lock:
            messages = []

            # Scan generations from newest to oldest
            for gen_idx in range(len(self.generations)):
                gen = (self.current_gen - gen_idx) % len(self.generations)

                for msg in reversed(self.generations[gen]):
                    if mark_accessed:
                        self.access_counts[msg["id"]] = self.access_counts.get(msg["id"], 0) + 1

                    messages.append(msg["content"])
                    if len(messages) >= count:
                        return messages[-count:]

            return messages

    async def get_message_by_id(self, message_id: str) -> Optional[dict]:
        """Get specific message by ID"""
        async with self._lock:
            for gen in self.generations:
                for msg in gen:
                    if msg["id"] == message_id:
                        self.access_counts[message_id] = self.access_counts.get(message_id, 0) + 1
                        return msg["content"]
            return None

    async def _promote_frequent_messages(self):
        """Promote frequently accessed messages to younger generations"""
        # Find messages accessed more than average
        total_accesses = sum(self.access_counts.values())
        avg_accesses = total_accesses / max(1, len(self.access_counts))

        promoted = []
        remaining = []

        for msg in self.generations[self.current_gen]:
            msg_id = msg["id"]
            access_count = self.access_counts.get(msg_id, 0)

            if access_count > avg_accesses * 1.5:  # 50% above average
                promoted.append(msg)
            else:
                remaining.append(msg)

        # Keep promoted messages in current generation, move others to next
        next_gen = (self.current_gen + 1) % len(self.generations)
        self.generations[next_gen].extend(remaining)
        self.generations[self.current_gen] = promoted

        # Clean up old access counts for removed messages
        current_ids = {msg["id"] for gen in self.generations for msg in gen}
        self.access_counts = {k: v for k, v in self.access_counts.items() if k in current_ids}

    def _rotate_generation(self):
        """Rotate to next generation"""
        self.current_gen = (self.current_gen + 1) % len(self.generations)

        # Clear oldest generation if it's the one being rotated to
        oldest_gen = (self.current_gen + 1) % len(self.generations)
        if len(self.generations[oldest_gen]) > 0:
            removed_count = len(self.generations[oldest_gen])
            self.total_messages -= removed_count

            # Clean up access counts
            removed_ids = {msg["id"] for msg in self.generations[oldest_gen]}
            self.access_counts = {
                k: v for k, v in self.access_counts.items() if k not in removed_ids
            }

            self.generations[oldest_gen].clear()

    async def clear_old_messages(self, max_age_seconds: float = 3600):
        """Clear messages older than specified age"""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            cutoff_time = current_time - max_age_seconds

            for gen in self.generations:
                gen[:] = [msg for msg in gen if msg["timestamp"] > cutoff_time]

            # Update total count
            self.total_messages = sum(len(gen) for gen in self.generations)

    def stats(self) -> dict[str, Any]:
        """Get memory history statistics"""
        return {
            "total_messages": self.total_messages,
            "generations": len(self.generations),
            "current_generation": self.current_gen,
            "messages_per_generation": [len(gen) for gen in self.generations],
            "access_tracking_count": len(self.access_counts),
            "avg_accesses": sum(self.access_counts.values()) / max(1, len(self.access_counts)),
        }


# Global memory pools
audio_buffer_pool = AudioBufferPool()
message_history = MemoryOptimizedMessageHistory()


def optimize_memory_usage():
    """Apply memory optimizations globally"""
    import gc

    # Force garbage collection
    collected = gc.collect()

    # Pre-allocate some common objects
    _preallocate_common_objects()

    log.info(f"Memory optimization completed, collected {collected} objects")


def _preallocate_common_objects():
    """Pre-allocate commonly used objects to reduce allocation overhead"""
    # Pre-allocate some audio buffers
    buffers = []
    for _ in range(5):
        buffers.append(audio_buffer_pool.get_buffer())

    # Return them to pool (now ready for use)
    for buffer in buffers:
        audio_buffer_pool.return_buffer(buffer)


# Initialize memory pools on import
try:
    optimize_memory_usage()
except Exception as e:
    log.warning(f"Failed to initialize memory optimizations: {e}")


__all__ = [
    "ObjectPool",
    "AudioBufferPool",
    "CircularAudioBuffer",
    "MemoryOptimizedMessageHistory",
    "audio_buffer_pool",
    "message_history",
    "optimize_memory_usage",
]
