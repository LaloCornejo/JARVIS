"""Threading and concurrency management for JARVIS"""

import asyncio
import logging
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ThreadingManager:
    """Manages threading and concurrency for JARVIS components"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=2)  # For CPU-intensive tasks
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.tasks = set()

    async def __aenter__(self):
        self.running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

    def run_in_thread(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run a blocking function in a thread pool"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.thread_pool,
            self._wrap_function(func, *args, **kwargs)
        )
        self.tasks.add(future)
        future.add_done_callback(self.tasks.discard)
        return future

    def run_cpu_intensive(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run CPU-intensive function in process pool"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.process_pool,
            self._wrap_function(func, *args, **kwargs)
        )
        self.tasks.add(future)
        future.add_done_callback(self.tasks.discard)
        return future

    def _wrap_function(self, func: Callable, *args, **kwargs):
        """Wrap function to handle exceptions properly"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in threaded function {func.__name__}: {e}")
            raise

    async def cancel_all_tasks(self):
        """Cancel all running tasks"""
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()


class TaskCoordinator:
    """Coordinates concurrent tasks and prevents conflicts"""

    def __init__(self):
        self.active_tasks = {}
        self.lock = asyncio.Lock()

    async def start_task(self, task_id: str, coro) -> Optional[Any]:
        """Start a task with ID tracking"""
        async with self.lock:
            if task_id in self.active_tasks:
                # Cancel existing task with same ID
                existing_task = self.active_tasks[task_id]
                if not existing_task.done():
                    existing_task.cancel()

            task = asyncio.create_task(coro)
            self.active_tasks[task_id] = task

        try:
            result = await task
            return result
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            return None
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return None
        finally:
            async with self.lock:
                self.active_tasks.pop(task_id, None)

    async def cancel_task(self, task_id: str):
        """Cancel a specific task"""
        async with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if not task.done():
                    task.cancel()
                del self.active_tasks[task_id]


class StreamManager:
    """Manages streaming data flows"""

    def __init__(self):
        self.streams = {}
        self.lock = asyncio.Lock()

    async def create_stream(self, stream_id: str) -> asyncio.Queue:
        """Create a new stream"""
        async with self.lock:
            if stream_id not in self.streams:
                self.streams[stream_id] = asyncio.Queue()
            return self.streams[stream_id]

    async def push_to_stream(self, stream_id: str, data: Any):
        """Push data to a stream"""
        async with self.lock:
            if stream_id in self.streams:
                await self.streams[stream_id].put(data)

    async def close_stream(self, stream_id: str):
        """Close a stream"""
        async with self.lock:
            if stream_id in self.streams:
                # Signal end of stream
                await self.streams[stream_id].put(None)
                del self.streams[stream_id]


def non_blocking(func):
    """Decorator to make functions non-blocking"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Run in thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper


class ResourceManager:
    """Manages shared resources and prevents race conditions"""

    def __init__(self):
        self.resources = {}
        self.locks = {}

    async def acquire_resource(self, resource_id: str):
        """Acquire a lock for a resource"""
        if resource_id not in self.locks:
            self.locks[resource_id] = asyncio.Lock()
        return self.locks[resource_id]

    async def with_resource(self, resource_id: str, func):
        """Execute function with resource lock"""
        lock = await self.acquire_resource(resource_id)
        async with lock:
            return await func()


# Global instances
threading_manager = ThreadingManager()
task_coordinator = TaskCoordinator()
stream_manager = StreamManager()
resource_manager = ResourceManager()


__all__ = [
    'ThreadingManager',
    'TaskCoordinator',
    'StreamManager',
    'ResourceManager',
    'non_blocking',
    'threading_manager',
    'task_coordinator',
    'stream_manager',
    'resource_manager'
]
