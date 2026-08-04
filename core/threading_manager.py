"""Threading and concurrency management for JARVIS"""

import asyncio
import logging
import queue
import time
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
            self.thread_pool, lambda: self._wrap_function(func, *args, **kwargs)
        )
        self.tasks.add(future)
        future.add_done_callback(self.tasks.discard)
        return future

    def run_cpu_intensive(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run CPU-intensive function in process pool"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.process_pool, lambda: self._wrap_function(func, *args, **kwargs)
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


class LoadMonitor:
    """Monitors system load for dynamic scaling decisions"""

    def __init__(self, sample_window: float = 5.0):
        self.sample_window = sample_window
        self.cpu_samples = []
        self.memory_samples = []
        self.last_sample_time = 0
        self._lock = asyncio.Lock()

    async def get_system_load(self) -> dict[str, float]:
        """Get current system load metrics"""
        async with self._lock:
            current_time = time.time()

            # Sample every sample_window seconds
            if current_time - self.last_sample_time >= self.sample_window:
                try:
                    import psutil

                    # Get CPU usage (average over 1 second)
                    cpu_percent = psutil.cpu_percent(interval=0.1)

                    # Get memory usage
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent

                    # Store samples (keep last 10 samples for smoothing)
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_percent)

                    if len(self.cpu_samples) > 10:
                        self.cpu_samples.pop(0)
                    if len(self.memory_samples) > 10:
                        self.memory_samples.pop(0)

                    self.last_sample_time = current_time

                except ImportError:
                    # Fallback if psutil not available
                    logger.warning("psutil not available, using basic load monitoring")
                    cpu_percent = 50.0  # Assume moderate load
                    memory_percent = 50.0

            # Calculate smoothed averages
            avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 50.0
            avg_memory = (
                sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 50.0
            )

            return {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "combined_load": (avg_cpu + avg_memory) / 2,
            }

    def should_scale_up(self, current_load: float, current_workers: int, max_workers: int) -> bool:
        """Determine if thread pool should scale up"""
        return current_load > 70.0 and current_workers < max_workers

    def should_scale_down(
        self, current_load: float, current_workers: int, min_workers: int
    ) -> bool:
        """Determine if thread pool should scale down"""
        return current_load < 30.0 and current_workers > min_workers


class DynamicThreadingManager:
    """Dynamic thread pool manager with auto-scaling based on system load"""

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 16,
        min_processes: int = 1,
        max_processes: int = 4,
        scale_interval: float = 10.0,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_processes = min_processes
        self.max_processes = max_processes
        self.scale_interval = scale_interval

        # Current pool sizes
        self.current_thread_workers = min_workers
        self.current_process_workers = min_processes

        # Initialize pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_thread_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_process_workers)

        # Load monitoring
        self.load_monitor = LoadMonitor()
        self.last_scale_time = 0

        # Queues and state
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.tasks = set()

        # Start background scaling task
        self._scaling_task = None

    async def __aenter__(self):
        self.running = True
        self._scaling_task = asyncio.create_task(self._auto_scale_pools())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

    async def _auto_scale_pools(self):
        """Background task to automatically scale thread pools based on load"""
        while self.running:
            try:
                await asyncio.sleep(self.scale_interval)
                await self._check_and_scale_pools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")

    async def _check_and_scale_pools(self):
        """Check system load and scale pools accordingly"""
        load_metrics = await self.load_monitor.get_system_load()
        combined_load = load_metrics["combined_load"]
        current_time = time.time()

        # Only scale if enough time has passed since last scale
        if current_time - self.last_scale_time < self.scale_interval:
            return

        # Scale thread pool
        if self.load_monitor.should_scale_up(
            combined_load, self.current_thread_workers, self.max_workers
        ):
            new_thread_workers = min(self.current_thread_workers * 2, self.max_workers)
            if new_thread_workers != self.current_thread_workers:
                await self._scale_thread_pool(new_thread_workers)
                logger.info(
                    f"Scaled thread pool from {self.current_thread_workers} to "
                    f"{new_thread_workers} workers (load: {combined_load:.1f}%)"
                )

        elif self.load_monitor.should_scale_down(
            combined_load, self.current_thread_workers, self.min_workers
        ):
            new_thread_workers = max(self.current_thread_workers // 2, self.min_workers)
            if new_thread_workers != self.current_thread_workers:
                await self._scale_thread_pool(new_thread_workers)
                logger.info(
                    f"Scaled thread pool from {self.current_thread_workers} to "
                    f"{new_thread_workers} workers (load: {combined_load:.1f}%)"
                )

        # Scale process pool (more conservative scaling)
        if combined_load > 80.0 and self.current_process_workers < self.max_processes:
            new_process_workers = min(self.current_process_workers + 1, self.max_processes)
            if new_process_workers != self.current_process_workers:
                await self._scale_process_pool(new_process_workers)
                logger.info(
                    f"Scaled process pool from {self.current_process_workers} to "
                    f"{new_process_workers} workers"
                )

        elif combined_load < 20.0 and self.current_process_workers > self.min_processes:
            new_process_workers = max(self.current_process_workers - 1, self.min_processes)
            if new_process_workers != self.current_process_workers:
                await self._scale_process_pool(new_process_workers)
                logger.info(
                    f"Scaled process pool from {self.current_process_workers} to "
                    f"{new_process_workers} workers"
                )

    async def _scale_thread_pool(self, new_workers: int):
        """Scale thread pool to new size"""
        # Shutdown current pool
        self.thread_pool.shutdown(wait=True)

        # Create new pool with new size
        self.current_thread_workers = new_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
        self.last_scale_time = time.time()

    async def _scale_process_pool(self, new_workers: int):
        """Scale process pool to new size"""
        # Shutdown current pool
        self.process_pool.shutdown(wait=True)

        # Create new pool with new size
        self.current_process_workers = new_workers
        self.process_pool = ProcessPoolExecutor(max_workers=new_workers)

    def run_in_thread(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run a blocking function in the dynamic thread pool"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.thread_pool, lambda: self._wrap_function(func, *args, **kwargs)
        )
        self.tasks.add(future)
        future.add_done_callback(self.tasks.discard)
        return future

    def run_cpu_intensive(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run CPU-intensive function in the dynamic process pool"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.process_pool, lambda: self._wrap_function(func, *args, **kwargs)
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

    def get_pool_stats(self) -> dict[str, Any]:
        """Get current pool statistics"""
        return {
            "thread_workers": self.current_thread_workers,
            "process_workers": self.current_process_workers,
            "active_threads": len([t for t in self.tasks if not t.done()]),
            "total_tasks": len(self.tasks),
        }


class PriorityTaskCoordinator:
    """Task coordinator with priority-based scheduling"""

    def __init__(self):
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        self.active_tasks = {}
        self.processing = False
        self.lock = asyncio.Lock()

    async def start_priority_task(self, task_id: str, coro, priority: str = "normal"):
        """Start task with specified priority"""
        queue_map = {
            "high": self.high_priority_queue,
            "normal": self.normal_priority_queue,
            "low": self.low_priority_queue,
        }

        if priority not in queue_map:
            priority = "normal"

        await queue_map[priority].put((task_id, coro))

        async with self.lock:
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_priority_tasks())

    async def _process_priority_tasks(self):
        """Process tasks by priority order"""
        try:
            while True:
                # Process high priority first
                if not self.high_priority_queue.empty():
                    task_id, coro = await self.high_priority_queue.get()
                elif not self.normal_priority_queue.empty():
                    task_id, coro = await self.normal_priority_queue.get()
                elif not self.low_priority_queue.empty():
                    task_id, coro = await self.low_priority_queue.get()
                else:
                    # No tasks remaining
                    async with self.lock:
                        self.processing = False
                    break

                # Check if task with same ID already running
                async with self.lock:
                    if task_id in self.active_tasks:
                        existing_task = self.active_tasks[task_id]
                        if not existing_task.done():
                            existing_task.cancel()

                    task = asyncio.create_task(coro)
                    self.active_tasks[task_id] = task

                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Task {task_id} was cancelled")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                finally:
                    async with self.lock:
                        self.active_tasks.pop(task_id, None)

        except Exception as e:
            logger.error(f"Error in priority task processing: {e}")
            async with self.lock:
                self.processing = False

    async def cancel_task(self, task_id: str):
        """Cancel a specific task"""
        async with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if not task.done():
                    task.cancel()
                del self.active_tasks[task_id]


# Global instances
threading_manager = ThreadingManager()
dynamic_threading_manager = DynamicThreadingManager()
task_coordinator = TaskCoordinator()
priority_task_coordinator = PriorityTaskCoordinator()
stream_manager = StreamManager()
resource_manager = ResourceManager()


__all__ = [
    "ThreadingManager",
    "DynamicThreadingManager",
    "TaskCoordinator",
    "PriorityTaskCoordinator",
    "StreamManager",
    "ResourceManager",
    "LoadMonitor",
    "non_blocking",
    "threading_manager",
    "dynamic_threading_manager",
    "task_coordinator",
    "priority_task_coordinator",
    "stream_manager",
    "resource_manager",
]
