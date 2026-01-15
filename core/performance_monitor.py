"""
Performance monitoring and optimization for JARVIS TUI

Tracks key metrics to help identify bottlenecks and optimize the system.
"""

import asyncio
import gc
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

log = logging.getLogger("jarvis.performance")


@dataclass
class MetricPoint:
    """Single data point for a metric"""

    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Performance metric with historical data"""

    name: str
    unit: str
    window_size: int = 60  # Keep last 60 points
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=60))

    def record(self, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a new metric value"""
        point = MetricPoint(timestamp=time.time(), value=value, tags=tags or {})
        self.points.append(point)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistical summary of metric"""
        if not self.points:
            return {"empty": True}

        values = [p.value for p in self.points]
        recent_values = values[-10:]  # Last 10 values

        return {
            "current": values[-1],
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "recent_mean": statistics.mean(recent_values),
            "samples": len(values),
            "unit": self.unit,
        }


class PerformanceMonitor:
    """Monitors and tracks TUI performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.enabled = True
        self.start_time = time.time()

        # Memory management
        self._memory_threshold_mb = 1000  # Trigger GC when memory exceeds 1GB
        self._last_gc_time = 0
        self._gc_cooldown_seconds = 30  # Minimum time between GC runs

        # Track various performance aspects
        self._init_tui_metrics()
        self._init_llm_metrics()
        self._init_audio_metrics()
        self._init_network_metrics()
        self._init_system_metrics()

    def _init_tui_metrics(self) -> None:
        """Initialize UI-related metrics"""
        self.metrics.update(
            {
                "ui_response_time": Metric("UI Response Time", "ms"),
                "ui_frame_time": Metric("UI Frame Time", "ms"),
                "ui_refresh_rate": Metric("UI Refresh Rate", "fps"),
                "ui_input_latency": Metric("UI Input Latency", "ms"),
            }
        )

    def _init_llm_metrics(self) -> None:
        """Initialize LLM-related metrics"""
        self.metrics.update(
            {
                "llm_response_time": Metric("LLM Response Time", "ms"),
                "llm_token_rate": Metric("LLM Token Rate", "tokens/sec"),
                "llm_first_token_time": Metric("LLM First Token Time", "ms"),
                "tool_execution_time": Metric("Tool Execution Time", "ms"),
                "tool_usage_count": Metric("Tool Usage Count", "calls"),
                "cache_hit_rate": Metric("Cache Hit Rate", "percent"),
            }
        )

    def _init_audio_metrics(self) -> None:
        """Initialize audio metrics"""
        self.metrics.update(
            {
                "tts_start_time": Metric("TTS Start Time", "ms"),
                "tts_chunk_time": Metric("TTS Chunk Time", "ms"),
                "stt_recognition_time": Metric("STT Recognition Time", "ms"),
                "audio_processing_time": Metric("Audio Processing Time", "ms"),
            }
        )

    def _init_network_metrics(self) -> None:
        """Initialize network metrics"""
        self.metrics.update(
            {
                "network_latency": Metric("Network Latency", "ms"),
                "network_request_time": Metric("Network Request Time", "ms"),
                "connection_pool_usage": Metric("Connection Pool Usage", "percent"),
            }
        )

    def _init_system_metrics(self) -> None:
        """Initialize system metrics"""
        self.metrics.update(
            {
                "memory_usage": Metric("Memory Usage", "MB"),
                "cpu_usage": Metric("CPU Usage", "percent"),
                "gc_collections": Metric("GC Collections", "collections"),
                "gc_collected": Metric("GC Collected Objects", "objects"),
            }
        )

    def record_ui_response(self, duration_ms: float, operation: str = "") -> None:
        """Record UI response time"""
        if not self.enabled:
            return

        tags = {"operation": operation} if operation else {}
        self.metrics["ui_response_time"].record(duration_ms, tags)
        log.debug(f"UI Response: {duration_ms:.1f}ms{operation and f' ({operation})'} ")

    def record_llm_response(self, duration_ms: float, success: bool) -> None:
        """Record LLM response time"""
        if not self.enabled:
            return

        tags = {"success": str(success)}
        self.metrics["llm_response_time"].record(duration_ms, tags)
        log.debug(f"LLM Response: {duration_ms:.1f}ms, Success: {success}")

    def record_tts_start(self, start_time: float) -> None:
        """Record TTS start time from a timestamp"""
        if not self.enabled:
            return

        duration = (time.time() - start_time) * 1000
        self.metrics["tts_start_time"].record(duration)

    def record_network_latency(self, latency_ms: float, endpoint: str = "") -> None:
        """Record network latency"""
        if not self.enabled:
            return

        tags = {"endpoint": endpoint} if endpoint else {}
        self.metrics["network_latency"].record(latency_ms, tags)
        log.debug(f"Network Latency: {latency_ms:.1f}ms{endpoint and f' ({endpoint})'}")

    def record_tool_usage(self, tool_name: str) -> None:
        """Record tool usage count"""
        if not self.enabled:
            return

        self.metrics["tool_usage_count"].record(1.0, {"tool": tool_name})
        log.debug(f"Tool used: {tool_name}")

    def record_cache_hit(self, cache_type: str, hit: bool) -> None:
        """Record cache hit/miss"""
        if not self.enabled:
            return

        # Record as percentage (1.0 for hit, 0.0 for miss)
        self.metrics["cache_hit_rate"].record(1.0 if hit else 0.0, {"type": cache_type})
        log.debug(f"Cache {cache_type}: {'hit' if hit else 'miss'}")

    def record_system_stats(self) -> None:
        """Record current system statistics and manage memory pressure"""
        if not self.enabled:
            return

        try:
            import os

            import psutil

            # Memory usage in MB
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics["memory_usage"].record(memory_mb)

            # Check for memory pressure and trigger garbage collection if needed
            self._manage_memory_pressure(memory_mb)

            # CPU usage percentage
            cpu_percent = process.cpu_percent(interval=0.1)
            self.metrics["cpu_usage"].record(cpu_percent)

            # Don't log system stats to avoid debug file spam
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            log.debug(f"Failed to record system stats: {e}")

    def _manage_memory_pressure(self, current_memory_mb: float) -> None:
        """Manage memory pressure by triggering garbage collection when needed"""
        current_time = time.time()

        # Check if we should trigger garbage collection
        if (
            current_memory_mb > self._memory_threshold_mb
            and current_time - self._last_gc_time > self._gc_cooldown_seconds
        ):
            # Log memory pressure situation
            log.debug(
                f"Memory pressure detected: {current_memory_mb:.1f}MB > "
                f"{self._memory_threshold_mb}MB"
            )

            # Perform garbage collection
            collected = gc.collect()
            uncollectable = gc.garbage

            # Record GC activity
            if "gc_collections" not in self.metrics:
                self.metrics["gc_collections"] = Metric("GC Collections", "collections")
                self.metrics["gc_collected"] = Metric("GC Collected Objects", "objects")

            self.metrics["gc_collections"].record(1.0)
            self.metrics["gc_collected"].record(float(collected))

            log.debug(
                f"Garbage collection: {collected} objects collected, "
                f"{len(uncollectable)} uncollectable"
            )

            # Update last GC time
            self._last_gc_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = metric.get_stats()

        # Add runtime metrics
        stats["uptime_seconds"] = time.time() - self.start_time
        stats["monitor_enabled"] = self.enabled

        return stats

    def format_performance_summary(self) -> str:
        """Format metrics as readable summary for UI display"""
        stats = self.get_stats()

        lines = []
        lines.append("Performance Metrics")
        lines.append("-" * 20)

        # UI Performance
        ui_stats = stats.get("ui_response_time", {})
        if not ui_stats.get("empty", False):
            lines.append(f"UI Response: {ui_stats.get('recent_mean', 0):.1f}ms avg")

        # LLM Performance
        llm_stats = stats.get("llm_response_time", {})
        if not llm_stats.get("empty", False):
            lines.append(f"LLM Response: {llm_stats.get('recent_mean', 0):.1f}ms avg")

        # Audio Performance
        tts_stats = stats.get("tts_start_time", {})
        if not tts_stats.get("empty", False):
            lines.append(f"TTS Start: {tts_stats.get('recent_mean', 0):.1f}ms avg")

        # Tool Usage
        tool_count_stats = stats.get("tool_usage_count", {})
        if not tool_count_stats.get("empty", False):
            total_calls = tool_count_stats.get("samples", 0)
            lines.append(f"Tool Calls: {total_calls}")

        # Cache Performance
        cache_stats = stats.get("cache_hit_rate", {})
        if not cache_stats.get("empty", False):
            hit_rate = cache_stats.get("recent_mean", 0) * 100
            lines.append(f"Cache Hit Rate: {hit_rate:.1f}%")

        # Network Performance
        network_stats = stats.get("network_latency", {})
        if not network_stats.get("empty", False):
            lines.append(f"Network Latency: {network_stats.get('recent_mean', 0):.1f}ms avg")

        # System Resources
        memory_stats = stats.get("memory_usage", {})
        if not memory_stats.get("empty", False):
            lines.append(f"Memory Usage: {memory_stats.get('current', 0):.1f}MB")

        cpu_stats = stats.get("cpu_usage", {})
        if not cpu_stats.get("empty", False):
            lines.append(f"CPU Usage: {cpu_stats.get('current', 0):.1f}%")

        # Garbage Collection
        gc_stats = stats.get("gc_collected", {})
        if not gc_stats.get("empty", False):
            total_collected = gc_stats.get("samples", 0)
            lines.append(f"GC Objects Collected: {int(total_collected)}")

        return "\n".join(lines)


class OptimizedConnectionPool:
    """High-performance HTTP connection pooling with HTTP/2 and multiplexing"""

    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.prewarmed_urls: set[str] = set()
        self.stats = self._init_pool_metrics()
        self._prewarm_task = None
        log.info("Optimized connection pool manager initialized with HTTP/2 support")

    def _init_pool_metrics(self) -> Dict[str, Any]:
        """Initialize comprehensive pool metrics"""
        return {
            "active_connections": 0,
            "total_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "closed_connections": 0,
            "http2_connections": 0,
            "multiplexed_requests": 0,
            "connection_reuse_rate": 0.0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
        }

    async def get_or_create_pool(self, base_url: str, max_connections: int = 20, **kwargs) -> Any:
        """Get or create an optimized connection pool with HTTP/2 support"""
        pool_key = base_url

        if pool_key not in self.pools:
            # Create new pool with HTTP/2 and optimal settings
            import httpx

            # HTTP/2 enabled client with multiplexing
            async_client = httpx.AsyncClient(
                http2=True,  # Enable HTTP/2 multiplexing
                limits=httpx.Limits(
                    max_keepalive_connections=max_connections,
                    max_connections=max_connections,
                    keepalive_expiry=120.0,  # Extended keepalive for better reuse
                ),
                timeout=httpx.Timeout(
                    timeout=30.0,
                    connect=5.0,  # Faster connection establishment
                    read=30.0,
                    write=10.0,
                    pool=5.0,
                ),
                base_url=base_url,
                **kwargs,
            )

            self.pools[pool_key] = async_client
            self.stats["total_connections"] += 1
            log.debug(f"Created HTTP/2 optimized pool for {base_url}")

            # Start background prewarming for frequently used endpoints
            if not self._prewarm_task and len(self.pools) <= 5:  # Limit prewarming
                self._prewarm_task = asyncio.create_task(self._prewarm_connections())

        else:
            self.stats["pool_hits"] += 1
            log.debug(f"Pool hit for {base_url}")

        self.stats["active_connections"] = len(self.pools)
        return self.pools[pool_key]

    async def _prewarm_connections(self):
        """Prewarm connections for frequently used endpoints"""
        try:
            await asyncio.sleep(1)  # Brief delay to allow pool creation

            prewarm_endpoints = [
                ("http://localhost:11434", "/api/tags"),  # Ollama
                ("http://localhost:11434", "/api/generate"),
                ("http://localhost:8020", "/api/tts"),  # XTTS
                ("https://api.github.com", "/user"),  # GitHub API
                ("https://www.googleapis.com", "/oauth2/v1/tokeninfo"),  # Google APIs
            ]

            for base_url, endpoint in prewarm_endpoints:
                try:
                    if base_url in self.pools:
                        client = self.pools[base_url]
                        # Send lightweight HEAD request to establish connection
                        await client.head(endpoint, timeout=2.0)
                        self.prewarmed_urls.add(f"{base_url}{endpoint}")
                        log.debug(f"Prewarmed connection to {base_url}{endpoint}")
                        await asyncio.sleep(0.1)  # Small delay between prewarm requests
                except Exception as e:
                    log.debug(f"Prewarm failed for {base_url}{endpoint}: {e}")

        except Exception as e:
            log.warning(f"Connection prewarming failed: {e}")
        finally:
            self._prewarm_task = None

    async def batch_requests(self, requests: list[dict]) -> list[dict]:
        """Execute multiple requests in parallel with connection multiplexing"""
        if not requests:
            return []

        # Group requests by base URL for optimal connection reuse
        url_groups = {}
        for req in requests:
            url = req.get("url", "")
            base_url = self._extract_base_url(url)
            if base_url not in url_groups:
                url_groups[base_url] = []
            url_groups[base_url].append(req)

        all_results = []
        tasks = []

        # Execute requests for each URL group concurrently
        for base_url, group_requests in url_groups.items():
            task = asyncio.create_task(self._execute_url_group(base_url, group_requests))
            tasks.append(task)

        # Gather all group results
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        for result in group_results:
            if isinstance(result, Exception):
                log.error(f"Request batch failed: {result}")
                continue
            all_results.extend(result)

        self.stats["multiplexed_requests"] += len(all_results)
        return all_results

    async def _execute_url_group(self, base_url: str, requests: list[dict]) -> list[dict]:
        """Execute requests for a single base URL using connection multiplexing"""
        client = await self.get_or_create_pool(base_url)
        results = []

        # Execute requests concurrently using the same connection pool
        tasks = []
        for req in requests:
            task = asyncio.create_task(self._execute_single_request(client, req))
            tasks.append(task)

        # Wait for all requests in this group
        request_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(request_results):
            if isinstance(result, Exception):
                log.error(f"Request {i} failed: {result}")
                results.append({"error": str(result), "status": 500})
            else:
                results.append(result)

        return results

    async def _execute_single_request(self, client, request: dict) -> dict:
        """Execute a single HTTP request"""
        method = request.get("method", "GET")
        url = request.get("url", "")
        headers = request.get("headers", {})
        data = request.get("data")
        json_data = request.get("json")
        timeout = request.get("timeout", 10.0)

        start_time = asyncio.get_event_loop().time()

        try:
            response = await client.request(
                method=method, url=url, headers=headers, data=data, json=json_data, timeout=timeout
            )

            response_time = asyncio.get_event_loop().time() - start_time

            # Update metrics
            self._update_response_metrics(response_time, response.status_code)

            return {
                "status": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "json": response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else None,
                "response_time": response_time,
            }

        except Exception:
            response_time = asyncio.get_event_loop().time() - start_time
            self._update_response_metrics(response_time, 500)
            raise

    def _update_response_metrics(self, response_time: float, status_code: int):
        """Update performance metrics"""
        # Update average response time (exponential moving average)
        alpha = 0.1
        self.stats["avg_response_time"] = (
            alpha * response_time + (1 - alpha) * self.stats["avg_response_time"]
        )

        # Update error rate
        is_error = status_code >= 400
        error_alpha = 0.05  # Slower update for error rate
        self.stats["error_rate"] = (
            error_alpha * (1 if is_error else 0) + (1 - error_alpha) * self.stats["error_rate"]
        )

    def _extract_base_url(self, url: str) -> str:
        """Extract base URL from full URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    async def close_pool(self, base_url: str) -> None:
        """Close a specific connection pool"""
        if base_url in self.pools:
            try:
                await self.pools[base_url].aclose()
            except Exception as e:
                log.warning(f"Error closing pool for {base_url}: {e}")

            del self.pools[base_url]
            self.stats["closed_connections"] += 1
            log.debug(f"Closed optimized pool for {base_url}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        pool_info = {}
        http2_count = 0

        for url, pool in self.pools.items():
            is_http2 = getattr(pool, "_http2", False)
            if is_http2:
                http2_count += 1

            pool_info[url] = {
                "active": not getattr(pool, "is_closed", True),
                "http2": is_http2,
                "prewarmed": any(url in pw for pw in self.prewarmed_urls),
                "url": url,
            }

        self.stats["http2_connections"] = http2_count
        self.stats["connection_reuse_rate"] = self.stats["pool_hits"] / max(
            1, self.stats["pool_hits"] + self.stats["pool_misses"]
        )

        return {
            "pools": pool_info,
            **self.stats,
        }

    async def cleanup_idle_connections(self, max_idle_time: float = 300.0):
        """Clean up idle connections to free resources"""
        current_time = asyncio.get_event_loop().time()
        to_close = []

        for url, pool in self.pools.items():
            # Check if pool has been idle too long
            if hasattr(pool, "_last_used"):
                idle_time = current_time - pool._last_used
                if idle_time > max_idle_time:
                    to_close.append(url)

        for url in to_close:
            await self.close_pool(url)
            log.info(f"Closed idle connection pool for {url}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all pools
        close_tasks = []
        for url in list(self.pools.keys()):
            close_tasks.append(self.close_pool(url))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)


# Backward compatibility alias
ConnectionPoolManager = OptimizedConnectionPool


# Global instances
performance_monitor = PerformanceMonitor()
connection_pool_manager = OptimizedConnectionPool()


__all__ = [
    "PerformanceMonitor",
    "OptimizedConnectionPool",
    "ConnectionPoolManager",  # Backward compatibility
    "performance_monitor",
    "connection_pool_manager",
]
