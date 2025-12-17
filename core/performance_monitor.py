: """
Performance monitoring and optimization for JARVIS TUI

Tracks key metrics to help identify bottlenecks and optimize the system.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Deque, Optional, Any

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
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
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
            "unit": self.unit
        }


class PerformanceMonitor:
    """Monitors and tracks TUI performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.enabled = True
        self.start_time = time.time()
        
        # Track various performance aspects
        self._init_tui_metrics()
        self._init_llm_metrics()
        self._init_audio_metrics()
        self._init_network_metrics()
    
    def _init_tui_metrics(self) -> None:
        """Initialize UI-related metrics"""
        self.metrics.update({
            "ui_response_time": Metric("UI Response Time", "ms"),
            "ui_frame_time": Metric("UI Frame Time", "ms"),
            "ui_refresh_rate": Metric("UI Refresh Rate", "fps"),
            "ui_input_latency": Metric("UI Input Latency", "ms"),
        })
    def _init_llm_metrics(self) -> None:
        """Initialize LLM-related metrics"""
        self.metrics.update({
            "llm_response_time": Metric("LLM Response Time", "ms"),
            "llm_token_rate": Metric("LLM Token Rate", "tokens/sec"),
            "llm_first_token_time": Metric("LLM First Token Time", "ms"),
            "tool_execution_time": Metric("Tool Execution Time", "ms"),
        })
    
    def _init_audio_metrics(self) -> None:
        """Initialize audio metrics"""
        self.metrics.update({
            "tts_start_time": Metric("TTS Start Time", "ms"),
            "tts_chunk_time": Metric("TTS Chunk Time", "ms"),
            "stt_recognition_time": Metric("STT Recognition Time", "ms"),
            "audio_processing_time": Metric("Audio Processing Time", "ms"),
        })
    
    def _init_network_metrics(self) -> None:
        """Initialize network metrics"""
        self.metrics.update({
            "network_latency": Metric("Network Latency", "ms"),
            "network_request_time": Metric("Network Request Time", "ms"),
            "connection_pool_usage": Metric("Connection Pool Usage", "percent"),
        })
    
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
        
        # Network Performance
        network_stats = stats.get("network_latency", {})
        if not network_stats.get("empty", False):
            lines.append(f"Network Latency: {network_stats.get('recent_mean', 0):.1f}ms avg")
        
        return "\n".join(lines)


class ConnectionPoolManager:
    """Optimizes HTTP connection pooling"""
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.stats = self._init_pool_metrics()
        log.info("Connection pool manager initialized")
    
    def _init_pool_metrics(self) -> Dict[str, Any]:
        """Initialize pool metrics"""
        return {
            "active_connection": 0,
            "total_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "closed_connections": 0
        }
    
    def get_or_create_pool(self, base_url: str, max_connections: int = 10, **kwargs) -> Any:
        """Get or create a connection pool for a base URL"""
        pool_key = base_url
        
        if pool_key not in self.pools:
            # Create new pool with optimal settings
            import httpx
            limits = httpx.Limits(
                max_keepalive_connections=max_connections, 
                max_connections=max_connections * 2,
                keepalive_expiry=30.0  # 30 second keepalive
            )
            
            async_client = httpx.AsyncClient(
                limits=limits,
                base_url=base_url,
                timeout=kwargs.get("timeout", 30.0),
                **kwargs
            )
            
            self.pools[pool_key] = async_client
            self.stats["total_connections"] += 1
            log.debug(f"Created new pool for {base_url}")
        
        else:
            self.stats["pool_hits"] += 1
            log.debug(f"Pool hit for {base_url}")
        
        statself.stats["active_connections"] = len(self.pools)
        return self.pools[pool_key]
    
    def close_pool(self, base_url: str) -> None:
        """Close a specific connection pool"""
        if base_url in self.pools:
            if hasattr(self.pools[base_url], 'aclose'):
                asyncio.create_task(self.pools[base_url].aclose())
            del self.pools[base_url]
            self.stats["closed_connections"] += 1
            log.debug(f"Closed pool for {base_url}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        pool_sizes = {
            url: {
                "active": hasattr(pool, "is_closed") and not pool.is_closed,
                "url": url
            }
            for url, pool in self.pools.items()
        }
        
        return {
            "pools": pool_sizes,
            **self.stats,
            "pool_efficiency": self.stats["pool_hits"] / max(1, self.stats["total_connections"])
        }
    
    def __del__(self):
        """Cleanup connection pools on deletion"""
        for pool in self.pools.values():
            if hasattr(pool, 'aclose') and hasattr(pool, 'is_closed') and not pool.is_closed:
                asyncio.create_task(pool.aclose())


# Global instances
performance_monitor = PerformanceMonitor()
connection_pool_manager = ConnectionPoolManager()

__all__ = [
    "PerformanceMonitor",
    "ConnectionPoolManager",
    "performance_monitor",
    "connection_pool_manager"
]