:# Response Cache for JARVIS
import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time

log = logging.getLogger("jarvis.cache")


class ResponseCache:
    """LRU cache for frequent LLM and tool responses"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, prompt: str, system: str = None, **kwargs) -> str:
        """Generate cache key from request parameters"""
        key_data = {
            "prompt": prompt,
            "system": system or "",
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached response if valid"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                # Check TTL
                if datetime.fromisoformat(entry["expires_at"]) > datetime.now():
                    # Update access time for LRU
                    self.access_times[key] = time.time()
                    log.debug(f"Cache hit for key: {key[:8]}...")
                    return entry["data"]
                else:
                    # Remove expired entry
                    self._remove_expired_entry(key)
                    log.debug(f"Cache expired for key: {key[:8]}...")
        return None
    
    async def set(self, key: str, data: Any) -> None:
        """Store response in cache with TTL"""
        async with self._lock:
            # Remove entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru_entry()
            
            expires_at = (datetime.now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            self.cache[key] = {
                "data": data,
                "expires_at": expires_at,
                "created_at": datetime.now().isoformat()
            }
            self.access_times[key] = time.time()
            log.debug(f"Cached response for key: {key[:8]}...")
    
    def _remove_expired_entry(self, key: str) -> None:
        """Remove a single expired entry"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_lru_entry(self) -> None:
        """Remove least recently used entry"""
        if self.access_times:
            lru_key = min(self.access_times, key=self.access_times.get)
            self._remove_expired_entry(lru_key)
            log.debug(f"Evicted LRU entry: {lru_key[:8]}...")
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            log.info("Response cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            now = datetime.now()
            valid_entries = sum(
                1 for entry in self.cache.values()
                if datetime.fromisoformat(entry["expires_at"]) > now
            )
            return {
                "total_entries": len(self.cache),
                "valid_entries": valid_entries,
                "eviction_count": len(self.access_times) - valid_entries,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }


class IntentCache(ResponseCache):
    """Specialized cache for user intent and frequent requests"""
    
    def __init__(self):
        # Lower TTL for dynamic responses, higher size for intent caching
        super().__init__(max_size=200, ttl_seconds=180)
    
    def _generate_key(self, user_input: str, context: str = None) -> str:
        """Generate intent-based cache key"""
        # Extract key words for intent matching
        words = user_input.lower().split()
        key_words = [w for w in words if len(w) > 2 and not w.startswith("/")]
        
        # Common queries that can be cached
        cacheable_phrases = [
            "what time", "current time", "what date", "today is", "weather",
            "system info", "memory usage", "disk space", "help", "commands",
            "who created you", "what are you", "version"
        ]
        
        # Check if input matches cacheable patterns
        input_lower = user_input.lower()
        for phrase in cacheable_phrases:
            if phrase in input_lower:
                # Create simpler key for cacheable intents
                return f"intent:{phrase}:{hash(context or '')}"
        
        # For non-cacheable intents, use full hashing
        return super()._generate_key(user_input, context)


class ToolResponseCache(ResponseCache):
    """Specialized cache for tool execution results"""
    
    def __init__(self):
        # Cache tool results for longer since they're often slower
        super().__init__(max_size=150, ttl_seconds=600)
    
    def generate_tool_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate cache key for tool responses"""
        # Remove dynamic data like timestamps from cache key
        clean_args = args.copy()
        cleanup_keys = ["timestamp", "time", "now", "id", "random"]
        for key in cleanup_keys:
            clean_args.pop(key, None)
        
        return self._generate_key(f"tool:{tool_name}", None, **clean_args)


# Global instances
response_cache = ResponseCache()
intent_cache = IntentCache()
tool_cache = ToolResponseCache()


def should_cache_response(user_input: str, response_content: str) -> bool:
    """Determine if a response should be cached based on request/response characteristics"""
    # Don't cache very short or very long responses
    if len(response_content) < 50 or len(response_content) > 2000:
        return False
    
    # Don't cache error responses
    if response_content.lower().strip() in ["error", "failed", "unavailable"]:
        return False
    
    # Cache simple factual queries
    cacheable_prefixes = [
        "what time", "current time", "what date", "system", "memory", 
        "disk space", "cpu usage", "help", "commands"
    ]
    
    input_lower = user_input.lower().strip()
    for prefix in cacheable_prefixes:
        if input_lower.startswith(prefix):
            return True
    
    # Cache simple greeting responses
    greeting_responses = ["hello", "hi ", "greetings", "welcome"]
    return any(grt in response_content.lower() for grt in greeting_responses)


__all__ = [
    "ResponseCache", 
    "IntentCache", 
    "ToolResponseCache",
    "response_cache", 
    "intent_cache", 
    "tool_cache",
    "should_cache_response"
]