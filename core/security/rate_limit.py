from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimitConfig:
    requests: int = 10
    window_seconds: int = 60


@dataclass
class RateLimitState:
    requests: list[float] = field(default_factory=list)
    blocked_until: float | None = None


class RateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, float | None]:
        async with self._lock:
            state = self._states[key]
            now = time.time()

            if state.blocked_until and now < state.blocked_until:
                return False, state.blocked_until

            cutoff = now - self.config.window_seconds
            state.requests = [t for t in state.requests if t > cutoff]

            if len(state.requests) >= self.config.requests:
                state.blocked_until = now + self.config.window_seconds
                return False, state.blocked_until

            state.requests.append(now)
            return True, None

    async def get_remaining(self, key: str) -> int:
        async with self._lock:
            state = self._states[key]
            now = time.time()
            cutoff = now - self.config.window_seconds
            state.requests = [t for t in state.requests if t > cutoff]
            return max(0, self.config.requests - len(state.requests))

    async def reset(self, key: str) -> None:
        async with self._lock:
            if key in self._states:
                del self._states[key]


class RateLimiterManager:
    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}

    def get_limiter(self, name: str, config: RateLimitConfig | None = None) -> RateLimiter:
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(config)
        return self._limiters[name]

    async def check(
        self, limiter_name: str, key: str, config: RateLimitConfig | None = None
    ) -> tuple[bool, float | None]:
        limiter = self.get_limiter(limiter_name, config)
        return await limiter.is_allowed(key)

    async def get_limiters_status(self) -> dict[str, dict[str, Any]]:
        status = {}
        for name, limiter in self._limiters.items():
            status[name] = {
                "config": {
                    "requests": limiter.config.requests,
                    "window_seconds": limiter.config.window_seconds,
                },
            }
        return status


rate_limiter_manager = RateLimiterManager()
