"""
Per-tenant rate limiting (a burst guard; the daily budget is quota_service's job).

Fixed window counter: "at most N hits per key per W seconds". Chosen over a token bucket because
it is trivially atomic in Redis (INCR + EXPIRE) and trivially testable. Known weakness: a client can
send N at the end of one window and N at the start of the next -- up to 2N in a short span. For
protecting a shared platform from a noisy neighbour that is acceptable; for a strict API contract use
a sliding window or token bucket.

Backends:
  memory -- a dict in this process. Correct for one process; with N API processes the real limit is
            N x the configured one.
  redis  -- shared by every process. If Redis is unreachable the limiter FAILS OPEN (and counts it):
            a cache outage must not become a platform outage.
"""
import logging
import time

from app.core import metrics
from app.core.config import settings

logger = logging.getLogger("ratelimit")


class MemoryLimiter:
    def __init__(self, clock=time.time):
        self._clock = clock
        self._windows: dict[str, tuple[int, int]] = {}  # key -> (window index, count)
        self._last_prune = 0.0

    async def hit(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = self._clock()
        index = int(now // window)
        self._prune(now, window)
        win, count = self._windows.get(key, (index, 0))
        if win != index:
            count = 0
        count += 1
        self._windows[key] = (index, count)
        retry_after = max(1, int((index + 1) * window - now) + 1)
        return count <= limit, retry_after

    def _prune(self, now: float, window: int) -> None:
        if now - self._last_prune < 60:
            return
        self._last_prune = now
        current = int(now // window)
        self._windows = {k: v for k, v in self._windows.items() if v[0] >= current - 1}


class RedisLimiter:
    def __init__(self, url: str, clock=time.time):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25)
        self._clock = clock

    async def hit(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = self._clock()
        index = int(now // window)
        redis_key = f"rl:{key}:{index}"
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, window + 1)
                count, _ = await pipe.execute()
        except Exception:
            logger.exception("rate limiter backend failed; failing open")
            metrics.RATE_LIMITER_ERRORS.inc()
            return True, 0
        return int(count) <= limit, max(1, int((index + 1) * window - now) + 1)


_limiter = None


def get_limiter():
    global _limiter
    if _limiter is None:
        _limiter = RedisLimiter(settings.redis_url) if settings.rate_limit_backend == "redis" else MemoryLimiter()
    return _limiter


def reset() -> None:
    """Forget all counters (tests)."""
    global _limiter
    _limiter = None
