"""
Admission control for expensive endpoints = authorization + rate limit + daily quota, in that order:
  1. RBAC       -- may this role do this at all?            (403, free)
  2. rate limit -- is this tenant bursting?                 (429 rate_limited, nearly free)
  3. quota      -- is the tenant's daily budget spent?      (429 quota_exceeded, one DB write)
Cheapest check first, so a flood is rejected before it costs a database round trip, and a rejected
request never consumes quota.
"""
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import require
from app.auth.rbac import Permission, Principal
from app.core import metrics, ratelimit
from app.core.config import settings
from app.core.db import get_session
from app.core.errors import RateLimitedError
from app.services import quota_service


def guard(permission: Permission, *, scope: str, metric: str):
    """scope: rate-limit bucket ('query'|'upload'); metric: quota counter ('queries'|'uploads')."""

    async def _admit(
        principal: Principal = Depends(require(permission)),
        session: AsyncSession = Depends(get_session),
    ) -> Principal:
        if settings.rate_limit_backend != "off":
            per_minute = settings.rate_limit_per_minute if scope == "query" else settings.upload_rate_limit_per_minute
            allowed, retry_after = await ratelimit.get_limiter().hit(f"t{principal.tenant_id}:{scope}", per_minute, 60)
            if not allowed:
                metrics.RATE_LIMITED.labels(scope).inc()
                raise RateLimitedError(
                    "Too many requests; slow down", retry_after, details={"scope": scope, "limit_per_minute": per_minute}
                )
        await quota_service.reserve(session, principal.tenant_id, metric)
        return principal

    return _admit


guard_query = guard(Permission.QUERY, scope="query", metric="queries")
guard_upload = guard(Permission.DOCUMENT_CREATE, scope="upload", metric="uploads")
