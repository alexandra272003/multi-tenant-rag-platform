"""
Per-tenant DAILY quotas, enforced atomically.

The whole trick is one conditional UPDATE:

    UPDATE usage_counters SET value = value + 1
     WHERE tenant_id = :t AND day = :d AND metric = :m AND value < :limit
    RETURNING value

It either increments (admitted) or matches no row (over quota), and the database serialises
concurrent updates to the row, so 50 simultaneous requests at 49/50 admit exactly one. A
read-then-write ("SELECT value; if value < limit: UPDATE") would admit several.

Semantics worth stating: a request is counted when ADMITTED and not refunded if it later fails --
the provider call may already have cost money, and refunds invite retry-loop abuse.
The cost is a hot row per tenant per metric; at this scale that is fine, at far larger scale shard the
counter (value per shard, sum on read).
"""
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics
from app.core.config import settings
from app.core.errors import QuotaExceededError
from app.models import Tenant, UsageCounter

METRICS = ("queries", "uploads")


def _utcnow() -> datetime:  # a seam: tests move the clock
    return datetime.now(timezone.utc)


def seconds_until_reset(now: datetime) -> int:
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


async def limit_for(session: AsyncSession, tenant_id: int, metric: str) -> int:
    column = {"queries": Tenant.daily_query_limit, "uploads": Tenant.daily_upload_limit}[metric]
    override = (await session.execute(select(column).where(Tenant.id == tenant_id))).scalar_one_or_none()
    if override is not None:
        return override
    return {"queries": settings.daily_query_limit, "uploads": settings.daily_upload_limit}[metric]


async def used_today(session: AsyncSession, tenant_id: int, metric: str, day: date) -> int:
    value = (await session.execute(
        select(UsageCounter.value).where(
            UsageCounter.tenant_id == tenant_id, UsageCounter.day == day, UsageCounter.metric == metric
        )
    )).scalar_one_or_none()
    return value or 0


async def reserve(session: AsyncSession, tenant_id: int, metric: str) -> int:
    """Admit one unit or raise QuotaExceededError. Commits (the count must persist even if
    the request later fails). Returns the new usage."""
    now = _utcnow()
    today = now.date()
    limit = await limit_for(session, tenant_id, metric)

    insert = (postgresql if session.bind.dialect.name == "postgresql" else sqlite).insert
    await session.execute(
        insert(UsageCounter).values(tenant_id=tenant_id, day=today, metric=metric, value=0).on_conflict_do_nothing()
    )
    result = await session.execute(
        update(UsageCounter)
        .where(
            UsageCounter.tenant_id == tenant_id, UsageCounter.day == today,
            UsageCounter.metric == metric, UsageCounter.value < limit,
        )
        .values(value=UsageCounter.value + 1)
        .returning(UsageCounter.value)
    )
    new_value = result.scalar_one_or_none()
    await session.commit()
    if new_value is None:
        metrics.QUOTA_REJECTED.labels(metric).inc()
        raise QuotaExceededError(
            f"Daily {metric} quota exhausted", retry_after=seconds_until_reset(now),
            details={"metric": metric, "limit": limit, "used": await used_today(session, tenant_id, metric, today),
                     "resets_at": (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")},
        )
    return new_value


async def usage_report(session: AsyncSession, tenant_id: int) -> dict:
    now = _utcnow()
    report = {
        "day": now.date().isoformat(),
        "resets_at": (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
        "rate_limit_per_minute": {"query": settings.rate_limit_per_minute, "upload": settings.upload_rate_limit_per_minute},
    }
    for metric in METRICS:
        limit = await limit_for(session, tenant_id, metric)
        used = await used_today(session, tenant_id, metric, now.date())
        report[metric] = {"used": used, "limit": limit, "remaining": max(0, limit - used)}
    return report
