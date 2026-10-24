"""
Celery tasks. Deliberately thin: all the logic is in services/ingestion_service.py
(plain async functions, unit-tested without a broker). This file only
  * bridges Celery's synchronous world to asyncio,
  * decides retry vs give-up for transient failures.
"""
import asyncio
import logging

from celery.exceptions import MaxRetriesExceededError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# Importing registers the per-transaction `SET LOCAL ROLE rag_app` + tenant hook on
# every Session. Without this import the worker would connect as the table owner and
# silently bypass row-level security -- it would still "work", just without the
# database-level isolation layer. Do not remove.
from app.core import tenancy as _tenancy  # noqa: F401
from app.core.config import settings
from app.core.errors import TransientIngestionError
from app.jobs.celery_app import celery_app
from app.services import ingestion_service

logger = logging.getLogger("ingestion.task")


def _backoff_seconds(retries_so_far: int) -> float:
    return settings.ingest_retry_backoff_seconds * (2 ** retries_so_far)


async def _with_session_factory(fn):
    # A fresh engine per task run, with NullPool: each asyncio.run() creates a new
    # event loop, and a pooled asyncpg connection is bound to the loop that created
    # it ("attached to a different loop"). Connection cost is noise next to an
    # embedding call.
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        return await fn(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        await engine.dispose()


async def run_job(tenant_id: int, document_id: int) -> str:
    return await _with_session_factory(
        lambda sf: ingestion_service.process_document(sf, tenant_id, document_id)
    )


async def give_up(tenant_id: int, document_id: int, message: str) -> bool:
    return await _with_session_factory(
        lambda sf: ingestion_service.mark_failed(sf, tenant_id, document_id, message)
    )


@celery_app.task(bind=True, name="ingest_document", max_retries=settings.ingest_max_retries)
def ingest_document_task(self, tenant_id: int, document_id: int) -> str:
    try:
        return asyncio.run(run_job(tenant_id, document_id))
    except TransientIngestionError as exc:
        retries = self.request.retries
        if retries >= self.max_retries:
            asyncio.run(give_up(tenant_id, document_id, f"gave up after {retries + 1} attempts: {exc}"))
            return "failed"
        logger.warning("retry %d/%d in %.0fs: tenant=%s document=%s",
                       retries + 1, self.max_retries, _backoff_seconds(retries), tenant_id, document_id)
        try:
            raise self.retry(exc=exc, countdown=_backoff_seconds(retries))
        except MaxRetriesExceededError:  # defensive: covered by the check above
            asyncio.run(give_up(tenant_id, document_id, str(exc)))
            return "failed"


async def _sweep() -> int:
    from app.jobs.dispatch import enqueue_ingestion
    return await _with_session_factory(lambda sf: ingestion_service.sweep_stuck(sf, enqueue_ingestion))


@celery_app.task(name="sweep_stuck_documents")
def sweep_stuck_documents_task() -> int:
    return asyncio.run(_sweep())
