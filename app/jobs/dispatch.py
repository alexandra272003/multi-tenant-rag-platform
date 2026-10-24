"""
The single place the API hands work to the queue. Kept separate so tests can
replace it (tests/conftest.py runs jobs inline or collects them), and so the router
never imports Celery directly.
"""
import asyncio


async def enqueue_ingestion(tenant_id: int, document_id: int) -> None:
    # Imported lazily: importing the task module builds the Celery app, which tests
    # and the API process do not need unless they really enqueue.
    from app.jobs.tasks import ingest_document_task

    # .delay() is a blocking network call to the broker; keep it off the event loop.
    await asyncio.to_thread(ingest_document_task.delay, tenant_id, document_id)
