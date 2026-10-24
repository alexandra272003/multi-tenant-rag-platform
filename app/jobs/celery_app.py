"""
Celery application for background ingestion.

Settings that matter, and why:
  * task_acks_late + reject_on_worker_lost: a task is acknowledged AFTER it finishes,
    so a worker that dies mid-job leaves the message to be redelivered (at-least-once).
    That is safe only because process_document() is idempotent.
  * worker_prefetch_multiplier=1: each worker takes one job at a time. Ingestion jobs
    are long and uneven; prefetching would let one worker hoard jobs while others idle.
  * no result backend: job state lives in the documents table, where the API reads it.
    A second place to keep status would be a second thing to disagree with the first.
  * a beat schedule runs the stuck-job sweeper (start the worker with `-B`, or run a
    separate `celery beat` process in production).
"""
from celery import Celery
from celery.signals import setup_logging

from app.core.config import settings

celery_app = Celery("rag", broker=settings.redis_url, include=["app.jobs.tasks"])

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": settings.ingest_lease_seconds * 2},
    timezone="UTC",
    beat_schedule={
        "sweep-stuck-documents": {
            "task": "sweep_stuck_documents",
            "schedule": float(settings.ingest_sweep_interval_seconds),
        }
    },
)


@setup_logging.connect
def _configure_worker_logging(**_kwargs):
    # Same JSON format as the API, so one log query covers both processes.
    from app.core.logging_setup import configure_logging
    configure_logging()
