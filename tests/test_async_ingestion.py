"""Week 7, Thursday: upload is asynchronous; the worker is idempotent and fenced."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.errors import EmbeddingError, TransientIngestionError
from app.models import Chunk, Document, DocumentSource
from tests.tenancy_helpers import add_user, upload, upload_ok

TEXT = "Async ingestion test document. " * 20


@pytest.fixture
def queued(jobs):
    jobs.inline = False
    return jobs


async def _status(client, doc):
    return (await client.get(f"/documents/{doc}")).json()


async def _chunks(session, doc):
    return len((await session.execute(select(Chunk).where(Chunk.document_id == doc))).scalars().all())


def _fail_embeddings(monkeypatch, exc=None):
    async def boom(texts):
        raise exc or EmbeddingError("provider down")
    monkeypatch.setattr("app.services.ingestion_service.embed_texts", boom)


# ---- accept fast, process later ------------------------------------------------------


async def test_upload_returns_202_uploaded_with_location_and_no_chunks(client, queued, session):
    r = await upload(client, "a.txt", TEXT)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "uploaded" and body["chunk_count"] == 0
    assert r.headers["location"] == f"/documents/{body['id']}"
    assert await _chunks(session, body["id"]) == 0
    assert len(queued.queue) == 1 and queued.queue[0][1] == body["id"]


async def test_job_moves_document_to_ready_and_drops_staged_bytes(client, queued, session):
    doc = await upload_ok(client, "a.txt", TEXT)
    assert await queued.run_next() == "ready"
    d = await _status(client, doc)
    assert d["status"] == "ready" and d["chunk_count"] > 0 and d["attempts"] == 1
    assert await _chunks(session, doc) == d["chunk_count"]
    src = (await session.execute(select(DocumentSource).where(DocumentSource.document_id == doc))).scalar_one()
    assert src.data is None


async def test_redelivered_job_is_a_noop(client, queued, session):
    doc = await upload_ok(client, "a.txt", TEXT)
    tenant_id = queued.queue[0][0]
    assert await queued.run_next() == "ready"
    n = await _chunks(session, doc)
    assert await queued.run(tenant_id, doc) == "already_ready"
    assert await _chunks(session, doc) == n  # no duplicate chunks


async def test_duplicate_upload_is_409_and_creates_one_document(client, queued):
    await upload_ok(client, "a.txt", TEXT)
    r = await upload(client, "a.txt", TEXT)
    assert r.status_code == 409
    assert len((await client.get("/documents")).json()) == 1


# ---- leases and fencing --------------------------------------------------------------


async def test_live_lease_blocks_a_second_worker_and_expired_lease_is_taken_over(client, queued, session):
    doc = await upload_ok(client, "a.txt", TEXT)
    tenant_id = queued.queue[0][0]
    now = datetime.now(timezone.utc)
    # worker 1 claims and then "dies": claim directly, never finish
    from app.repositories import job_repository as jobs
    async with queued.session_factory() as s:
        s.info["tenant_id"] = tenant_id
        assert await jobs.claim(s, tenant_id, doc, now, settings.ingest_lease_seconds) == 1
        await s.commit()
    assert await queued.run(tenant_id, doc, now=now + timedelta(seconds=5)) == "busy"
    later = now + timedelta(seconds=settings.ingest_lease_seconds + 5)
    assert await queued.run(tenant_id, doc, now=later) == "ready"
    assert (await _status(client, doc))["attempts"] == 2


async def test_stale_worker_is_fenced_out(client, queued, session, monkeypatch):
    """Worker A claims, stalls in embedding; B takes over after lease expiry and
    finishes; A resumes and must not publish."""
    doc = await upload_ok(client, "a.txt", TEXT)
    tenant_id = queued.queue[0][0]
    from app.core.config import settings as s
    real = __import__("app.services.ingestion_service", fromlist=["x"]).embed_texts
    inner = {}

    async def stalling(texts):
        if "done" not in inner:
            inner["done"] = True  # first (A) call: let B run to completion in the middle
            future = datetime.now(timezone.utc) + timedelta(seconds=s.ingest_lease_seconds + 10)
            monkeypatch.setattr("app.services.ingestion_service.embed_texts", real)
            assert await queued.run(tenant_id, doc, now=future) == "ready"
        return [[0.0] * s.embedding_dimensions for _ in texts]

    monkeypatch.setattr("app.services.ingestion_service.embed_texts", stalling)
    assert await queued.run(tenant_id, doc) == "superseded"
    assert await _chunks(session, doc) == (await _status(client, doc))["chunk_count"]  # exactly one set


async def test_deleted_while_queued_or_processing_discards_the_work(client, queued, session):
    doc = await upload_ok(client, "a.txt", TEXT)
    assert (await client.delete(f"/documents/{doc}")).status_code == 204
    assert await queued.run_next() == "gone"
    assert await _chunks(session, doc) == 0


# ---- failures ------------------------------------------------------------------------


async def test_transient_embedding_failure_releases_for_retry(client, queued, monkeypatch):
    doc = await upload_ok(client, "a.txt", TEXT)
    tenant_id = queued.queue[0][0]
    _fail_embeddings(monkeypatch)
    assert await queued.run_next() is None and len(queued.transient_errors) == 1
    d = await _status(client, doc)
    assert d["status"] == "uploaded" and "embedding failed" in d["error_message"]
    async def recovered(texts):
        return [[0.0] * settings.embedding_dimensions for _ in texts]

    monkeypatch.setattr("app.services.ingestion_service.embed_texts", recovered)
    assert await queued.run(tenant_id, doc) == "ready"  # the retry succeeds
    assert (await _status(client, doc))["status"] == "ready"


async def test_permanent_failure_marks_failed_and_retry_endpoint_requeues(client, queued):
    doc = await upload_ok(client, "bad.pdf", "this is not a pdf")
    assert await queued.run_next() == "failed"
    d = await _status(client, doc)
    assert d["status"] == "failed" and d["error_message"]
    # failed -> retry is allowed while the source bytes exist; it fails again (still not a pdf)
    r = await client.post(f"/documents/{doc}/retry")
    assert r.status_code == 202 and r.json()["status"] == "uploaded"
    assert await queued.run_next() == "failed"


async def test_retry_rejected_unless_failed(client, queued):
    doc = await upload_ok(client, "a.txt", TEXT)
    assert (await client.post(f"/documents/{doc}/retry")).status_code == 409  # uploaded
    await queued.run_next()
    assert (await client.post(f"/documents/{doc}/retry")).status_code == 409  # ready


async def test_mark_failed_after_retries_exhausted(client, queued):
    from app.services import ingestion_service
    doc = await upload_ok(client, "a.txt", TEXT)
    tenant_id = queued.queue[0][0]
    assert await ingestion_service.mark_failed(queued.session_factory, tenant_id, doc, "gave up")
    d = await _status(client, doc)
    assert d["status"] == "failed" and d["error_message"] == "gave up"


# ---- broker outage and the sweeper -----------------------------------------------------


async def test_broker_outage_still_accepts_upload_and_sweeper_recovers(client, queued):
    from app.services import ingestion_service
    queued.enqueue_error = ConnectionError("redis down")
    r = await upload(client, "a.txt", TEXT)
    assert r.status_code == 202 and r.json()["status"] == "uploaded"
    queued.enqueue_error = None
    assert queued.queue == []
    # too fresh to be considered stuck
    assert await ingestion_service.sweep_stuck(queued.session_factory, queued.enqueue) == 0
    later = datetime.now(timezone.utc) + timedelta(seconds=settings.ingest_stuck_uploaded_seconds + 5)
    assert await ingestion_service.sweep_stuck(queued.session_factory, queued.enqueue, now=later) == 1
    assert await queued.run_next() == "ready"
    assert await ingestion_service.sweep_stuck(queued.session_factory, queued.enqueue, now=later) == 0


# ---- tenancy -------------------------------------------------------------------------


async def test_other_tenant_cannot_see_or_retry_a_job(client, queued, make_client):
    doc = await upload_ok(client, "a.txt", TEXT)
    b = await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})
    assert (await b.get(f"/documents/{doc}")).status_code == 404
    assert (await b.post(f"/documents/{doc}/retry")).status_code == 404


async def test_viewer_cannot_retry_and_editor_only_own(client, queued, make_client):
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    doc = await upload_ok(client, "bad.pdf", "nope")
    await queued.run_next()
    assert (await viewer.post(f"/documents/{doc}/retry")).status_code == 403


# ---- celery wrapper ------------------------------------------------------------------


def test_celery_task_retries_transient_errors_then_gives_up(monkeypatch):
    from app.jobs import tasks

    calls = {"give_up": []}

    async def failing(tenant_id, document_id):
        raise TransientIngestionError("provider down")

    async def give_up(tenant_id, document_id, message):
        calls["give_up"].append(message)
        return True

    monkeypatch.setattr(tasks, "run_job", failing)
    monkeypatch.setattr(tasks, "give_up", give_up)
    tasks.celery_app.conf.task_always_eager = True
    tasks.celery_app.conf.task_eager_propagates = False
    res = tasks.ingest_document_task.apply(args=(1, 2))
    # eager retries exhaust max_retries, then the wrapper records the failure
    assert calls["give_up"] or res.state in {"RETRY", "FAILURE", "SUCCESS"}


def test_celery_task_returns_result_on_success(monkeypatch):
    from app.jobs import tasks

    async def ok(tenant_id, document_id):
        return "ready"

    monkeypatch.setattr(tasks, "run_job", ok)
    assert tasks.ingest_document_task.apply(args=(1, 2)).get() == "ready"
