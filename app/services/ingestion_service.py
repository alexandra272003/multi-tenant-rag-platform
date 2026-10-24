import hashlib
import logging
import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.rbac import Principal

from app.core.config import settings
from app.core.metrics import INGEST_JOBS
from app.core.embeddings import embed_texts
from app.core.errors import (
    AppError,
    ConflictError,
    DuplicateDocumentError,
    InvalidChunkParamsError,
    InvalidDocumentError,
    NotFoundError,
    PayloadTooLargeError,
    TransientIngestionError,
)
from app.ingestion.chunking import chunk_text, count_tokens
from app.ingestion.cleaning import clean_text
from app.ingestion.parsers import extract_pages, source_type_for
from app.models import Chunk, Document
from app.repositories import document_repository as repo
from app.repositories import job_repository as jobs
from app.schemas import ChunkPreviewResponse, PreviewChunk
from app.services.audit_service import add_audit

logger = logging.getLogger("ingestion")


def _chunk_pages(pages, chunk_size: int, chunk_overlap: int):
    """
    pages -> [(page_number, Chunk)] with cleaning applied per page.

    Chunks never span pages: that keeps each chunk's page number exact (so a
    citation can say "page 4"), at the cost of not merging a sentence that
    runs across a page break. A deliberate trade-off.
    """
    out = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue  # blank page, or a scanned page with no text layer
        for chunk in chunk_text(cleaned, chunk_size, chunk_overlap):
            out.append((page.number, chunk))
    return out


def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_overlap >= chunk_size:
        raise InvalidChunkParamsError("chunk_overlap must be smaller than chunk_size")


async def ingest_document(
    session: AsyncSession, principal: Principal, filename: str, data: bytes,
    chunk_size: int, chunk_overlap: int, visibility: str = "tenant",
) -> Document:
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_bytes} byte limit"
        )
    if not data:
        raise InvalidDocumentError("File is empty")
    _validate_params(chunk_size, chunk_overlap)

    source_type = source_type_for(filename)  # rejects unsupported types early
    sha256 = hashlib.sha256(data).hexdigest()

    existing = await repo.find_duplicate(session, principal.tenant_id, sha256, chunk_size, chunk_overlap)
    if existing is not None:
        raise DuplicateDocumentError(
            "This file was already ingested with these chunk settings",
            details={"existing_document_id": existing.id},
        )

    pages = extract_pages(filename, data)
    paged_chunks = _chunk_pages(pages, chunk_size, chunk_overlap)
    if not paged_chunks:
        raise InvalidDocumentError(
            "No extractable text found (scanned PDFs need OCR, which is not supported)"
        )

    # Day 29: embed every chunk's text before persisting, so the row is
    # written once with its vector already attached rather than in a
    # second pass. embed_texts batches internally and raises EmbeddingError
    # (mapped to a 502) if the provider fails after retries -- nothing is
    # saved for a document whose embeddings couldn't be generated.
    vectors = await embed_texts([c.text for _, c in paged_chunks])

    document = Document(
        tenant_id=principal.tenant_id,
        owner_id=principal.user_id,
        updated_by_id=principal.user_id,
        visibility=visibility,
        filename=filename,
        source_type=source_type,
        content_sha256=sha256,
        page_count=len(pages),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(paged_chunks),
    )
    chunk_rows = [
        Chunk(
            tenant_id=principal.tenant_id,
            chunk_index=i,
            page_number=page_number,
            content=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            token_count=c.token_count,
            embedding=vectors[i],
        )
        for i, (page_number, c) in enumerate(paged_chunks)
    ]
    return await repo.create_document_with_chunks(session, document, chunk_rows, principal.user_id)


# =============================================================================
# Week 7 (Thu): asynchronous ingestion
#
#   API   accept_upload()      validate, store bytes + a row with status 'uploaded', enqueue
#   WORKER process_document()  claim -> parse/chunk/embed -> atomically write chunks + 'ready'
#
# The legacy all-in-one ingest_document() above is kept for offline scripts
# (eval/) that want a finished document in one call.
# =============================================================================

_PARSE_FAILURE = "Could not process this file ({kind})"


async def accept_upload(
    session: AsyncSession, principal: Principal, filename: str, data: bytes,
    chunk_size: int, chunk_overlap: int, visibility: str = "tenant",
) -> Document:
    """Everything that can be rejected cheaply and synchronously, then a durable
    record. The expensive work (parse, chunk, embed) happens in the worker."""
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_bytes} byte limit")
    if not data:
        raise InvalidDocumentError("File is empty")
    _validate_params(chunk_size, chunk_overlap)
    source_type = source_type_for(filename)  # 415 for unsupported types, immediately
    sha256 = hashlib.sha256(data).hexdigest()

    existing = await repo.find_duplicate(session, principal.tenant_id, sha256, chunk_size, chunk_overlap)
    if existing is not None:
        details = {"existing_document_id": existing.id, "status": existing.status}
        if existing.status == "failed":
            details["hint"] = f"POST /documents/{existing.id}/retry to process it again"
        raise DuplicateDocumentError(
            "This file was already uploaded with these chunk settings", details=details
        )

    document = Document(
        tenant_id=principal.tenant_id, owner_id=principal.user_id, updated_by_id=principal.user_id,
        visibility=visibility, filename=filename, source_type=source_type, content_sha256=sha256,
        status="uploaded", page_count=0, chunk_count=0,  # unknown until the worker has parsed it
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    return await repo.create_pending_document(session, document, data, principal.user_id)


def _lost(result: str, tenant_id: int, document_id: int, **extra) -> str:
    INGEST_JOBS.labels(result).inc()
    logger.info("ingest %s tenant=%s document=%s %s", result, tenant_id, document_id, extra or "")
    return result


async def process_document(
    session_factory, tenant_id: int, document_id: int, *, now: datetime | None = None
) -> str:
    """
    The worker's unit of work. Safe to run twice, concurrently or late -- the queue
    delivers at-least-once, so this function must be idempotent:

      * already ready        -> 'already_ready'  (a redelivered message: do nothing)
      * claimed by a live worker -> 'busy'
      * failed / deleted     -> 'failed_noop' / 'gone'
      * otherwise claim it, do the work, and commit chunks + 'ready' atomically ->
        'ready', or 'superseded' if this attempt lost the document meanwhile.

    A tenant-bound session is used for every statement, so on Postgres row-level
    security applies inside the worker exactly as it does in the API.

    Raises TransientIngestionError when a retry could help (the document is put
    back to 'uploaded'). Permanent problems mark the document 'failed' and return.
    """
    started = time.perf_counter()
    now = now or datetime.now(timezone.utc)

    # 1. claim, and persist the claim BEFORE the slow work so other workers see it
    async with session_factory() as session:
        session.info["tenant_id"] = tenant_id
        token = await jobs.claim(session, tenant_id, document_id, now, settings.ingest_lease_seconds)
        if token is None:
            await session.rollback()
            status = await jobs.current_status(session, tenant_id, document_id)
            return _lost(
                {"ready": "already_ready", "processing": "busy", "failed": "failed_noop"}.get(status, "gone"),
                tenant_id, document_id,
            )
        job = await jobs.load_job(session, tenant_id, document_id)
        await session.commit()

    filename, chunk_size, chunk_overlap, data = job
    if data is None:
        await _record_failure(session_factory, tenant_id, document_id, token, "The uploaded file is no longer available")
        return "failed"

    # 2. deterministic work: parsing and chunking give the same answer every time,
    #    so any error here is permanent and retrying would be pointless
    try:
        pages = extract_pages(filename, data)
        paged_chunks = _chunk_pages(pages, chunk_size, chunk_overlap)
        if not paged_chunks:
            raise InvalidDocumentError(
                "No extractable text found (scanned PDFs need OCR, which is not supported)"
            )
    except AppError as exc:
        await _record_failure(session_factory, tenant_id, document_id, token, exc.message)
        return "failed"
    except Exception as exc:  # corrupt PDF, encoding problem... never echo internals back
        await _record_failure(
            session_factory, tenant_id, document_id, token, _PARSE_FAILURE.format(kind=type(exc).__name__)
        )
        return "failed"

    # 3. external work: the embedding provider can fail for reasons that go away
    try:
        vectors = await embed_texts([c.text for _, c in paged_chunks])
    except Exception as exc:
        message = getattr(exc, "message", None) or f"{type(exc).__name__}"
        async with session_factory() as session:
            session.info["tenant_id"] = tenant_id
            await jobs.release(session, tenant_id, document_id, token, f"embedding failed: {message}")
            await session.commit()
        logger.warning("ingest transient-failure tenant=%s document=%s attempt=%s: %s",
                       tenant_id, document_id, token, message)
        raise TransientIngestionError(message) from exc

    # 4. publish: the fencing check, every chunk, the status change, dropping the
    #    staged bytes and the audit row all commit together or not at all
    async with session_factory() as session:
        session.info["tenant_id"] = tenant_id
        if not await jobs.finish(session, tenant_id, document_id, token, len(pages), len(paged_chunks)):
            await session.rollback()
            return _lost("superseded", tenant_id, document_id, attempt=token)
        session.add_all([
            Chunk(
                tenant_id=tenant_id, document_id=document_id, chunk_index=i, page_number=page_number,
                content=c.text, char_start=c.char_start, char_end=c.char_end,
                token_count=c.token_count, embedding=vectors[i],
            )
            for i, (page_number, c) in enumerate(paged_chunks)
        ])
        await jobs.clear_source(session, tenant_id, document_id)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        add_audit(
            session, tenant_id=tenant_id, actor_user_id=None, action="document.ingested",
            resource_type="document", resource_id=document_id,
            details={"chunks": len(paged_chunks), "pages": len(pages), "attempts": token, "ms": elapsed_ms},
        )
        await session.commit()
    return _lost("ready", tenant_id, document_id, chunks=len(paged_chunks), ms=elapsed_ms, attempt=token)


async def _record_failure(session_factory, tenant_id, document_id, token, message: str) -> None:
    async with session_factory() as session:
        session.info["tenant_id"] = tenant_id
        if await jobs.fail(session, tenant_id, document_id, message, token=token):
            add_audit(
                session, tenant_id=tenant_id, actor_user_id=None, action="document.ingest_failed",
                resource_type="document", resource_id=document_id, details={"error": message, "attempt": token},
            )
        await session.commit()
    logger.warning("ingest failed tenant=%s document=%s: %s", tenant_id, document_id, message)


async def mark_failed(session_factory, tenant_id: int, document_id: int, message: str) -> bool:
    """Retries exhausted: nobody holds the document, so no fencing token."""
    async with session_factory() as session:
        session.info["tenant_id"] = tenant_id
        changed = await jobs.fail(session, tenant_id, document_id, message)
        if changed:
            add_audit(
                session, tenant_id=tenant_id, actor_user_id=None, action="document.ingest_failed",
                resource_type="document", resource_id=document_id,
                details={"error": message, "reason": "retries exhausted"},
            )
        await session.commit()
    return changed


async def request_retry(session: AsyncSession, principal: Principal, access: AccessFilter, document: Document) -> None:
    """User-initiated retry of a FAILED document (e.g. after the embedding provider
    came back). Only failed documents qualify, and only while the staged upload
    still exists."""
    if document.status != "failed":
        raise ConflictError(
            "Only failed documents can be retried", details={"status": document.status}
        )
    if not await jobs.has_source(session, document.tenant_id, document.id):
        raise ConflictError("The original upload is no longer stored; upload the file again")
    if not await jobs.requeue(session, document.tenant_id, document.id):
        await session.rollback()
        raise ConflictError("The document changed state; reload it and try again")
    add_audit(
        session, tenant_id=document.tenant_id, actor_user_id=principal.user_id,
        action="document.retry_requested", resource_type="document", resource_id=document.id,
    )
    await session.commit()


async def sweep_stuck(session_factory, enqueue, *, now: datetime | None = None) -> int:
    """Re-enqueue documents whose job stopped moving. Covers the two ways a job is
    lost without any error: the enqueue itself failed (broker down at upload time),
    or a worker died mid-job. Re-enqueueing is always safe because process_document
    is idempotent. Cross-tenant by nature, so it runs on an unrestricted session."""
    now = now or datetime.now(timezone.utc)
    async with session_factory() as session:
        session.info["unrestricted"] = True
        stuck = await jobs.find_stuck(
            session, now, settings.ingest_lease_seconds, settings.ingest_stuck_uploaded_seconds
        )
    for tenant_id, document_id in stuck:
        await enqueue(tenant_id, document_id)
    if stuck:
        logger.warning("sweeper re-enqueued %d stuck document(s)", len(stuck))
    return len(stuck)


async def get_document_or_404(
    session: AsyncSession, access: AccessFilter, document_id: int
) -> Document:
    document = await repo.get_document(session, access, document_id)
    if document is None:
        raise NotFoundError(f"Document {document_id} not found")
    return document


def preview_chunks(text: str, chunk_size: int, chunk_overlap: int, clean: bool) -> ChunkPreviewResponse:
    """Chunk text without saving anything -- for experimenting with settings."""
    _validate_params(chunk_size, chunk_overlap)
    body = clean_text(text) if clean else text
    chunks = chunk_text(body, chunk_size, chunk_overlap)
    return ChunkPreviewResponse(
        cleaned_char_count=len(body),
        total_tokens=count_tokens(body),
        chunk_count=len(chunks),
        chunks=[
            PreviewChunk(
                chunk_index=i,
                content=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
            )
            for i, c in enumerate(chunks)
        ],
    )
