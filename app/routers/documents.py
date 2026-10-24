import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.deps import require
from app.auth.limits import guard_upload
from app.auth.rbac import Permission, Principal, can_write_document
from app.core.config import settings
from app.core.db import get_session
from app.core.errors import ForbiddenError, PayloadTooLargeError
from app.jobs import dispatch
from app.repositories import document_repository as repo
from app.schemas import ChunkRead, DocumentRead, DocumentUpdate
from app.services import ingestion_service

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger("documents")


async def _enqueue(tenant_id: int, document_id: int) -> None:
    """Hand the job to the queue. If the broker is down the document stays 'uploaded'
    -- durable in the database -- and the stuck-job sweeper enqueues it later. The
    upload itself must not fail just because the queue is unavailable."""
    try:
        await dispatch.enqueue_ingestion(tenant_id, document_id)
    except Exception:
        logger.exception("enqueue failed tenant=%s document=%s; the sweeper will recover it",
                         tenant_id, document_id)


@router.post("", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    chunk_size: int = Query(default=settings.default_chunk_size, ge=5, le=4000),
    chunk_overlap: int = Query(default=settings.default_chunk_overlap, ge=0),
    visibility: Literal["tenant", "private"] = Query(default="tenant"),
    principal: Principal = Depends(guard_upload),
    session: AsyncSession = Depends(get_session),
):
    # Read one byte past the limit: enough to detect "too big" without
    # pulling an arbitrarily large upload fully into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_bytes} byte limit")
    document = await ingestion_service.accept_upload(
        session, principal, file.filename or "upload", data, chunk_size, chunk_overlap, visibility
    )
    await _enqueue(document.tenant_id, document.id)
    # 202 Accepted: the work is not done. Poll GET /documents/{id} until status is
    # 'ready' (or 'failed', with error_message).
    response.headers["Location"] = f"/documents/{document.id}"
    return document


@router.post("/{document_id}/retry", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def retry_document(
    document_id: int,
    response: Response,
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Re-run ingestion for a FAILED document (same permission rule as editing it)."""
    access = AccessFilter.from_principal(principal)
    document = await ingestion_service.get_document_or_404(session, access, document_id)
    if not can_write_document(principal, document):
        raise ForbiddenError("You may only retry documents you created", details={"required": Permission.DOCUMENT_WRITE_ANY.value})
    await ingestion_service.request_retry(session, principal, access, document)
    await _enqueue(document.tenant_id, document.id)
    response.headers["Location"] = f"/documents/{document.id}"
    await session.refresh(document)
    return document


@router.get("", response_model=list[DocumentRead])
async def list_documents(
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    return await repo.list_documents(session, AccessFilter.from_principal(principal))


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: int,
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    return await ingestion_service.get_document_or_404(
        session, AccessFilter.from_principal(principal), document_id
    )


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def list_chunks(
    document_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    access = AccessFilter.from_principal(principal)
    await ingestion_service.get_document_or_404(session, access, document_id)
    return await repo.list_chunks(session, access, document_id, limit, offset)


@router.patch("/{document_id}", response_model=DocumentRead)
async def update_document(
    document_id: int,
    payload: DocumentUpdate,
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Edit title / visibility with optimistic locking: send the `version`
    you read as `expected_version`; a stale version gets 409 version_conflict."""
    access = AccessFilter.from_principal(principal)
    document = await ingestion_service.get_document_or_404(session, access, document_id)  # 404 first: no existence leak
    if not can_write_document(principal, document):
        raise ForbiddenError("You may only edit documents you created", details={"required": Permission.DOCUMENT_WRITE_ANY.value})
    changes = payload.model_dump(exclude_none=True, exclude={"expected_version"})
    return await repo.update_document(
        session, access, document_id, payload.expected_version, changes, principal.user_id
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    principal: Principal = Depends(require(Permission.DOCUMENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    access = AccessFilter.from_principal(principal)
    document = await ingestion_service.get_document_or_404(session, access, document_id)
    if not can_write_document(principal, document):
        raise ForbiddenError("You may only delete documents you created", details={"required": Permission.DOCUMENT_WRITE_ANY.value})
    await repo.soft_delete_document(session, access, document, principal.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
