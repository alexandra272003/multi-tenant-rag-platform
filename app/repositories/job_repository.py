"""
Database operations for the ingestion job lifecycle.

Every transition is ONE conditional UPDATE, so the database -- not Python --
arbitrates between workers. This is the same idea as the optimistic lock on
document edits and the conditional UPDATE that prevents overselling: "check and
change" must be a single atomic step, or two workers can both pass the check.

    claim   uploaded -> processing                     (also: expired processing -> processing)
    finish  processing -> ready      only for the attempt that claimed it (fencing token)
    release processing -> uploaded   transient failure, a retry will claim it again
    fail    uploaded|processing -> failed
    requeue failed -> uploaded       user asked for a retry
"""
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentSource

_LIVE = Document.deleted_at.is_(None)


def _scoped(tenant_id: int, document_id: int):
    # tenant_id is in EVERY statement: the worker is a trust boundary too, and a
    # wrong (tenant, document) pair must match nothing instead of someone else's row.
    return (Document.id == document_id, Document.tenant_id == tenant_id)


async def claim(
    session: AsyncSession, tenant_id: int, document_id: int, now: datetime, lease_seconds: int
) -> int | None:
    """Take the document for processing. Returns the fencing token (the new
    `attempts` value) or None if it cannot be claimed: already ready, already
    failed, deleted, unknown, or being processed under a lease that has not
    expired yet. An EXPIRED lease may be taken over: its worker is presumed dead."""
    cutoff = now - timedelta(seconds=lease_seconds)
    stmt = (
        update(Document)
        .where(
            *_scoped(tenant_id, document_id), _LIVE,
            or_(
                Document.status == "uploaded",
                and_(Document.status == "processing", Document.started_at < cutoff),
            ),
        )
        .values(status="processing", attempts=Document.attempts + 1, started_at=now, error_message=None)
        .returning(Document.attempts)
        .execution_options(synchronize_session=False)
    )
    row = (await session.execute(stmt)).first()
    return row[0] if row else None


async def current_status(session: AsyncSession, tenant_id: int, document_id: int) -> str | None:
    result = await session.execute(
        select(Document.status).where(*_scoped(tenant_id, document_id), _LIVE)
    )
    return result.scalar_one_or_none()


async def load_job(session: AsyncSession, tenant_id: int, document_id: int):
    """(filename, chunk_size, chunk_overlap, raw bytes or None) for a claimed document."""
    result = await session.execute(
        select(Document.filename, Document.chunk_size, Document.chunk_overlap, DocumentSource.data)
        .join(DocumentSource, DocumentSource.document_id == Document.id, isouter=True)
        .where(*_scoped(tenant_id, document_id))
    )
    return result.first()


async def finish(
    session: AsyncSession, tenant_id: int, document_id: int, token: int, page_count: int, chunk_count: int
) -> bool:
    """processing -> ready, but ONLY for the attempt that holds `token` and only if
    the document was not deleted meanwhile. False means this worker lost the
    document (lease expired and someone else claimed it, or a user deleted it):
    the caller must discard its result."""
    stmt = (
        update(Document)
        .where(
            *_scoped(tenant_id, document_id), _LIVE,
            Document.status == "processing", Document.attempts == token,
        )
        .values(status="ready", page_count=page_count, chunk_count=chunk_count,
                error_message=None, started_at=None)
        .execution_options(synchronize_session=False)
    )
    return (await session.execute(stmt)).rowcount == 1


async def release(
    session: AsyncSession, tenant_id: int, document_id: int, token: int, message: str
) -> bool:
    """processing -> uploaded after a TRANSIENT failure, so the next attempt can claim it."""
    stmt = (
        update(Document)
        .where(*_scoped(tenant_id, document_id), Document.status == "processing", Document.attempts == token)
        .values(status="uploaded", error_message=message[:500], started_at=None)
        .execution_options(synchronize_session=False)
    )
    return (await session.execute(stmt)).rowcount == 1


async def fail(
    session: AsyncSession, tenant_id: int, document_id: int, message: str, token: int | None = None
) -> bool:
    """-> failed. With a token, only that attempt may fail it; without (retries
    exhausted, nobody holds the document) any in-flight row may."""
    conditions = [*_scoped(tenant_id, document_id), _LIVE]
    if token is not None:
        conditions += [Document.status == "processing", Document.attempts == token]
    else:
        conditions += [Document.status.in_(("uploaded", "processing"))]
    stmt = (
        update(Document).where(*conditions)
        .values(status="failed", error_message=message[:500], started_at=None)
        .execution_options(synchronize_session=False)
    )
    return (await session.execute(stmt)).rowcount == 1


async def requeue(session: AsyncSession, tenant_id: int, document_id: int) -> bool:
    """failed -> uploaded (the user pressed retry)."""
    stmt = (
        update(Document)
        .where(*_scoped(tenant_id, document_id), _LIVE, Document.status == "failed")
        .values(status="uploaded", error_message=None, started_at=None)
        .execution_options(synchronize_session=False)
    )
    return (await session.execute(stmt)).rowcount == 1


async def has_source(session: AsyncSession, tenant_id: int, document_id: int) -> bool:
    result = await session.execute(
        select(DocumentSource.data).where(
            DocumentSource.document_id == document_id, DocumentSource.tenant_id == tenant_id
        )
    )
    row = result.first()
    return row is not None and row[0] is not None


async def clear_source(session: AsyncSession, tenant_id: int, document_id: int) -> None:
    await session.execute(
        update(DocumentSource)
        .where(DocumentSource.document_id == document_id, DocumentSource.tenant_id == tenant_id)
        .values(data=None)
        .execution_options(synchronize_session=False)
    )


async def find_stuck(
    session: AsyncSession, now: datetime, lease_seconds: int, uploaded_seconds: int
) -> list[tuple[int, int]]:
    """(tenant_id, document_id) of in-flight documents that stopped moving:
    'processing' past its lease (worker died) and 'uploaded' for too long (the
    enqueue never happened, or the message was lost). Cross-tenant by nature, so
    the caller must use an unrestricted session."""
    result = await session.execute(
        select(Document.tenant_id, Document.id).where(
            _LIVE,
            or_(
                and_(Document.status == "processing", Document.started_at < now - timedelta(seconds=lease_seconds)),
                and_(Document.status == "uploaded", Document.updated_at < now - timedelta(seconds=uploaded_seconds)),
            ),
        ).order_by(Document.id)
    )
    return [(t, d) for t, d in result.all()]
