from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.core.errors import DuplicateDocumentError, NotFoundError, VersionConflictError
from app.models import Chunk, Document, DocumentSource
from app.services.audit_service import add_audit


async def find_duplicate(
    session: AsyncSession, tenant_id: int, sha256: str, chunk_size: int, chunk_overlap: int
) -> Document | None:
    """Duplicate = same file + settings among THIS tenant's live documents.
    (Tenant-scoped: otherwise the 409 would tell tenant B that tenant A
    already uploaded this exact file.)"""
    result = await session.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.deleted_at.is_(None),
            Document.content_sha256 == sha256,
            Document.chunk_size == chunk_size,
            Document.chunk_overlap == chunk_overlap,
        )
    )
    return result.scalar_one_or_none()


async def create_document_with_chunks(
    session: AsyncSession, document: Document, chunks: list[Chunk], actor_user_id: int | None
) -> Document:
    """
    One commit for the document, ALL its chunks and its audit row: ingestion
    is atomic, so a failure halfway never leaves a document with only some
    of its chunks, or a document nobody recorded. The partial unique index is
    the backstop for two identical uploads racing past the pre-check.
    """
    document.chunks = chunks
    session.add(document)
    try:
        await session.flush()  # assigns document.id (and trips the unique index here)
        add_audit(
            session, tenant_id=document.tenant_id, actor_user_id=actor_user_id,
            action="document.created", resource_type="document", resource_id=document.id,
            details={"filename": document.filename, "chunks": document.chunk_count,
                     "visibility": document.visibility},
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise DuplicateDocumentError("This file was already ingested with these chunk settings")
    await session.refresh(document)
    return document


async def create_pending_document(
    session: AsyncSession, document: Document, data: bytes, actor_user_id: int | None
) -> Document:
    """Accept an upload: the document row (status 'uploaded', no chunks yet), the raw
    bytes the worker will need, and the audit row, in ONE commit. The partial unique
    index still arbitrates two identical uploads racing past the duplicate pre-check."""
    session.add(document)
    try:
        await session.flush()
        session.add(DocumentSource(document_id=document.id, tenant_id=document.tenant_id, data=data))
        add_audit(
            session, tenant_id=document.tenant_id, actor_user_id=actor_user_id,
            action="document.uploaded", resource_type="document", resource_id=document.id,
            details={"filename": document.filename, "bytes": len(data), "visibility": document.visibility},
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise DuplicateDocumentError("This file was already ingested with these chunk settings")
    await session.refresh(document)
    return document


async def list_documents(session: AsyncSession, access: AccessFilter) -> list[Document]:
    result = await session.execute(
        select(Document).where(access.document_clause()).order_by(Document.id.desc())
    )
    return list(result.scalars().all())


async def get_document(session: AsyncSession, access: AccessFilter, document_id: int) -> Document | None:
    """None for: missing, other tenant, soft-deleted, or private-and-not-yours.
    Callers turn all four into the same 404 so existence never leaks."""
    result = await session.execute(
        select(Document).where(Document.id == document_id, access.document_clause())
    )
    return result.scalar_one_or_none()


async def list_chunks(
    session: AsyncSession, access: AccessFilter, document_id: int, limit: int, offset: int
) -> list[Chunk]:
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.tenant_id == access.tenant_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def update_document(
    session: AsyncSession, access: AccessFilter, document_id: int,
    expected_version: int, changes: dict, actor_user_id: int | None,
) -> Document:
    """
    Optimistic locking: ONE atomic statement
        UPDATE documents SET ..., version = version + 1
        WHERE id = ? AND version = <what the client read> AND <still visible>
    If the row changed since the client read it, zero rows match and we
    answer 409 -- no lock is held while the user thinks, and a lost update
    is impossible because the check and the write are the same statement.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(
        update(Document)
        .where(
            Document.id == document_id,
            Document.version == expected_version,
            access.document_clause(),
        )
        .values(**changes, version=Document.version + 1, updated_at=now, updated_by_id=actor_user_id)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        await session.rollback()
        current = await get_document(session, access, document_id)
        if current is None:
            raise NotFoundError(f"Document {document_id} not found")
        raise VersionConflictError(
            "The document was modified by someone else; re-read it and retry",
            details={"current_version": current.version, "expected_version": expected_version},
        )
    add_audit(
        session, tenant_id=access.tenant_id, actor_user_id=actor_user_id,
        action="document.updated", resource_type="document", resource_id=document_id,
        details={**changes, "from_version": expected_version},
    )
    await session.commit()
    document = await get_document(session, access, document_id)
    await session.refresh(document)
    return document


async def soft_delete_document(
    session: AsyncSession, access: AccessFilter, document: Document, actor_user_id: int | None
) -> None:
    """Mark deleted instead of removing: the row and its chunks remain for
    audit/restore, but every read path filters `deleted_at IS NULL`, and the
    partial unique index frees the file to be uploaded again."""
    now = datetime.now(timezone.utc)
    await session.execute(
        update(Document)
        .where(Document.id == document.id, Document.tenant_id == access.tenant_id)
        .values(deleted_at=now, deleted_by_id=actor_user_id, updated_at=now,
                updated_by_id=actor_user_id, version=Document.version + 1)
        .execution_options(synchronize_session=False)
    )
    add_audit(session, tenant_id=access.tenant_id, actor_user_id=actor_user_id,
              action="document.deleted", resource_type="document", resource_id=document.id,
              details={"filename": document.filename})
    await session.commit()
