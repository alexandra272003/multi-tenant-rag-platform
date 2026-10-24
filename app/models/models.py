from datetime import date, datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, CheckConstraint, Date, DateTime, ForeignKey, Index, LargeBinary, String, Text, UniqueConstraint, text, true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Tenant(Base):
    """An isolation boundary: one customer organisation. Everything a
    tenant owns (users, documents, chunks, audit rows) carries tenant_id."""

    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(60), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Per-tenant overrides of the default daily quotas (NULL = use the platform default).
    daily_query_limit: Mapped[int | None] = mapped_column(nullable=True)
    daily_upload_limit: Mapped[int | None] = mapped_column(nullable=True)


class User(Base):
    """A person, belonging to exactly one tenant, with one role there.
    (A user in several tenants would need a memberships table -- a
    deliberate simplification, noted in the README.)"""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('owner','admin','editor','viewer')", name="ck_users_role"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)  # stored lowercased
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10))
    # Deactivation, not deletion: audit rows and document ownership keep
    # pointing at a real user. Checked on EVERY request, so it takes effect
    # immediately even for tokens that haven't expired.
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The same file may be ingested more than once with DIFFERENT chunk
        # settings (the Day 33 experiment); the exact same file + settings
        # twice is a duplicate -- but only WITHIN a tenant (a global rule
        # would tell tenant B that tenant A already uploaded this file) and
        # only among live rows (so a soft-deleted document can be re-uploaded).
        Index(
            "uq_doc_active_content", "tenant_id", "content_sha256", "chunk_size", "chunk_overlap",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        Index("ix_documents_tenant_id_id", "tenant_id", "id"),
        CheckConstraint("visibility IN ('tenant','private')", name="ck_documents_visibility"),
        CheckConstraint("version >= 1", name="ck_documents_version"),
        CheckConstraint(
            "status IN ('uploaded','processing','ready','failed')", name="ck_documents_status"
        ),
        # Serves the stuck-job sweeper (find uploaded/processing rows that stopped
        # moving) without indexing every finished document.
        Index(
            "ix_documents_inflight", "status", "updated_at",
            postgresql_where=text("status IN ('uploaded','processing')"),
            sqlite_where=text("status IN ('uploaded','processing')"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    # Creator. Nullable for documents that predate tenancy (backfilled into
    # the default tenant with no owner) and for system/offline ingestion.
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(10))  # pdf | txt | md
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int]
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    chunk_count: Mapped[int]
    # 'tenant' = every member can read; 'private' = owner, admins and owners.
    visibility: Mapped[str] = mapped_column(String(10), default="tenant", server_default="tenant")
    # Optimistic-lock counter: bumped on every change; updates must quote
    # the version they read (see document_repository.update_document).
    version: Mapped[int] = mapped_column(default=1, server_default="1")
    # Ingestion lifecycle (Week 7, Thu):
    #   uploaded -> processing -> ready          (happy path)
    #   uploaded -> processing -> uploaded ...   (transient failure, will be retried)
    #   processing -> failed                     (permanent failure, or retries exhausted)
    #   failed -> uploaded                       (POST /documents/{id}/retry)
    # Rows written by the legacy synchronous path are born 'ready', and so is every
    # row that existed before this column was added (server_default).
    status: Mapped[str] = mapped_column(String(12), default="ready", server_default="ready")
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Incremented every time a worker claims the document. Doubles as the
    # FENCING TOKEN: a worker may only finish the attempt it claimed, so a worker
    # that was presumed dead and comes back late cannot overwrite a newer attempt.
    attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    updated_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # Soft delete: the row (and its chunks) stay for audit/restore, but every
    # read path filters deleted_at IS NULL via AccessFilter.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class DocumentSource(Base):
    """The raw uploaded bytes, kept ONLY while the document is waiting to be (or
    being) processed -- the worker needs them, and a retry needs them. They are
    nulled when the document becomes ready. Stored in the database to keep this
    project to one stateful dependency; production would use object storage and
    keep just the object key here (see docs/WEEK7_INGESTION.md)."""

    __tablename__ = "document_sources"

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),
        Index("ix_chunks_tenant_document", "tenant_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Denormalised from the parent document on purpose: lets retrieval and
    # row-level security filter chunks by tenant WITHOUT joining documents,
    # and makes tenant isolation a property of the chunk row itself.
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int]  # position within the document, 0-based
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(Text)
    # Offsets are into the CLEANED text of this chunk's page (or of the whole
    # document for txt/md), so content == cleaned_page_text[char_start:char_end].
    char_start: Mapped[int]
    char_end: Mapped[int]
    token_count: Mapped[int]
    # Day 29: one embedding per chunk, generated locally at ingestion time.
    # Postgres: a real pgvector column (ivfflat cosine index). SQLite (test
    # suite only): a JSON array fallback -- same data, different storage.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions).with_variant(JSON(), "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class AuditLog(Base):
    """Append-only record of who did what, in which tenant. Written in the
    SAME transaction as the change it describes, so an action and its audit
    row commit or roll back together."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(40))  # e.g. document.created
    resource_type: Mapped[str] = mapped_column(String(30))
    resource_id: Mapped[int | None] = mapped_column(nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    """A chat thread. PRIVATE to the user who started it: tenant isolation says
    which organisation, user_id says which person -- not even an admin may read
    someone else's conversation through the API."""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_tenant_user", "tenant_id", "user_id", "updated_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user','assistant')", name="ck_messages_role"),
        Index("ix_messages_conversation", "conversation_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    # Assistant messages: the verified citations, stored with the answer so a
    # reloaded conversation still shows its sources. Snapshot, not a live join: the
    # document may since have been deleted or made private.
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    standalone_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UsageCounter(Base):
    """One row per (tenant, UTC day, metric). Incremented atomically by
    quota_service.reserve(); old days are history and can be archived."""

    __tablename__ = "usage_counters"

    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    metric: Mapped[str] = mapped_column(String(20), primary_key=True)
    value: Mapped[int] = mapped_column(default=0)
