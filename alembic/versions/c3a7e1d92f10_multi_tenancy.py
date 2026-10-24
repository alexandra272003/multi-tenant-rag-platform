"""multi-tenancy: tenants, users, audit_log; tenant_id/owner/visibility/version/soft-delete on documents; tenant_id on chunks

Revision ID: c3a7e1d92f10
Revises: 8a1c2f9d4b3e
Create Date: 2026-10-14

Expand -> backfill -> contract, in one migration because the data is small:
  1. create the new tables;
  2. create tenant 1 ("default") to adopt every pre-existing row;
  3. add the new document/chunk columns as NULLABLE, backfill them, THEN
     tighten to NOT NULL (adding a NOT NULL column with no default to a
     populated table fails, and a default would hide forgotten backfills);
  4. replace the global unique constraint with a per-tenant, live-rows-only
     partial unique index;
  5. add the indexes the new query shapes need.

Reproducible: `alembic upgrade head` on an empty DB and on a populated one
both end in the same schema (verified by scripts/verify_postgres.py, which
also checks the models and the migrated schema do not drift apart).

Downgrade is supported but LOSSY by design: tenants, users and the audit log
are dropped. It also re-creates the old GLOBAL unique constraint, which fails
if two tenants hold identical files -- refusing is better than deleting data.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3a7e1d92f10"
down_revision: Union[str, Sequence[str], None] = "8a1c2f9d4b3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. new tables ---------------------------------------------------------
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("slug", sa.String(60), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("slug", name="tenants_slug_key"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="users_email_key"),
        sa.CheckConstraint("role IN ('owner','admin','editor','viewer')", name="ck_users_role"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("resource_type", sa.String(30), nullable=False),
        sa.Column("resource_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_audit_tenant_created", "audit_log", ["tenant_id", "created_at"])

    # 2. tenant 1 adopts everything that exists today --------------------------
    op.execute("INSERT INTO tenants (id, name, slug) VALUES (1, 'Default', 'default')")
    # The explicit id above doesn't advance the sequence: without this the
    # first self-service signup would try id 1 again and collide.
    op.execute("SELECT setval(pg_get_serial_sequence('tenants', 'id'), (SELECT MAX(id) FROM tenants))")

    # 3. documents: add nullable -> backfill -> tighten --------------------------
    op.add_column("documents", sa.Column("tenant_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("title", sa.String(255), nullable=True))
    op.add_column("documents", sa.Column("visibility", sa.String(10), nullable=False, server_default="tenant"))
    op.add_column("documents", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("documents", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("updated_by_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("deleted_by_id", sa.Integer(), nullable=True))
    op.execute("UPDATE documents SET tenant_id = 1, updated_at = created_at")
    op.alter_column("documents", "tenant_id", nullable=False)
    op.alter_column("documents", "updated_at", nullable=False)
    op.create_foreign_key("fk_documents_tenant", "documents", "tenants", ["tenant_id"], ["id"])
    op.create_foreign_key("fk_documents_owner", "documents", "users", ["owner_id"], ["id"])
    op.create_foreign_key("fk_documents_updated_by", "documents", "users", ["updated_by_id"], ["id"])
    op.create_foreign_key("fk_documents_deleted_by", "documents", "users", ["deleted_by_id"], ["id"])
    op.create_check_constraint("ck_documents_visibility", "documents", "visibility IN ('tenant','private')")
    op.create_check_constraint("ck_documents_version", "documents", "version >= 1")

    # 4. uniqueness becomes per-tenant and ignores soft-deleted rows --------------
    op.drop_constraint("uq_doc_content_params", "documents", type_="unique")
    op.create_index(
        "uq_doc_active_content", "documents",
        ["tenant_id", "content_sha256", "chunk_size", "chunk_overlap"],
        unique=True, postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("ix_documents_tenant_id_id", "documents", ["tenant_id", "id"])

    # chunks: denormalised tenant_id (filter + RLS without a join) ----------------
    op.add_column("chunks", sa.Column("tenant_id", sa.Integer(), nullable=True))
    op.execute("UPDATE chunks SET tenant_id = d.tenant_id FROM documents d WHERE chunks.document_id = d.id")
    op.alter_column("chunks", "tenant_id", nullable=False)
    op.create_foreign_key("fk_chunks_tenant", "chunks", "tenants", ["tenant_id"], ["id"])
    op.create_index("ix_chunks_tenant_document", "chunks", ["tenant_id", "document_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_tenant_document", table_name="chunks")
    op.drop_constraint("fk_chunks_tenant", "chunks", type_="foreignkey")
    op.drop_column("chunks", "tenant_id")

    op.drop_index("ix_documents_tenant_id_id", table_name="documents")
    op.drop_index("uq_doc_active_content", table_name="documents")
    # Fails (loudly) if two tenants hold identical files: see module docstring.
    op.create_unique_constraint(
        "uq_doc_content_params", "documents", ["content_sha256", "chunk_size", "chunk_overlap"]
    )
    op.drop_constraint("ck_documents_version", "documents", type_="check")
    op.drop_constraint("ck_documents_visibility", "documents", type_="check")
    for fk in ("fk_documents_deleted_by", "fk_documents_updated_by", "fk_documents_owner", "fk_documents_tenant"):
        op.drop_constraint(fk, "documents", type_="foreignkey")
    for col in ("deleted_by_id", "deleted_at", "updated_by_id", "updated_at", "version",
                "visibility", "title", "owner_id", "tenant_id"):
        op.drop_column("documents", col)

    op.drop_index("ix_audit_tenant_created", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_users_tenant_id", table_name="users")
    op.drop_table("users")
    op.drop_table("tenants")
