"""async ingestion: document status/attempts/lease columns, raw-upload staging table

Revision ID: e5c9a3b7d2f1
Revises: d4b8f2a1e6c3
Create Date: 2026-10-22

Expand-only, so it is safe to apply while the old code is still running:
  * the new documents columns all have server defaults, so every existing row
    becomes status='ready' (correct: it was ingested synchronously) and
    attempts=0 without a backfill pass;
  * document_sources is a new table;
  * it gets the same row-level-security treatment as the other tenant tables,
    and `rag_app` may SELECT / INSERT / UPDATE it (UPDATE, not DELETE: the
    worker clears the bytes by setting data = NULL).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5c9a3b7d2f1"
down_revision: Union[str, Sequence[str], None] = "d4b8f2a1e6c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::int"


def upgrade() -> None:
    op.add_column("documents", sa.Column("status", sa.String(12), nullable=False, server_default="ready"))
    op.add_column("documents", sa.Column("error_message", sa.String(500), nullable=True))
    op.add_column("documents", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("documents", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.create_check_constraint(
        "ck_documents_status", "documents", "status IN ('uploaded','processing','ready','failed')"
    )
    op.create_index(
        "ix_documents_inflight", "documents", ["status", "updated_at"],
        postgresql_where=sa.text("status IN ('uploaded','processing')"),
    )

    op.create_table(
        "document_sources",
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_document_sources_tenant_id", "document_sources", ["tenant_id"])

    op.execute("GRANT SELECT, INSERT, UPDATE ON document_sources TO rag_app")
    op.execute("ALTER TABLE document_sources ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE document_sources FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON document_sources "
        f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON document_sources")
    op.drop_index("ix_document_sources_tenant_id", table_name="document_sources")
    op.drop_table("document_sources")
    op.drop_index("ix_documents_inflight", table_name="documents")
    op.drop_constraint("ck_documents_status", "documents", type_="check")
    for col in ("started_at", "attempts", "error_message", "status"):
        op.drop_column("documents", col)
