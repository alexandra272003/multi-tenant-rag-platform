"""conversational RAG: conversations and messages

Revision ID: f6d1b4c8a2e3
Revises: e5c9a3b7d2f1
Create Date: 2026-10-23

Expand-only (two new tables). Both get row-level security like every tenant table.
rag_app may DELETE here (deleting a conversation is a hard delete, unlike documents),
and needs USAGE on the new tables' id sequences -- the sequence grant in the RLS
migration only covered the sequences that existed then.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6d1b4c8a2e3"
down_revision: Union[str, Sequence[str], None] = "e5c9a3b7d2f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::int"


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_conversations_tenant_user", "conversations", ["tenant_id", "user_id", "updated_at"])
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("standalone_query", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('user','assistant')", name="ck_messages_role"),
    )
    op.create_index("ix_messages_conversation", "messages", ["conversation_id", "id"])

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON conversations, messages TO rag_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_app")
    for table in ("conversations", "messages"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
        )


def downgrade() -> None:
    for table in ("messages", "conversations"):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
    op.drop_index("ix_messages_conversation", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_tenant_user", table_name="conversations")
    op.drop_table("conversations")
