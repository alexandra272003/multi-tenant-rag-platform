"""per-tenant quotas: usage_counters + optional per-tenant limit overrides

Revision ID: a7e2c5d9b3f4
Revises: f6d1b4c8a2e3
Create Date: 2026-10-24

Expand-only. The two tenants columns are nullable (NULL = platform default), so existing
tenants keep working untouched. usage_counters has a composite primary key and no sequence.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7e2c5d9b3f4"
down_revision: Union[str, Sequence[str], None] = "f6d1b4c8a2e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::int"


def upgrade() -> None:
    op.add_column("tenants", sa.Column("daily_query_limit", sa.Integer(), nullable=True))
    op.add_column("tenants", sa.Column("daily_upload_limit", sa.Integer(), nullable=True))
    op.create_table(
        "usage_counters",
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id"), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("metric", sa.String(20), primary_key=True),
        sa.Column("value", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute("GRANT SELECT, INSERT, UPDATE ON usage_counters TO rag_app")
    op.execute("ALTER TABLE usage_counters ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE usage_counters FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON usage_counters USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON usage_counters")
    op.drop_table("usage_counters")
    op.drop_column("tenants", "daily_upload_limit")
    op.drop_column("tenants", "daily_query_limit")
