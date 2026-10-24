"""row-level security: tenant isolation enforced by the database itself

Revision ID: d4b8f2a1e6c3
Revises: c3a7e1d92f10
Create Date: 2026-10-14

Second, independent isolation layer (the first is the application's
AccessFilter). See app/core/tenancy.py for how the application opts in:
each transaction runs `SET LOCAL ROLE rag_app` and sets `app.tenant_id`.

  * `rag_app` is a NOLOGIN role holding only the privileges the app needs:
    - documents, chunks : SELECT, INSERT, UPDATE   (no DELETE: deletion is soft)
    - audit_log         : SELECT, INSERT           (append-only: no UPDATE/DELETE)
    - users, tenants    : SELECT, INSERT, UPDATE   (no RLS: needed to log in
                          BEFORE a tenant context exists; app-layer scoped)
  * FORCE ROW LEVEL SECURITY makes the policies apply to the table owner too.
  * The policy `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::int`
    is FAIL-CLOSED: with no tenant set the comparison is NULL, so no rows match.
    (NULLIF matters: after a transaction that used set_config(..., true) a
    pooled connection reports '' not NULL, and ''::int would raise.)
  * Superusers bypass RLS entirely -- which is exactly why the app drops to
    `rag_app` instead of relying on the connection role.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d4b8f2a1e6c3"
down_revision: Union[str, Sequence[str], None] = "c3a7e1d92f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TENANT_TABLES = ("documents", "chunks", "audit_log")
_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::int"


def upgrade() -> None:
    op.execute(
        "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_app') "
        "THEN CREATE ROLE rag_app NOLOGIN; END IF; END $$"
    )
    # Lets the connecting user SET ROLE rag_app (a no-op for superusers).
    op.execute("GRANT rag_app TO CURRENT_USER")
    op.execute("GRANT USAGE ON SCHEMA public TO rag_app")
    op.execute("GRANT SELECT, INSERT, UPDATE ON documents, chunks, users, tenants TO rag_app")
    op.execute("GRANT SELECT, INSERT ON audit_log TO rag_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_app")

    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
        )


def downgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM rag_app")
    op.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM rag_app")
    op.execute("REVOKE USAGE ON SCHEMA public FROM rag_app")
    op.execute("REVOKE rag_app FROM CURRENT_USER")
    op.execute("DROP ROLE IF EXISTS rag_app")
