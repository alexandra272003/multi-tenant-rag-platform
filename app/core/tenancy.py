"""
Database-level tenant isolation (Postgres row-level security).

Application code already filters by tenant (AccessFilter). This is the
SECOND, independent layer: even a query that forgot its WHERE clause --
or an SQL-injection-style bug -- can only see the current tenant's rows,
because the database itself enforces it.

How it works
  * Migration d4b8f2a1e6c3 creates a NOLOGIN role `rag_app` with
    least-privilege grants, and enables FORCE ROW LEVEL SECURITY with a
    policy on documents / chunks / audit_log:
        tenant_id = current_setting('app.tenant_id', true)::int
    An unset setting yields NULL -> the predicate is never true -> ZERO
    ROWS. Fail closed, not open.
  * At the start of EVERY transaction (SQLAlchemy `after_begin`), this
    module runs `SET LOCAL ROLE rag_app` and sets `app.tenant_id` from
    `session.info`. LOCAL means it lasts only for that transaction, so a
    pooled connection can never carry one tenant's context into another
    request. (A hook per transaction, not per request: every commit starts
    a new transaction, and the setting would otherwise silently vanish.)
  * Why SET ROLE and not a separate login: the app connects as the schema
    owner (superusers and owners bypass RLS). Dropping to `rag_app` inside
    the transaction makes the policies bind without a second connection
    string. Production would connect as a non-owner role directly.

SQLite (the unit-test database) has no RLS: the hook is a no-op there, and
the application-layer tests carry the isolation proof. Postgres behaviour
is verified by scripts/verify_postgres.py.
"""
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

APP_ROLE = "rag_app"


@event.listens_for(Session, "after_begin")
def _apply_tenant_context(session, transaction, connection) -> None:
    if connection.dialect.name != "postgresql":
        return
    if session.info.get("unrestricted"):
        return  # maintenance scripts only (eval, bootstrap): runs as the connection role, bypassing RLS
    connection.exec_driver_sql(f"SET LOCAL ROLE {APP_ROLE}")
    tenant_id = session.info.get("tenant_id")
    if tenant_id is not None:
        # int() guarantees what we interpolate is a number, never text.
        connection.exec_driver_sql(f"SELECT set_config('app.tenant_id', '{int(tenant_id)}', true)")


async def set_tenant_context(session: AsyncSession, tenant_id: int) -> None:
    """Bind this session to a tenant: for future transactions via the hook,
    and for the transaction that is ALREADY open (e.g. the one that just
    looked the user up before we knew their tenant)."""
    session.info["tenant_id"] = tenant_id
    if session.bind.dialect.name == "postgresql" and session.in_transaction():
        await session.execute(
            text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(int(tenant_id))}
        )
