from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog


def add_audit(
    session: AsyncSession, *, tenant_id: int, actor_user_id: int | None, action: str,
    resource_type: str, resource_id: int | None = None, details: dict | None = None,
) -> AuditLog:
    """Stage an audit row in the CURRENT transaction (no commit here): the
    caller's commit persists the change and its audit record together, so
    there is never a change without a record or a record without a change."""
    row = AuditLog(
        tenant_id=tenant_id, actor_user_id=actor_user_id, action=action,
        resource_type=resource_type, resource_id=resource_id, details=details,
    )
    session.add(row)
    return row
