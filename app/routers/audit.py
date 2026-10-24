from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import require
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.models import AuditLog
from app.schemas import AuditRead

router = APIRouter(tags=["audit"])


@router.get("/audit", response_model=list[AuditRead])
async def list_audit(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require(Permission.AUDIT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """The tenant's own audit trail, newest first. Tenant-filtered in the
    query (and by row-level security under Postgres)."""
    result = await session.execute(
        select(AuditLog)
        .where(AuditLog.tenant_id == principal.tenant_id)
        .order_by(AuditLog.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())
