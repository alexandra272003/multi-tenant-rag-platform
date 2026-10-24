import hmac

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import require
from app.auth.rbac import Permission, Principal
from app.core import metrics
from app.core.config import settings
from app.core.db import get_session
from app.core.errors import UnauthorizedError
from app.services import quota_service

router = APIRouter(tags=["ops"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request):
    """Prometheus scrape endpoint. Open by default (scrape from inside the network); set
    METRICS_TOKEN to require a bearer token."""
    if settings.metrics_token:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {settings.metrics_token}"):
            raise UnauthorizedError("Invalid metrics token")
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@router.get("/usage")
async def usage(
    principal: Principal = Depends(require(Permission.AUDIT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Today's usage and limits for YOUR tenant (admins and owners)."""
    return await quota_service.usage_report(session, principal.tenant_id)
