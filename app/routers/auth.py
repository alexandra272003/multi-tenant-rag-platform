from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_principal
from app.auth.rbac import Principal
from app.core.db import get_session
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserRead
from app.services import auth_service

router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, session: AsyncSession = Depends(get_session)):
    """Create a tenant and its first user (the OWNER). Returns a token so the
    caller is logged in immediately."""
    tenant, user, token = await auth_service.register_tenant(
        session, payload.tenant_name, payload.email, payload.password
    )
    return TokenResponse(access_token=token, tenant_id=tenant.id, tenant_slug=tenant.slug, role=user.role)


@router.post("/auth/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_session)):
    user, token = await auth_service.authenticate(session, payload.email, payload.password)
    return TokenResponse(access_token=token, tenant_id=user.tenant_id, role=user.role)


@router.get("/me", response_model=UserRead)
async def me(principal: Principal = Depends(get_principal), session: AsyncSession = Depends(get_session)):
    from app.models import User  # local: avoid a router-level import cycle with models in type hints
    return await session.get(User, principal.user_id)
