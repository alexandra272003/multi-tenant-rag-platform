from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import require
from app.auth.rbac import Permission, Principal, Role
from app.core.db import get_session
from app.schemas import UserCreate, UserRead, UserUpdate
from app.services import auth_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserRead])
async def list_users(
    principal: Principal = Depends(require(Permission.USER_READ)),
    session: AsyncSession = Depends(get_session),
):
    return await auth_service.list_users(session, principal)


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    principal: Principal = Depends(require(Permission.USER_MANAGE)),
    session: AsyncSession = Depends(get_session),
):
    return await auth_service.create_user(
        session, principal, payload.email, payload.password, Role(payload.role)
    )


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    principal: Principal = Depends(require(Permission.USER_MANAGE)),
    session: AsyncSession = Depends(get_session),
):
    return await auth_service.update_user(
        session, principal, user_id,
        Role(payload.role) if payload.role else None, payload.is_active,
    )
