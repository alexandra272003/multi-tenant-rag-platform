from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import Permission, Principal, Role
from app.core.db import get_session
from app.core.logging_setup import bind
from app.core.errors import ForbiddenError, UnauthorizedError
from app.core.tenancy import set_tenant_context
from app.core.tokens import decode_access_token
from app.models import User

_bearer = HTTPBearer(auto_error=False)  # we raise our own 401 so the error shape matches the rest of the API


async def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> Principal:
    """
    Token -> Principal, with the DATABASE as the source of truth:
      * the user must still exist and be active (deactivation is immediate);
      * the role is read from the row, not the token (role changes are immediate);
      * the token's tenant must match the user's tenant.
    Then the session is bound to that tenant for application filters AND
    for Postgres row-level security (core/tenancy.py). Cost: one primary-key
    lookup per request -- the price of revocable, non-stale authorization.
    """
    if credentials is None:
        raise UnauthorizedError("Authentication required")
    claims = decode_access_token(credentials.credentials)
    try:
        user_id = int(claims["sub"])
    except (TypeError, ValueError):
        raise UnauthorizedError("Invalid or expired token") from None

    user = await session.get(User, user_id)
    if user is None or not user.is_active or user.tenant_id != claims.get("tid"):
        raise UnauthorizedError("Invalid or expired token")  # one message: don't reveal which check failed

    await set_tenant_context(session, user.tenant_id)
    bind(tenant_id=user.tenant_id, user_id=user.id)
    return Principal(user_id=user.id, tenant_id=user.tenant_id, role=Role(user.role), email=user.email)


def require(permission: Permission):
    """Dependency factory: `principal = Depends(require(Permission.X))`."""

    async def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.can(permission):
            raise ForbiddenError(
                "Your role does not permit this action", details={"required": permission.value}
            )
        return principal

    return _check
