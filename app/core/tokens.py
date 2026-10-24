import time

import jwt

from app.core.config import settings
from app.core.errors import UnauthorizedError


def create_access_token(user_id: int, tenant_id: int, role: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "tid": tenant_id,
        "role": role,  # informational only: authorization re-reads the role from the DB
        "iat": now,
        "exp": now + settings.jwt_ttl_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        # Pin the algorithm list: never let the token header choose it (the
        # classic "alg: none" / RS-HS confusion attacks), and REQUIRE the
        # claims we depend on instead of treating a missing exp as "forever".
        return jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub", "tid"]},
        )
    except jwt.PyJWTError:
        raise UnauthorizedError("Invalid or expired token") from None
