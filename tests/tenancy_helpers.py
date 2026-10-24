"""Helpers for the Week 7 multi-tenant / RBAC tests."""
import time

import jwt

from app.core.config import settings


async def upload(client, name: str, body: str, visibility: str | None = None, **params):
    q = {"chunk_size": 500, "chunk_overlap": 0, **params}
    if visibility:
        q["visibility"] = visibility
    return await client.post("/documents", files={"file": (name, body.encode())}, params=q)


async def upload_ok(client, name: str, body: str, visibility: str | None = None) -> int:
    r = await upload(client, name, body, visibility)
    assert r.status_code == 202, r.text
    return r.json()["id"]


async def add_user(owner_client, make_client, email: str, role: str, password: str = "password-123"):
    """Create a user in the owner's tenant and return a client logged in as them."""
    r = await owner_client.post("/users", json={"email": email, "password": password, "role": role})
    assert r.status_code == 201, r.text
    client = await make_client(login=(email, password))
    client.user_id = r.json()["id"]
    return client


def mint_token(user_id: int, tenant_id: int, *, secret: str | None = None, algorithm: str = "HS256",
               exp_in: int = 3600, drop: tuple = (), **overrides) -> str:
    """Build a JWT directly (for forgery / expiry tests)."""
    now = int(time.time())
    payload = {"sub": str(user_id), "tid": tenant_id, "role": "owner", "iat": now, "exp": now + exp_in}
    payload.update(overrides)
    for key in drop:
        payload.pop(key, None)
    return jwt.encode(payload, secret or settings.jwt_secret, algorithm=algorithm)


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
