"""Week 7, Day 2: authentication -- who is calling?"""
import base64
import json

import pytest

from app.core.passwords import hash_password, verify_password
from tests.tenancy_helpers import add_user, auth, mint_token

ME = {"tenant_name": "Acme Inc", "email": "boss@acme.test", "password": "password-123"}


# ---- registration ---------------------------------------------------------------


async def test_register_creates_tenant_and_owner(make_client):
    anon = await make_client()
    r = await anon.post("/auth/register", json=ME)
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "owner" and body["token_type"] == "bearer"
    assert body["tenant_slug"] == "acme-inc" and body["access_token"]
    me = await anon.get("/me", headers=auth(body["access_token"]))
    assert me.json()["email"] == "boss@acme.test" and me.json()["tenant_id"] == body["tenant_id"]


async def test_two_tenants_with_the_same_name_get_distinct_slugs(make_client):
    anon = await make_client()
    a = (await anon.post("/auth/register", json=ME)).json()
    b = (await anon.post("/auth/register", json={**ME, "email": "other@acme.test"})).json()
    assert a["tenant_slug"] != b["tenant_slug"] and a["tenant_id"] != b["tenant_id"]


async def test_duplicate_email_is_409_case_insensitively(make_client):
    anon = await make_client()
    await anon.post("/auth/register", json=ME)
    r = await anon.post("/auth/register", json={**ME, "email": "BOSS@Acme.TEST"})
    assert r.status_code == 409 and r.json()["error"]["details"]["field"] == "email"


@pytest.mark.parametrize("patch", [{"password": "short"}, {"email": "not-an-email"}, {"tenant_name": "x"}])
async def test_register_validates_input(make_client, patch):
    anon = await make_client()
    r = await anon.post("/auth/register", json={**ME, **patch})
    assert r.status_code == 422


async def test_password_is_never_stored_in_plain_text(make_client, session):
    from sqlalchemy import select
    from app.models import User
    anon = await make_client()
    await anon.post("/auth/register", json=ME)
    stored = (await session.execute(select(User.password_hash))).scalars().first()
    assert "password-123" not in stored and stored.startswith("scrypt$")


def test_password_hashing_salts_and_verifies():
    h1, h2 = hash_password("same-password"), hash_password("same-password")
    assert h1 != h2  # per-hash random salt
    assert verify_password("same-password", h1) and not verify_password("wrong", h1)
    assert not verify_password("x", "garbage") and not verify_password("x", "scrypt$1$2$3$a$b")


# ---- login ------------------------------------------------------------------------


async def test_login_succeeds_and_token_works(make_client):
    anon = await make_client()
    await anon.post("/auth/register", json=ME)
    r = await anon.post("/auth/login", json={"email": "Boss@acme.test", "password": "password-123"})
    assert r.status_code == 200
    assert (await anon.get("/me", headers=auth(r.json()["access_token"]))).status_code == 200


async def test_wrong_password_and_unknown_email_are_indistinguishable(make_client):
    anon = await make_client()
    await anon.post("/auth/register", json=ME)
    wrong = await anon.post("/auth/login", json={"email": ME["email"], "password": "nope-nope"})
    unknown = await anon.post("/auth/login", json={"email": "ghost@acme.test", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # no user enumeration


async def test_deactivated_user_cannot_log_in(client, make_client):
    viewer = await add_user(client, make_client, "v@x.test", "viewer")
    await client.patch(f"/users/{viewer.user_id}", json={"is_active": False})
    anon = await make_client()
    r = await anon.post("/auth/login", json={"email": "v@x.test", "password": "password-123"})
    assert r.status_code == 401


# ---- token validation ---------------------------------------------------------------


@pytest.mark.parametrize("path", ["/documents", "/users", "/audit", "/me"])
async def test_protected_endpoints_require_a_token(make_client, path):
    anon = await make_client()
    r = await anon.get(path)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("method,path", [("post", "/retrieve"), ("post", "/ask"), ("post", "/ask/langchain"),
                                         ("post", "/ask/structured"), ("post", "/agent"), ("post", "/agent/stream")])
async def test_every_query_endpoint_requires_a_token(make_client, method, path):
    anon = await make_client()
    r = await getattr(anon, method)(path, json={"query": "hello"})
    assert r.status_code == 401


async def test_public_endpoints_stay_public(make_client):
    anon = await make_client()
    assert (await anon.get("/ping")).status_code == 200
    r = await anon.post("/chunk-preview", json={"text": "hello world " * 5, "chunk_size": 20, "chunk_overlap": 0})
    assert r.status_code == 200


async def test_garbage_and_wrong_scheme_tokens_are_rejected(make_client):
    anon = await make_client()
    assert (await anon.get("/me", headers={"Authorization": "Bearer not.a.jwt"})).status_code == 401
    assert (await anon.get("/me", headers={"Authorization": "Basic Zm9vOmJhcg=="})).status_code == 401


async def test_expired_token_is_rejected(client, make_client):
    me = (await client.get("/me")).json()
    anon = await make_client()
    expired = mint_token(me["id"], me["tenant_id"], exp_in=-10)
    assert (await anon.get("/me", headers=auth(expired))).status_code == 401


async def test_token_signed_with_another_secret_is_rejected(client, make_client):
    me = (await client.get("/me")).json()
    anon = await make_client()
    forged = mint_token(me["id"], me["tenant_id"], secret="attacker-secret-attacker-secret-1234")
    assert (await anon.get("/me", headers=auth(forged))).status_code == 401


async def test_alg_none_token_is_rejected(client, make_client):
    """The classic bypass: a token whose header says "alg": "none" and has no signature."""
    me = (await client.get("/me")).json()
    b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    token = f'{b64({"alg": "none", "typ": "JWT"})}.{b64({"sub": str(me["id"]), "tid": me["tenant_id"], "exp": 9999999999})}.'
    anon = await make_client()
    assert (await anon.get("/me", headers=auth(token))).status_code == 401


@pytest.mark.parametrize("drop", [("exp",), ("tid",), ("sub",)])
async def test_token_missing_required_claims_is_rejected(client, make_client, drop):
    me = (await client.get("/me")).json()
    anon = await make_client()
    token = mint_token(me["id"], me["tenant_id"], drop=drop)
    assert (await anon.get("/me", headers=auth(token))).status_code == 401


async def test_token_naming_the_wrong_tenant_is_rejected(client, make_client):
    """Even a correctly SIGNED token can't claim a tenant the user isn't in."""
    me = (await client.get("/me")).json()
    anon = await make_client()
    wrong_tenant = mint_token(me["id"], me["tenant_id"] + 999)
    assert (await anon.get("/me", headers=auth(wrong_tenant))).status_code == 401


async def test_token_for_a_nonexistent_user_is_rejected(make_client):
    anon = await make_client()
    assert (await anon.get("/me", headers=auth(mint_token(424242, 1)))).status_code == 401


# ---- revocation: the database is the source of truth ----------------------------------


async def test_deactivation_takes_effect_on_the_next_request_even_with_a_valid_token(client, make_client):
    editor = await add_user(client, make_client, "e@x.test", "editor")
    assert (await editor.get("/documents")).status_code == 200
    await client.patch(f"/users/{editor.user_id}", json={"is_active": False})
    assert (await editor.get("/documents")).status_code == 401  # token still unexpired


async def test_role_change_takes_effect_immediately_despite_the_role_claim_in_the_token(client, make_client):
    admin = await add_user(client, make_client, "a@x.test", "admin")
    ok = await admin.post("/users", json={"email": "n1@x.test", "password": "password-123", "role": "viewer"})
    assert ok.status_code == 201
    await client.patch(f"/users/{admin.user_id}", json={"role": "viewer"})  # demote; token still says admin
    denied = await admin.post("/users", json={"email": "n2@x.test", "password": "password-123", "role": "viewer"})
    assert denied.status_code == 403


# ---- races ------------------------------------------------------------------------


async def test_simultaneous_signups_with_one_email_yield_one_201_and_no_500s(make_client):
    """Found by running against real Postgres: the unique-violation fired at
    flush(), outside the handler, so a signup race became a 500."""
    import asyncio
    clients = [await make_client() for _ in range(4)]
    results = await asyncio.gather(*[
        c.post("/auth/register", json={**ME, "tenant_name": f"Race {i}"}) for i, c in enumerate(clients)
    ])
    assert sorted(r.status_code for r in results) == [201, 409, 409, 409]


async def test_simultaneous_signups_with_the_same_tenant_name_all_succeed_with_distinct_slugs(make_client):
    import asyncio
    clients = [await make_client() for _ in range(4)]
    results = await asyncio.gather(*[
        c.post("/auth/register", json={**ME, "email": f"user{i}@race.test"}) for i, c in enumerate(clients)
    ])
    assert [r.status_code for r in results] == [201] * 4
    assert len({r.json()["tenant_slug"] for r in results}) == 4


async def test_simultaneous_user_creation_with_one_email_yields_one_201(client):
    import asyncio
    results = await asyncio.gather(*[
        client.post("/users", json={"email": "dup@t.test", "password": "password-123", "role": "viewer"})
        for _ in range(4)
    ])
    assert sorted(r.status_code for r in results) == [201, 409, 409, 409]


# ---- configuration safety -----------------------------------------------------------


def test_default_jwt_secret_is_refused_outside_development():
    """A default signing key means anyone can mint tokens, so the app must not boot with it in production."""
    from pydantic import ValidationError
    from app.core.config import Settings

    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(environment="production")
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(environment="staging")
    assert Settings(environment="production", jwt_secret="x" * 48).environment == "production"
    assert Settings(environment="development").environment == "development"  # dev default stays usable
