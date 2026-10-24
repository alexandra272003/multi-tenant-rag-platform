"""Week 7, Day 2: authorization -- what may each role do?"""
import pytest

from app.auth.rbac import ROLE_PERMISSIONS, Permission, Principal, Role, can_assign_role
from tests.tenancy_helpers import add_user, upload, upload_ok


@pytest.fixture
async def team(client, make_client):
    """One tenant, four roles. `client` is the owner."""
    return {
        "owner": client,
        "admin": await add_user(client, make_client, "admin@t.test", "admin"),
        "editor": await add_user(client, make_client, "editor@t.test", "editor"),
        "viewer": await add_user(client, make_client, "viewer@t.test", "viewer"),
    }


# ---- the permission table (pure) -----------------------------------------------


def test_permission_table_matches_the_documented_matrix():
    P = Permission
    assert ROLE_PERMISSIONS[Role.VIEWER] == {P.DOCUMENT_READ, P.QUERY}
    assert ROLE_PERMISSIONS[Role.EDITOR] == {P.DOCUMENT_READ, P.QUERY, P.DOCUMENT_CREATE, P.DOCUMENT_WRITE_OWN}
    assert ROLE_PERMISSIONS[Role.ADMIN] == set(P)
    assert ROLE_PERMISSIONS[Role.OWNER] == set(P)
    # strict hierarchy: each role's permissions contain the one below
    assert ROLE_PERMISSIONS[Role.VIEWER] < ROLE_PERMISSIONS[Role.EDITOR] < ROLE_PERMISSIONS[Role.ADMIN]


@pytest.mark.parametrize("actor,target,new,allowed", [
    (Role.OWNER, None, Role.OWNER, True), (Role.OWNER, Role.ADMIN, Role.VIEWER, True),
    (Role.ADMIN, None, Role.EDITOR, True), (Role.ADMIN, None, Role.VIEWER, True),
    (Role.ADMIN, None, Role.ADMIN, False), (Role.ADMIN, None, Role.OWNER, False),
    (Role.ADMIN, Role.EDITOR, Role.VIEWER, True), (Role.ADMIN, Role.EDITOR, Role.ADMIN, False),
    (Role.ADMIN, Role.OWNER, Role.VIEWER, False), (Role.ADMIN, Role.ADMIN, Role.VIEWER, False),
    (Role.EDITOR, None, Role.VIEWER, False), (Role.VIEWER, None, Role.VIEWER, False),
])
def test_role_assignment_rules(actor, target, new, allowed):
    assert can_assign_role(actor, target, new) is allowed


# ---- role x endpoint matrix, against the real API --------------------------------

# (role, expected status) for each action
MATRIX = {
    "list_documents": {"viewer": 200, "editor": 200, "admin": 200, "owner": 200},
    "retrieve":       {"viewer": 200, "editor": 200, "admin": 200, "owner": 200},
    "upload":         {"viewer": 403, "editor": 202, "admin": 202, "owner": 202},
    "list_users":     {"viewer": 403, "editor": 403, "admin": 200, "owner": 200},
    "create_user":    {"viewer": 403, "editor": 403, "admin": 201, "owner": 201},
    "audit":          {"viewer": 403, "editor": 403, "admin": 200, "owner": 200},
}


@pytest.mark.parametrize("action", list(MATRIX))
@pytest.mark.parametrize("role", ["viewer", "editor", "admin", "owner"])
async def test_role_permission_matrix(team, action, role):
    c = team[role]
    calls = {
        "list_documents": lambda: c.get("/documents"),
        "retrieve": lambda: c.post("/retrieve", json={"query": "anything"}),
        "upload": lambda: upload(c, f"{role}.txt", f"text by {role}"),
        "list_users": lambda: c.get("/users"),
        "create_user": lambda: c.post("/users", json={"email": f"new-{role}@t.test", "password": "password-123", "role": "viewer"}),
        "audit": lambda: c.get("/audit"),
    }
    assert (await calls[action]()).status_code == MATRIX[action][role]


async def test_forbidden_response_names_the_missing_permission(team):
    r = await team["viewer"].get("/users")
    assert r.status_code == 403
    assert r.json()["error"] == {"code": "forbidden", "message": "Your role does not permit this action",
                                 "details": {"required": "user:read"}}


# ---- object-level checks: role alone is not enough -----------------------------------


async def test_editor_can_edit_and_delete_own_document(team):
    doc = await upload_ok(team["editor"], "mine.txt", "editor's document")
    ok = await team["editor"].patch(f"/documents/{doc}", json={"title": "Renamed", "expected_version": 1})
    assert ok.status_code == 200 and ok.json()["title"] == "Renamed"
    assert (await team["editor"].delete(f"/documents/{doc}")).status_code == 204


async def test_editor_cannot_edit_or_delete_someone_elses_document(team, make_client, client):
    owners_doc = await upload_ok(team["owner"], "owners.txt", "belongs to the owner")
    other_editor = await add_user(client, make_client, "editor2@t.test", "editor")
    for caller in (team["editor"], other_editor):
        assert (await caller.patch(f"/documents/{owners_doc}", json={"title": "x", "expected_version": 1})).status_code == 403
        assert (await caller.delete(f"/documents/{owners_doc}")).status_code == 403
    # ...and nothing changed
    assert (await team["owner"].get(f"/documents/{owners_doc}")).json()["version"] == 1


async def test_viewer_cannot_edit_or_delete_anything(team):
    doc = await upload_ok(team["owner"], "doc.txt", "some text")
    assert (await team["viewer"].patch(f"/documents/{doc}", json={"title": "x", "expected_version": 1})).status_code == 403
    assert (await team["viewer"].delete(f"/documents/{doc}")).status_code == 403


async def test_admin_can_edit_any_document_in_the_tenant(team):
    doc = await upload_ok(team["editor"], "e.txt", "editor's document")
    r = await team["admin"].patch(f"/documents/{doc}", json={"title": "Admin edit", "expected_version": 1})
    assert r.status_code == 200
    assert (await team["admin"].delete(f"/documents/{doc}")).status_code == 204


# ---- managing users ---------------------------------------------------------------------


async def test_admin_can_create_editors_and_viewers_but_not_admins_or_owners(team):
    admin = team["admin"]
    for role, status in [("editor", 201), ("viewer", 201), ("admin", 403), ("owner", 403)]:
        r = await admin.post("/users", json={"email": f"{role}-by-admin@t.test", "password": "password-123", "role": role})
        assert r.status_code == status, role


async def test_owner_can_create_any_role(team):
    for role in ("owner", "admin", "editor", "viewer"):
        r = await team["owner"].post("/users", json={"email": f"{role}2@t.test", "password": "password-123", "role": role})
        assert r.status_code == 201


async def test_admin_cannot_promote_themselves_or_others_to_admin(team, client):
    users = {u["email"]: u["id"] for u in (await client.get("/users")).json()}
    assert (await team["admin"].patch(f"/users/{users['editor@t.test']}", json={"role": "admin"})).status_code == 403
    assert (await team["admin"].patch(f"/users/{team['admin'].user_id}", json={"role": "owner"})).status_code == 403


async def test_admin_can_move_people_between_editor_and_viewer(team):
    r = await team["admin"].patch(f"/users/{team['editor'].user_id}", json={"role": "viewer"})
    assert r.status_code == 200 and r.json()["role"] == "viewer"


async def test_admin_cannot_touch_an_owner(team, client):
    owner_id = (await client.get("/me")).json()["id"]
    assert (await team["admin"].patch(f"/users/{owner_id}", json={"role": "viewer"})).status_code == 403
    assert (await team["admin"].patch(f"/users/{owner_id}", json={"is_active": False})).status_code == 403


async def test_nobody_can_change_their_own_role_or_status(team, client):
    owner_id = (await client.get("/me")).json()["id"]
    assert (await client.patch(f"/users/{owner_id}", json={"role": "viewer"})).status_code == 403
    assert (await client.patch(f"/users/{owner_id}", json={"is_active": False})).status_code == 403


async def test_owner_can_deactivate_and_reactivate(team, client):
    uid = team["editor"].user_id
    assert (await client.patch(f"/users/{uid}", json={"is_active": False})).json()["is_active"] is False
    assert (await client.patch(f"/users/{uid}", json={"is_active": True})).json()["is_active"] is True
    assert (await team["editor"].get("/documents")).status_code == 200


async def test_admin_can_deactivate_an_editor_but_not_another_admin(team, client, make_client):
    other_admin = await add_user(client, make_client, "admin2@t.test", "admin")
    assert (await team["admin"].patch(f"/users/{team['editor'].user_id}", json={"is_active": False})).status_code == 200
    assert (await team["admin"].patch(f"/users/{other_admin.user_id}", json={"is_active": False})).status_code == 403


async def test_user_update_requires_a_change_and_rejects_unknown_fields(team, client):
    uid = team["viewer"].user_id
    assert (await client.patch(f"/users/{uid}", json={})).status_code == 422
    assert (await client.patch(f"/users/{uid}", json={"tenant_id": 99})).status_code == 422  # extra="forbid"


async def test_a_tenant_cannot_lose_its_last_owner(session):
    """Defence in depth: the self-change rule already makes this unreachable
    through the API, so exercise the service rule directly."""
    from sqlalchemy import select
    from app.core.errors import ConflictError
    from app.models import User
    from app.services import auth_service

    await auth_service.register_tenant(session, "Solo", "solo@t.test", "password-123")
    owner = (await session.execute(select(User).where(User.email == "solo@t.test"))).scalar_one()
    system = Principal.system(owner.tenant_id)  # a non-user actor, so the self-change rule doesn't apply
    with pytest.raises(ConflictError, match="at least one active owner"):
        await auth_service.update_user(session, system, owner.id, Role.VIEWER, None)
    with pytest.raises(ConflictError, match="at least one active owner"):
        await auth_service.update_user(session, system, owner.id, None, False)
