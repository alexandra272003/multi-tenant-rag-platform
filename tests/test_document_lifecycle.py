"""Week 7, Day 3: versioning (optimistic locking), soft delete, audit fields."""
import asyncio

from sqlalchemy import select

from app.models import AuditLog, Chunk, Document
from tests.tenancy_helpers import add_user, upload, upload_ok


# ---- audit fields -----------------------------------------------------------------


async def test_new_document_has_owner_version_visibility_and_timestamps(client):
    me = (await client.get("/me")).json()
    r = await upload(client, "a.txt", "some document text")
    d = r.json()
    assert d["owner_id"] == me["id"] and d["version"] == 1 and d["visibility"] == "tenant"
    assert d["title"] is None and d["created_at"] and d["updated_at"]


async def test_update_bumps_version_and_updated_at_and_records_who(client, session):
    me = (await client.get("/me")).json()
    doc = await upload_ok(client, "a.txt", "some document text")
    before = (await client.get(f"/documents/{doc}")).json()
    await asyncio.sleep(0.01)
    r = await client.patch(f"/documents/{doc}", json={"title": "Q3 Report", "expected_version": 1})
    after = r.json()
    assert after["version"] == 2 and after["title"] == "Q3 Report"
    assert after["updated_at"] > before["updated_at"] and after["created_at"] == before["created_at"]
    row = (await session.execute(select(Document).where(Document.id == doc))).scalar_one()
    assert row.updated_by_id == me["id"]


# ---- optimistic locking -------------------------------------------------------------


async def test_stale_version_gets_409_with_the_current_version_and_changes_nothing(client):
    doc = await upload_ok(client, "a.txt", "some document text")
    assert (await client.patch(f"/documents/{doc}", json={"title": "first", "expected_version": 1})).status_code == 200
    stale = await client.patch(f"/documents/{doc}", json={"title": "second", "expected_version": 1})
    assert stale.status_code == 409
    err = stale.json()["error"]
    assert err["code"] == "version_conflict"
    assert err["details"] == {"current_version": 2, "expected_version": 1}
    current = (await client.get(f"/documents/{doc}")).json()
    assert current["title"] == "first" and current["version"] == 2  # the stale write did not land


async def test_retrying_with_the_fresh_version_succeeds(client):
    doc = await upload_ok(client, "a.txt", "some document text")
    await client.patch(f"/documents/{doc}", json={"title": "first", "expected_version": 1})
    r = await client.patch(f"/documents/{doc}", json={"title": "second", "expected_version": 2})
    assert r.status_code == 200 and r.json()["version"] == 3


async def test_two_simultaneous_edits_exactly_one_wins(client, make_client):
    """The lost-update race: two editors read version 1 and both write.
    One atomic UPDATE ... WHERE version = 1 means exactly one matches."""
    doc = await upload_ok(client, "a.txt", "some document text")
    admin = await add_user(client, make_client, "admin@t.test", "admin")
    responses = await asyncio.gather(
        client.patch(f"/documents/{doc}", json={"title": "from owner", "expected_version": 1}),
        admin.patch(f"/documents/{doc}", json={"title": "from admin", "expected_version": 1}),
    )
    assert sorted(r.status_code for r in responses) == [200, 409]
    final = (await client.get(f"/documents/{doc}")).json()
    winner = next(r for r in responses if r.status_code == 200).json()
    assert final["version"] == 2 and final["title"] == winner["title"]


async def test_many_concurrent_edits_never_lose_an_update(client):
    doc = await upload_ok(client, "a.txt", "some document text")
    results = await asyncio.gather(*[
        client.patch(f"/documents/{doc}", json={"title": f"t{i}", "expected_version": 1}) for i in range(8)
    ])
    assert [r.status_code for r in results].count(200) == 1
    assert [r.status_code for r in results].count(409) == 7
    assert (await client.get(f"/documents/{doc}")).json()["version"] == 2


async def test_patch_validation(client):
    doc = await upload_ok(client, "a.txt", "some document text")
    assert (await client.patch(f"/documents/{doc}", json={"expected_version": 1})).status_code == 422  # nothing to change
    assert (await client.patch(f"/documents/{doc}", json={"title": "x"})).status_code == 422  # no version
    assert (await client.patch(f"/documents/{doc}", json={"title": "x", "expected_version": 0})).status_code == 422
    assert (await client.patch(f"/documents/{doc}", json={"title": "x", "expected_version": 1, "owner_id": 5})).status_code == 422
    assert (await client.patch(f"/documents/{doc}", json={"title": "", "expected_version": 1})).status_code == 422
    assert (await client.patch(f"/documents/{doc}", json={"visibility": "public", "expected_version": 1})).status_code == 422


# ---- soft delete ----------------------------------------------------------------------


async def test_soft_delete_hides_the_document_everywhere(client):
    doc = await upload_ok(client, "gone.txt", "text about the gizmo project")
    assert (await client.post("/retrieve", json={"query": "gizmo"})).json()["results"]
    assert (await client.delete(f"/documents/{doc}")).status_code == 204
    assert (await client.get(f"/documents/{doc}")).status_code == 404
    assert (await client.get(f"/documents/{doc}/chunks")).status_code == 404
    assert (await client.get("/documents")).json() == []
    assert (await client.post("/retrieve", json={"query": "gizmo"})).json()["results"] == []
    assert (await client.delete(f"/documents/{doc}")).status_code == 404  # already gone
    assert (await client.patch(f"/documents/{doc}", json={"title": "x", "expected_version": 2})).status_code == 404


async def test_soft_delete_keeps_the_rows_for_audit_and_restore(client, session):
    me = (await client.get("/me")).json()
    doc = await upload_ok(client, "gone.txt", "text about the gizmo project")
    await client.delete(f"/documents/{doc}")
    row = (await session.execute(select(Document).where(Document.id == doc))).scalar_one()
    assert row.deleted_at is not None and row.deleted_by_id == me["id"] and row.version == 2
    chunks = (await session.execute(select(Chunk).where(Chunk.document_id == doc))).scalars().all()
    assert chunks  # still there, just unreachable through every read path


async def test_a_deleted_file_can_be_uploaded_again_but_a_live_duplicate_cannot(client):
    first = await upload_ok(client, "a.txt", "identical content for the dedupe check")
    assert (await upload(client, "a.txt", "identical content for the dedupe check")).status_code == 409
    await client.delete(f"/documents/{first}")
    again = await upload(client, "a.txt", "identical content for the dedupe check")
    assert again.status_code == 202 and again.json()["id"] != first  # partial unique index frees the slot


async def test_deleted_documents_do_not_reach_the_agent(client, scripted_agent):
    from tests.agent_helpers import text_turn, tool_turn
    doc = await upload_ok(client, "gone.txt", "text about the gizmo project")
    await client.delete(f"/documents/{doc}")
    model = scripted_agent([tool_turn("document_search", {"query": "gizmo"}), text_turn("ok")])
    await client.post("/agent", json={"query": "q"})
    assert model.calls[1][-1].content == "No matching passages found."


# ---- audit trail ------------------------------------------------------------------------


async def test_audit_records_who_did_what_in_order(client):
    me = (await client.get("/me")).json()
    doc = await upload_ok(client, "a.txt", "some document text")
    await client.patch(f"/documents/{doc}", json={"title": "T", "expected_version": 1})
    await client.delete(f"/documents/{doc}")
    log = (await client.get("/audit")).json()  # newest first
    actions = [e["action"] for e in log if e["resource_type"] == "document"]
    # Ingestion is two audited events now: the user's upload, then the worker's (a system
    # action, so it has no actor).
    assert actions == ["document.deleted", "document.updated", "document.ingested", "document.uploaded"]
    by_action = {e["action"]: e for e in log if e["resource_type"] == "document"}
    assert by_action["document.ingested"]["actor_user_id"] is None
    assert all(e["actor_user_id"] == me["id"] for a, e in by_action.items() if a != "document.ingested")
    updated = next(e for e in log if e["action"] == "document.updated")
    assert updated["details"] == {"title": "T", "from_version": 1} and updated["resource_id"] == doc


async def test_audit_covers_tenant_user_and_login_events(client, make_client):
    await add_user(client, make_client, "v@t.test", "viewer")
    actions = [e["action"] for e in (await client.get("/audit")).json()]
    assert {"tenant.created", "user.created", "auth.login"} <= set(actions)


async def test_user_changes_are_audited_with_details(client, make_client):
    v = await add_user(client, make_client, "v@t.test", "viewer")
    await client.patch(f"/users/{v.user_id}", json={"role": "editor"})
    entry = next(e for e in (await client.get("/audit")).json() if e["action"] == "user.updated")
    assert entry["resource_id"] == v.user_id and entry["details"] == {"role": "editor"}


async def test_audit_pagination_and_access(client, make_client):
    for i in range(5):
        await upload_ok(client, f"f{i}.txt", f"document number {i} text")
    page1 = (await client.get("/audit", params={"limit": 2})).json()
    page2 = (await client.get("/audit", params={"limit": 2, "offset": 2})).json()
    assert len(page1) == len(page2) == 2 and not {e["id"] for e in page1} & {e["id"] for e in page2}
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    assert (await viewer.get("/audit")).status_code == 403


async def test_failed_operations_leave_no_audit_row(client, session):
    """The audit row commits with the change, so a rejected change (duplicate
    upload) must not leave a phantom 'document.uploaded' behind."""
    await upload_ok(client, "a.txt", "identical content for the audit check")
    n_before = len((await session.execute(select(AuditLog).where(AuditLog.action == "document.uploaded"))).scalars().all())
    assert (await upload(client, "a.txt", "identical content for the audit check")).status_code == 409
    n_after = len((await session.execute(select(AuditLog).where(AuditLog.action == "document.uploaded"))).scalars().all())
    assert n_before == n_after == 1
