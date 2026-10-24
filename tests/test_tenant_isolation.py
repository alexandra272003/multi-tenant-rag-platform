"""
Week 7, Day 1-2: "Tenant A cannot read Tenant B" -- at the API, in retrieval,
in Q&A, in the agent and its SQL tool, and for private documents inside a tenant.

Layout: `client` is the OWNER of tenant A (it holds the secret). `b` is the
OWNER of tenant B. Every test asks the same question from B's side: can I
see, infer, or touch anything of A's?
"""
import pytest

from tests.agent_helpers import text_turn, tool_turn
from tests.tenancy_helpers import add_user, upload, upload_ok

SECRET = "SECRET-TOKEN-XYZ"
A_TEXT = f"{SECRET} payroll: alice earns 900k. Project Atlas launches in March."


@pytest.fixture
async def a_doc(client):
    return await upload_ok(client, "a_payroll.txt", A_TEXT)


@pytest.fixture
async def b(make_client):
    return await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})


# ============================== document endpoints ====================================


async def test_b_sees_none_of_as_documents(client, b, a_doc):
    assert (await b.get("/documents")).json() == []
    assert len((await client.get("/documents")).json()) == 1


@pytest.mark.parametrize("method,path_fmt,kwargs", [
    ("get", "/documents/{id}", {}),
    ("get", "/documents/{id}/chunks", {}),
    ("patch", "/documents/{id}", {"json": {"title": "pwned", "expected_version": 1}}),
    ("delete", "/documents/{id}", {}),
])
async def test_b_gets_404_not_403_on_as_document(client, b, a_doc, method, path_fmt, kwargs):
    r = await getattr(b, method)(path_fmt.format(id=a_doc), **kwargs)
    assert r.status_code == 404  # 404, never 403: don't confirm the id exists
    # ...and the same response as for an id that doesn't exist at all
    ghost = await getattr(b, method)(path_fmt.format(id=999999), **kwargs)
    assert r.json()["error"]["code"] == ghost.json()["error"]["code"] == "not_found"


async def test_a_document_is_untouched_after_bs_attempts(client, b, a_doc):
    await b.patch(f"/documents/{a_doc}", json={"title": "pwned", "expected_version": 1})
    await b.delete(f"/documents/{a_doc}")
    doc = (await client.get(f"/documents/{a_doc}")).json()
    assert doc["title"] is None and doc["version"] == 1


async def test_duplicate_detection_is_per_tenant_and_leaks_nothing(client, b, a_doc):
    """Same bytes uploaded by B must succeed -- a global uniqueness rule
    would tell B that A already holds this exact file."""
    r = await upload(b, "copy.txt", A_TEXT)
    assert r.status_code == 202
    again = await upload(client, "a_payroll.txt", A_TEXT)  # A still can't double-upload
    assert again.status_code == 409
    assert str(r.json()["id"]) not in again.text  # A's 409 doesn't expose B's document either


async def test_users_are_isolated(client, b, make_client):
    await add_user(client, make_client, "alice@a.test", "editor")
    a_users = {u["email"] for u in (await client.get("/users")).json()}
    b_users = {u["email"] for u in (await b.get("/users")).json()}
    assert "alice@a.test" in a_users and a_users.isdisjoint(b_users)
    alice_id = next(u["id"] for u in (await client.get("/users")).json() if u["email"] == "alice@a.test")
    assert (await b.patch(f"/users/{alice_id}", json={"role": "viewer"})).status_code == 404
    assert (await b.patch(f"/users/{alice_id}", json={"is_active": False})).status_code == 404


async def test_audit_trails_are_isolated(client, b, a_doc):
    a_actions = [e["action"] for e in (await client.get("/audit")).json()]
    assert {"document.uploaded", "document.ingested"} <= set(a_actions)
    b_actions = {e["action"] for e in (await b.get("/audit")).json()}
    assert not b_actions & {"document.uploaded", "document.ingested"}


# ============================== retrieval and Q&A ====================================


async def test_retrieval_never_crosses_tenants(client, b, a_doc):
    mine = (await client.post("/retrieve", json={"query": "payroll"})).json()["results"]
    assert mine and SECRET in mine[0]["content"]
    theirs = await b.post("/retrieve", json={"query": "payroll alice salary"})
    assert theirs.json()["results"] == []
    # naming A's document explicitly doesn't help
    forced = await b.post("/retrieve", json={"query": "payroll", "document_id": a_doc})
    assert forced.json()["results"] == []


async def test_retrieval_does_not_leak_through_a_shared_corpus_when_b_has_its_own_docs(client, b, a_doc):
    await upload_ok(b, "b_notes.txt", "Tenant B lunch menu: soup and bread.")
    results = (await b.post("/retrieve", json={"query": "payroll", "top_k": 10})).json()["results"]
    assert {r["filename"] for r in results} == {"b_notes.txt"}
    assert SECRET not in str(results)


@pytest.mark.parametrize("path", ["/ask", "/ask/langchain", "/ask/structured"])
async def test_qa_endpoints_refuse_without_calling_the_llm_when_b_has_no_documents(
    client, b, a_doc, monkeypatch, path
):
    """B has no documents, so there is no evidence: deterministic refusal, and
    the model (which would be handed A's text) is never invoked."""
    async def must_not_run(*args, **kwargs):
        raise AssertionError("LLM was called although tenant B has no readable documents")

    from tests.conftest import FakeLangChainChain

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", must_not_run)
    monkeypatch.setattr("app.langchain_port.chain._chain", FakeLangChainChain(must_not_run))
    monkeypatch.setattr("app.langchain_port.structured._structured_chain", FakeLangChainChain(must_not_run))
    r = await b.post(path, json={"query": "what does alice earn?", "document_id": a_doc})
    assert r.status_code == 200
    body = r.json()
    assert SECRET not in r.text and body["retrieved_count"] == 0 and body["citations"] == []


async def test_llm_prompt_for_b_never_contains_as_text(client, b, a_doc, monkeypatch):
    await upload_ok(b, "b_notes.txt", "Tenant B lunch menu: soup and bread.")
    seen = []

    async def spy(messages):
        seen.append(str(messages))
        return "Soup [1]."

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", spy)
    await b.post("/ask", json={"query": "payroll alice salary", "top_k": 10})
    assert seen and SECRET not in seen[0] and "alice earns" not in seen[0]


# ============================== the agent and its SQL tool ============================


async def test_agent_in_b_cannot_search_as_document_even_by_id(client, b, a_doc, scripted_agent):
    model = scripted_agent([
        tool_turn("document_search", {"query": "payroll", "document_id": a_doc}, "c1"),
        tool_turn("document_search", {"query": "payroll alice"}, "c2"),
        text_turn("nothing"),
    ])
    r = await b.post("/agent", json={"query": "q", "document_ids": [a_doc]})  # even forging scope
    assert SECRET not in r.text
    assert r.json()["trace"][1]["event"] == "tool_denied"
    assert "No matching passages found." == model.calls[2][-1].content


async def test_agent_sql_tool_in_b_lists_only_bs_documents(client, b, a_doc, scripted_agent):
    await upload_ok(b, "b_notes.txt", "Tenant B lunch menu.")
    model = scripted_agent([tool_turn("query_documents", {"sql": "select filename from documents"}), text_turn("ok")])
    await b.post("/agent", json={"query": "q"})
    out = model.calls[1][-1].content
    assert "b_notes.txt" in out and "a_payroll.txt" not in out


async def test_forged_agent_scope_can_only_shrink_never_widen(client, b, a_doc, scripted_agent):
    own = await upload_ok(b, "b_notes.txt", "Tenant B lunch menu.")
    model = scripted_agent([tool_turn("document_search", {"query": "lunch menu"}), text_turn("ok")])
    await b.post("/agent", json={"query": "q", "document_ids": [own, a_doc]})
    out = model.calls[1][-1].content
    assert "b_notes.txt" in out and SECRET not in out


# ============================== private documents inside one tenant ====================


@pytest.fixture
async def team(client, make_client):
    return {
        "owner": client,
        "admin": await add_user(client, make_client, "admin@a.test", "admin"),
        "editor1": await add_user(client, make_client, "ed1@a.test", "editor"),
        "editor2": await add_user(client, make_client, "ed2@a.test", "editor"),
        "viewer": await add_user(client, make_client, "viewer@a.test", "viewer"),
    }


async def test_private_documents_are_visible_only_to_owner_and_admins(team):
    pid = await upload_ok(team["editor1"], "private.txt", "ed1 private memo about layoffs", visibility="private")
    shared = await upload_ok(team["editor1"], "shared.txt", "ed1 shared memo about lunch")
    expect = {"editor1": True, "admin": True, "owner": True, "editor2": False, "viewer": False}
    for who, can_see in expect.items():
        c = team[who]
        ids = {d["id"] for d in (await c.get("/documents")).json()}
        assert (pid in ids) is can_see, who
        assert shared in ids, who  # tenant-visible docs are visible to everyone
        assert (await c.get(f"/documents/{pid}")).status_code == (200 if can_see else 404), who
        assert (await c.get(f"/documents/{pid}/chunks")).status_code == (200 if can_see else 404), who


async def test_private_documents_are_excluded_from_retrieval(team):
    await upload_ok(team["editor1"], "private.txt", "layoffs planned for Q4", visibility="private")
    for who, expected in {"editor1": 1, "admin": 1, "editor2": 0, "viewer": 0}.items():
        r = await team[who].post("/retrieve", json={"query": "layoffs"})
        assert len(r.json()["results"]) == expected, who


async def test_private_documents_are_excluded_from_agent_scope(team, scripted_agent):
    await upload_ok(team["editor1"], "private.txt", "layoffs planned for Q4", visibility="private")
    model = scripted_agent([tool_turn("document_search", {"query": "layoffs"}), text_turn("ok")])
    await team["editor2"].post("/agent", json={"query": "q"})
    assert model.calls[1][-1].content == "No matching passages found."
    model2 = scripted_agent([tool_turn("document_search", {"query": "layoffs"}), text_turn("ok")])
    await team["admin"].post("/agent", json={"query": "q"})
    assert "layoffs planned" in model2.calls[1][-1].content


async def test_owner_can_make_a_document_private_after_upload_and_it_disappears_for_others(team):
    doc = await upload_ok(team["editor1"], "memo.txt", "quarterly memo text")
    assert (await team["viewer"].get(f"/documents/{doc}")).status_code == 200
    r = await team["editor1"].patch(f"/documents/{doc}", json={"visibility": "private", "expected_version": 1})
    assert r.status_code == 200 and r.json()["visibility"] == "private"
    assert (await team["viewer"].get(f"/documents/{doc}")).status_code == 404
    assert (await team["viewer"].post("/retrieve", json={"query": "quarterly"})).json()["results"] == []


async def test_invalid_visibility_is_rejected(team):
    r = await upload(team["editor1"], "x.txt", "some text", visibility="public")
    assert r.status_code == 422


# ============================== retrieval API shape: scoping is not optional ============


def test_retrieval_cannot_be_called_without_an_access_filter():
    """Fail-closed by construction: forgetting to scope is a TypeError,
    not a silent cross-tenant query."""
    import inspect
    from app.services import retrieval_service
    sig = inspect.signature(retrieval_service.retrieve)
    access = sig.parameters["access"]
    assert access.kind is inspect.Parameter.KEYWORD_ONLY and access.default is inspect.Parameter.empty


def test_langchain_retriever_requires_an_access_filter():
    from pydantic import ValidationError
    from app.langchain_port.retriever import PgVectorRetriever
    with pytest.raises(ValidationError):
        PgVectorRetriever(session=None, top_k=3)
