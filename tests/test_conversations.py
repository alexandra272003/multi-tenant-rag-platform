"""Week 7, Friday: tenant-scoped conversational RAG."""
import pytest
from sqlalchemy import select

from app.core.errors import ProviderError
from app.models import Message
from tests.tenancy_helpers import add_user, upload_ok

DOC = "The launch of Project Atlas was approved by Dana Whitfield in March."


@pytest.fixture
async def doc(client):
    return await upload_ok(client, "atlas.txt", DOC)


@pytest.fixture
def llm(monkeypatch):
    """Records every prompt; condense calls and answer calls are told apart by the system prompt."""
    calls = {"condense": [], "answer": []}

    async def fake(messages):
        if messages[0]["content"].startswith("Rewrite"):
            calls["condense"].append(messages)
            return "Who approved the launch of Project Atlas?"
        calls["answer"].append(messages)
        return "Dana Whitfield approved it [1]."

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", fake)
    return calls


async def _conv(c, title=None):
    r = await c.post("/conversations", json={"title": title} if title else {})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _say(c, conv, q, **kw):
    return await c.post(f"/conversations/{conv}/messages", json={"query": q, **kw})


# ---- the happy path ----------------------------------------------------------------------


async def test_first_turn_answers_with_verified_citations_and_persists_both_messages(client, doc, llm):
    conv = await _conv(client)
    r = await _say(client, conv, "Who approved Project Atlas?")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["role"] == "assistant" and "Dana" in body["answer"]["content"]
    assert body["answer"]["citations"][0]["document_id"] == doc
    assert body["all_citations_valid"] is True and body["retrieved_count"] >= 1
    assert llm["condense"] == []  # no history yet: nothing to condense

    detail = (await client.get(f"/conversations/{conv}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"] == "Who approved Project Atlas?"  # auto-titled from the first question


async def test_follow_up_is_condensed_and_history_reaches_the_model(client, doc, llm):
    conv = await _conv(client)
    await _say(client, conv, "Who approved Project Atlas?")
    r = await _say(client, conv, "and when?")
    assert r.json()["standalone_query"] == "Who approved the launch of Project Atlas?"
    assert len(llm["condense"]) == 1
    prompt = llm["answer"][-1]
    roles = [m["role"] for m in prompt]
    assert roles == ["system", "user", "assistant", "user"]
    assert "Source numbers refer ONLY" in prompt[0]["content"]
    assert "QUESTION: and when?" in prompt[-1]["content"]  # the user's own words, not the rewrite


async def test_citation_markers_are_stripped_from_replayed_history(client, doc, llm):
    conv = await _conv(client)
    await _say(client, conv, "Who approved Project Atlas?")
    await _say(client, conv, "and when?")
    replayed = llm["answer"][-1][2]["content"]
    assert "[1]" not in replayed and "Dana Whitfield approved it" in replayed


async def test_history_window_is_bounded(client, doc, llm, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.chat_history_messages", 2)
    conv = await _conv(client)
    for i in range(4):
        await _say(client, conv, f"question {i} about Atlas?")
    history_in_last_prompt = llm["answer"][-1][1:-1]
    assert len(history_in_last_prompt) == 2 and history_in_last_prompt[0]["role"] == "user"


async def test_condense_failure_falls_back_to_the_raw_query(client, doc, monkeypatch):
    async def flaky(messages):
        if messages[0]["content"].startswith("Rewrite"):
            raise ProviderError("down")
        return "Dana [1]."
    monkeypatch.setattr("app.services.generation_service.get_chat_completion", flaky)
    conv = await _conv(client)
    await _say(client, conv, "Who approved Project Atlas?")
    r = await _say(client, conv, "and when?")
    assert r.status_code == 200 and r.json()["standalone_query"] == "and when?"


@pytest.mark.parametrize("bad", ["", "line one\nline two", "x" * 600])
async def test_unusable_rewrites_are_ignored(client, doc, monkeypatch, bad):
    async def fake(messages):
        return bad if messages[0]["content"].startswith("Rewrite") else "ok [1]"
    monkeypatch.setattr("app.services.generation_service.get_chat_completion", fake)
    conv = await _conv(client)
    await _say(client, conv, "first question about Atlas")
    r = await _say(client, conv, "follow up")
    assert r.json()["standalone_query"] == "follow up"


async def test_provider_failure_saves_nothing(client, doc, monkeypatch, session):
    async def down(messages):
        raise ProviderError("provider down")
    monkeypatch.setattr("app.services.generation_service.complete", down)
    conv = await _conv(client)
    r = await _say(client, conv, "Who approved Project Atlas?")
    assert r.status_code >= 500 or r.status_code == 502
    assert (await client.get(f"/conversations/{conv}")).json()["messages"] == []


async def test_no_readable_documents_refuses_without_calling_the_llm(client, monkeypatch):
    async def must_not_run(messages):
        raise AssertionError("LLM called with no evidence")
    monkeypatch.setattr("app.services.generation_service.get_chat_completion", must_not_run)
    conv = await _conv(client)
    r = await _say(client, conv, "anything?")
    assert r.status_code == 200 and r.json()["retrieved_count"] == 0
    assert "don't know" in r.json()["answer"]["content"]


async def test_conversation_full_returns_409(client, doc, llm, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.chat_max_messages", 2)
    conv = await _conv(client)
    assert (await _say(client, conv, "Who approved Atlas?")).status_code == 200
    assert (await _say(client, conv, "again?")).status_code == 409


async def test_validation(client, doc):
    conv = await _conv(client)
    assert (await _say(client, conv, "")).status_code == 422
    assert (await _say(client, conv, "q", top_k=0)).status_code == 422
    assert (await client.post(f"/conversations/{conv}/messages", json={"query": "q", "tenant_id": 2})).status_code == 422


# ---- privacy and isolation --------------------------------------------------------------


async def test_conversations_are_private_even_from_admins_and_owners(client, make_client, doc, llm):
    editor = await add_user(client, make_client, "ed@t.test", "editor")
    conv = await _conv(editor)
    await _say(editor, conv, "Who approved Project Atlas?")
    for who in (client, await add_user(client, make_client, "admin@t.test", "admin")):  # owner and admin
        assert (await who.get(f"/conversations/{conv}")).status_code == 404
        assert (await _say(who, conv, "hi")).status_code == 404
        assert (await who.delete(f"/conversations/{conv}")).status_code == 404
        assert (await who.get("/conversations")).json() == []
    assert len((await editor.get("/conversations")).json()) == 1


async def test_other_tenant_gets_404_and_cannot_post(client, make_client, doc, llm):
    conv = await _conv(client)
    b = await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})
    assert (await b.get(f"/conversations/{conv}")).status_code == 404
    assert (await _say(b, conv, "q")).status_code == 404
    assert (await b.get("/conversations")).json() == []


async def test_viewers_may_chat(client, make_client, doc, llm):
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    conv = await _conv(viewer)
    assert (await _say(viewer, conv, "Who approved Project Atlas?")).status_code == 200


async def test_requires_authentication(make_client):
    anon = await make_client()
    assert (await anon.get("/conversations")).status_code in (401, 403)
    assert (await anon.post("/conversations", json={})).status_code in (401, 403)


async def test_retrieval_in_a_conversation_never_crosses_tenants(client, make_client, doc, llm):
    b = await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})
    conv = await _conv(b)
    r = await _say(b, conv, "Who approved Project Atlas?", document_id=doc)  # forging A's document id
    assert r.json()["retrieved_count"] == 0 and llm["answer"] == []


async def test_document_made_private_after_the_chat_stops_being_quotable(client, make_client, llm):
    """Access is evaluated per turn: an earlier answer does not keep the document reachable."""
    editor = await add_user(client, make_client, "ed@t.test", "editor")
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    doc = await upload_ok(editor, "memo.txt", DOC)
    conv = await _conv(viewer)
    assert (await _say(viewer, conv, "Who approved Project Atlas?")).json()["retrieved_count"] >= 1
    assert (await editor.patch(f"/documents/{doc}", json={"visibility": "private", "expected_version": 1})).status_code == 200
    llm["answer"].clear()
    later = await _say(viewer, conv, "Who approved Project Atlas again?")
    assert later.json()["retrieved_count"] == 0 and llm["answer"] == []


async def test_deleted_document_stops_being_quotable_but_old_citations_remain_readable(client, doc, llm):
    conv = await _conv(client)
    await _say(client, conv, "Who approved Project Atlas?")
    await client.delete(f"/documents/{doc}")
    assert (await _say(client, conv, "Who approved Project Atlas?")).json()["retrieved_count"] == 0
    first = (await client.get(f"/conversations/{conv}")).json()["messages"][1]
    assert first["citations"][0]["document_id"] == doc  # a snapshot, not a live join


# ---- delete / list ----------------------------------------------------------------------------


async def test_delete_removes_conversation_and_messages(client, doc, llm, session):
    conv = await _conv(client)
    await _say(client, conv, "Who approved Project Atlas?")
    assert (await client.delete(f"/conversations/{conv}")).status_code == 204
    assert (await client.get(f"/conversations/{conv}")).status_code == 404
    left = (await session.execute(select(Message).where(Message.conversation_id == conv))).scalars().all()
    assert left == []


async def test_list_is_newest_activity_first(client, doc, llm):
    a, b = await _conv(client, "A"), await _conv(client, "B")
    await _say(client, a, "Who approved Project Atlas?")
    assert [c["id"] for c in (await client.get("/conversations")).json()] == [a, b]
