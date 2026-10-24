import httpx
from openai import APIConnectionError


async def test_ask_returns_answer_with_valid_citation(client, monkeypatch):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake(messages):
        return "Water boils at 100C at sea level [1]."

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", fake)

    resp = await client.post("/ask", json={"query": "boiling point of water"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_count"] == 1
    assert len(body["citations"]) == 1
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["filename"] == "notes.txt"
    assert body["all_citations_valid"] is True


async def test_ask_flags_fabricated_citation_numbers(client, monkeypatch):
    """
    Proves citations are VERIFIED, not trusted: the model is told not to
    cite a number outside the context, but models don't always obey
    instructions, so the app checks anyway.
    """
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake(messages):
        # Only 1 chunk was ever retrieved -- [99] does not exist.
        return "Here is a real claim [1] and a fabricated one [99]."

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", fake)

    resp = await client.post("/ask", json={"query": "boiling point of water"})
    body = resp.json()
    assert body["all_citations_valid"] is False
    # Only the real citation is resolved into the structured list.
    assert [c["number"] for c in body["citations"]] == [1]


async def test_ask_against_empty_corpus_skips_llm_call_and_refuses(client, monkeypatch):
    """
    Proves the refusal is deterministic: with nothing retrieved, the app
    never calls the LLM at all, rather than hoping the model says
    "I don't know" on its own.
    """
    called = False

    async def fake(messages):
        nonlocal called
        called = True
        return "unused"

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", fake)

    resp = await client.post("/ask", json={"query": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_count"] == 0
    assert body["answer"] == "I don't know based on the provided documents."
    assert body["citations"] == []
    assert called is False


async def test_ask_provider_failure_returns_502(client, monkeypatch):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Some content that will be retrieved.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def failing(messages):
        raise APIConnectionError(request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr("app.services.generation_service.get_chat_completion", failing)

    resp = await client.post("/ask", json={"query": "anything"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "provider_error"


async def test_ask_rejects_empty_query(client):
    resp = await client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


async def test_ask_respects_document_id_filter(client, monkeypatch):
    doc1 = (
        await client.post(
            "/documents",
            files={"file": ("doc1.txt", b"The sky is blue during a clear day.")},
            params={"chunk_size": 500, "chunk_overlap": 0},
        )
    ).json()
    doc2 = (
        await client.post(
            "/documents",
            files={"file": ("doc2.txt", b"The ocean is also blue most of the time.")},
            params={"chunk_size": 500, "chunk_overlap": 0},
        )
    ).json()

    resp = await client.post(
        "/ask", json={"query": "why is it blue", "document_id": doc2["id"]}
    )
    body = resp.json()
    assert body["retrieved_count"] >= 1
    assert all(c["document_id"] == doc2["id"] for c in body["citations"])
    assert doc1["id"] not in {c["document_id"] for c in body["citations"]}
