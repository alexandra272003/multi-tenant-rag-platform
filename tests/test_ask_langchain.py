import httpx
from openai import APIConnectionError

from tests.conftest import FakeLangChainChain


async def test_ask_langchain_returns_answer_with_valid_citation(client, mock_langchain_llm):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    resp = await client.post("/ask/langchain", json={"query": "boiling point of water"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_count"] == 1
    assert len(body["citations"]) == 1
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["filename"] == "notes.txt"
    assert body["all_citations_valid"] is True


async def test_ask_langchain_flags_fabricated_citation_numbers(client, monkeypatch):
    """Citations are verified here too, not trusted just because this path goes through a framework."""
    from app.langchain_port import chain as chain_module

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake_fn(input_dict):
        # Only 1 chunk was ever retrieved -- [99] does not exist.
        return "A real claim [1] and a fabricated one [99]."

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))

    resp = await client.post("/ask/langchain", json={"query": "boiling point of water"})
    body = resp.json()
    assert body["all_citations_valid"] is False
    assert [c["number"] for c in body["citations"]] == [1]


async def test_ask_langchain_against_empty_corpus_refuses_without_calling_llm(client, monkeypatch):
    """Same deterministic-refusal guarantee as the hand-built /ask: no evidence, no LLM call."""
    from app.langchain_port import chain as chain_module

    called = False

    async def fake_fn(input_dict):
        nonlocal called
        called = True
        return "unused"

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))

    resp = await client.post("/ask/langchain", json={"query": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_count"] == 0
    assert body["answer"] == "I don't know based on the provided documents."
    assert body["citations"] == []
    assert called is False


async def test_ask_langchain_provider_failure_returns_502(client, monkeypatch):
    from app.langchain_port import chain as chain_module

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Some content that will be retrieved.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def failing(input_dict):
        raise APIConnectionError(request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(failing))

    resp = await client.post("/ask/langchain", json={"query": "anything"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "provider_error"


async def test_ask_langchain_rejects_empty_query(client):
    resp = await client.post("/ask/langchain", json={"query": ""})
    assert resp.status_code == 422


async def test_ask_langchain_respects_document_id_filter(client, mock_langchain_llm):
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
        "/ask/langchain", json={"query": "why is it blue", "document_id": doc2["id"]}
    )
    body = resp.json()
    assert body["retrieved_count"] >= 1
    assert all(c["document_id"] == doc2["id"] for c in body["citations"])
    assert doc1["id"] not in {c["document_id"] for c in body["citations"]}


async def test_ask_and_ask_langchain_return_the_same_response_shape(client, monkeypatch):
    """
    The actual point of the Day 35 port: identical external contract
    (same JSON keys, same retrieved_count, same citation structure) from
    two different internal implementations, given the same request.
    """
    from app.langchain_port import chain as chain_module

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake_get_chat_completion(messages):
        return "Water boils at 100C [1]."

    async def fake_fn(input_dict):
        return "Water boils at 100C [1]."

    monkeypatch.setattr(
        "app.services.generation_service.get_chat_completion", fake_get_chat_completion
    )
    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))

    r1 = (await client.post("/ask", json={"query": "boiling point"})).json()
    r2 = (await client.post("/ask/langchain", json={"query": "boiling point"})).json()

    assert set(r1.keys()) == set(r2.keys())
    assert r1["retrieved_count"] == r2["retrieved_count"]
    assert len(r1["citations"]) == len(r2["citations"])
    assert r1["all_citations_valid"] == r2["all_citations_valid"] is True


# ---- Week 6 (Sat): token accounting ---------------------------------------------------

async def _upload_water(client):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )


async def test_ask_langchain_reports_provider_token_usage(client, monkeypatch):
    from langchain_core.messages import AIMessage
    from app.langchain_port import chain as chain_module

    await _upload_water(client)

    async def fake_fn(input_dict):
        return AIMessage(
            content="Water boils at 100C [1].",
            usage_metadata={"input_tokens": 420, "output_tokens": 17, "total_tokens": 437},
        )

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))
    body = (await client.post("/ask/langchain", json={"query": "boiling point of water"})).json()
    assert body["usage"] == {"input_tokens": 420, "output_tokens": 17, "model_calls": 1}


async def test_missing_provider_usage_is_zero_tokens_but_still_one_call(client, mock_langchain_llm):
    """The fake returns a plain string: 0 tokens + 1 call reads as 'not reported', never 'free'."""
    await _upload_water(client)
    body = (await client.post("/ask/langchain", json={"query": "boiling point of water"})).json()
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0, "model_calls": 1}


async def test_no_retrieved_chunks_means_no_llm_call_and_zero_usage(client, mock_langchain_llm):
    body = (await client.post("/ask/langchain", json={"query": "anything at all"})).json()
    assert body["retrieved_count"] == 0
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
