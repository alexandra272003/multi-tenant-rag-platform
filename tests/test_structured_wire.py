"""
Day 37: the REAL LangChain ChatOpenAI client + OpenAI SDK, with only the
network replaced by httpx.MockTransport. test_ask_structured.py fakes the
whole chain; this file proves the pieces the fake skips: what actually
goes over the wire, and that a malformed reply reaches the retry loop
instead of exploding inside the SDK (the bug that motivated binding
response_format and parsing ourselves -- see structured.py's docstring).
"""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

GOOD = json.dumps({"answerable": True, "answer": "100C [1].", "cited_sources": [1]})


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def wire(monkeypatch):
    """Install a real ChatOpenAI behind a scripted mock HTTP server."""
    from app.langchain_port import structured as mod

    def install(replies: list[str]):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return completion(replies[min(len(requests) - 1, len(replies) - 1)])

        real_llm = ChatOpenAI(
            model="m", api_key="k", base_url="http://mock/v1", max_retries=0,
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        monkeypatch.setattr(mod, "_structured_chain", mod._prompt | real_llm.bind(**mod._bind_kwargs()))
        return requests

    return install


@pytest.fixture
async def doc(client):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )


async def test_request_carries_json_schema_response_format(client, doc, wire):
    requests = wire([GOOD])
    resp = await client.post("/ask/structured", json={"query": "boiling point"})
    assert resp.status_code == 200
    rf = requests[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "GroundedAnswer"
    assert set(rf["json_schema"]["schema"]["properties"]) == {"answerable", "answer", "cited_sources"}
    assert requests[0]["messages"][0]["role"] == "system"


async def test_non_json_reply_is_retried_not_a_500(client, doc, wire):
    requests = wire(["Sure! The answer is 100C.", GOOD])
    resp = await client.post("/ask/structured", json={"query": "boiling point"})
    assert resp.status_code == 200
    assert resp.json()["attempts"] == 2
    # Second request really contains the bad reply + the correction.
    msgs = requests[1]["messages"]
    assert msgs[-2] == {"role": "assistant", "content": "Sure! The answer is 100C."}
    assert msgs[-1]["role"] == "user" and "invalid" in msgs[-1]["content"].lower()


async def test_persistently_bad_replies_end_in_502_after_three_http_calls(client, doc, wire):
    requests = wire(["nope"])
    resp = await client.post("/ask/structured", json={"query": "boiling point"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "invalid_model_output"
    assert len(requests) == 3


async def test_provider_http_error_maps_to_provider_error_without_validation_retry(
    client, doc, monkeypatch
):
    from app.langchain_port import structured as mod

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key", "type": "auth"}})

    real_llm = ChatOpenAI(
        model="m", api_key="k", base_url="http://mock/v1", max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(mod, "_structured_chain", mod._prompt | real_llm.bind(**mod._bind_kwargs()))
    resp = await client.post("/ask/structured", json={"query": "boiling point"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "provider_error"
    assert calls == 1  # auth failures are not worth re-asking


@pytest.mark.parametrize(
    "method,expected",
    [("json_schema", "json_schema"), ("json_mode", "json_object")],
)
def test_structured_method_setting_changes_response_format(monkeypatch, method, expected):
    from app.core.config import settings
    from app.langchain_port import structured as mod

    monkeypatch.setattr(settings, "llm_structured_method", method)
    assert mod._bind_kwargs()["response_format"]["type"] == expected


def test_structured_method_none_sends_no_response_format(monkeypatch):
    from app.core.config import settings
    from app.langchain_port import structured as mod

    monkeypatch.setattr(settings, "llm_structured_method", "none")
    assert mod._bind_kwargs() == {}


def test_unknown_structured_method_fails_loudly(monkeypatch):
    from app.core.config import settings
    from app.langchain_port import structured as mod

    monkeypatch.setattr(settings, "llm_structured_method", "yolo")
    with pytest.raises(ValueError):
        mod._bind_kwargs()
