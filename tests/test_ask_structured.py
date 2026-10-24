"""
Day 37: structured output + retry-on-invalid-output.

The fake chain returns a plain AIMessage whose content is the model's raw
text (JSON, garbage, whatever the test scripts). Parsing + validation run
for real through GroundedAnswer, so a payload that is invalid here is
invalid for the same reason it would be in production. The real
LangChain/OpenAI-SDK request path is covered separately in
test_structured_wire.py.
"""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError
from pydantic import ValidationError

from app.schemas import GroundedAnswer
from tests.conftest import FakeLangChainChain


def model_message(payload: dict | str) -> AIMessage:
    """What the model returns: a plain message whose content is JSON text."""
    return AIMessage(content=payload if isinstance(payload, str) else json.dumps(payload))


GOOD = {"answerable": True, "answer": "Water boils at 100C [1].", "cited_sources": [1]}


@pytest.fixture
def scripted(monkeypatch):
    """
    Install a scripted fake model: each call returns the next reply in the
    list (the last one repeats). Returns a `calls` list holding the input
    dict of every call, so tests can assert on count and on feedback.
    """
    from app.langchain_port import structured as mod

    def install(replies: list):
        calls: list[dict] = []

        async def fn(input_dict):
            calls.append(input_dict)
            reply = replies[min(len(calls) - 1, len(replies) - 1)]
            if isinstance(reply, Exception):
                raise reply
            return model_message(reply)

        monkeypatch.setattr(mod, "_structured_chain", FakeLangChainChain(fn))
        return calls

    return install


@pytest.fixture
async def doc(client):
    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )


# ---- schema contract (no HTTP) --------------------------------------------


def test_schema_accepts_valid_answer():
    a = GroundedAnswer(answerable=True, answer="x [1]", cited_sources=[1])
    assert a.cited_sources == [1]


def test_schema_normalizes_duplicate_and_unordered_citations():
    a = GroundedAnswer(answerable=True, answer="x", cited_sources=[3, 1, 3])
    assert a.cited_sources == [1, 3]


@pytest.mark.parametrize(
    "payload",
    [
        {"answerable": True, "answer": "x", "cited_sources": []},  # answerable but no cites
        {"answerable": False, "answer": "x", "cited_sources": [1]},  # refusal with cites
        {"answerable": True, "answer": "", "cited_sources": [1]},  # empty answer
        {"answerable": True, "answer": "x", "cited_sources": [0]},  # sources are 1-based
        {"answerable": "maybe", "answer": "x", "cited_sources": [1]},  # not a bool
        {"answer": "x", "cited_sources": [1]},  # missing field
    ],
)
def test_schema_rejects_contract_violations(payload):
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate(payload)


# ---- happy path ------------------------------------------------------------


async def test_valid_first_attempt_makes_one_call(client, doc, scripted):
    calls = scripted([GOOD])
    resp = await client.post("/ask/structured", json={"query": "boiling point of water"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answerable"] is True
    assert body["attempts"] == 1
    assert body["validation_errors"] == []
    assert [c["number"] for c in body["citations"]] == [1]
    assert body["citations"][0]["filename"] == "notes.txt"
    assert len(calls) == 1
    assert calls[0]["feedback"] == []  # nothing to correct on attempt 1


async def test_response_has_the_documented_keys(client, doc, scripted):
    scripted([GOOD])
    body = (await client.post("/ask/structured", json={"query": "boiling"})).json()
    assert set(body) == {
        "query", "answerable", "answer", "citations",
        "retrieved_count", "attempts", "validation_errors", "usage",
    }


# ---- retry on invalid output ----------------------------------------------


async def test_invalid_then_valid_retries_and_recovers(client, doc, scripted):
    calls = scripted(["this is not json at all", GOOD])
    resp = await client.post("/ask/structured", json={"query": "boiling point"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["attempts"] == 2
    assert len(body["validation_errors"]) == 1
    assert len(calls) == 2


async def test_retry_feeds_back_bad_output_and_the_specific_error(client, doc, scripted):
    bad = '{"answerable": true, "answer": "Boils at 100C", "cited_sources": []}'
    calls = scripted([bad, GOOD])
    await client.post("/ask/structured", json={"query": "boiling point"})

    feedback = calls[1]["feedback"]
    assert len(feedback) == 2
    assert "Boils at 100C" in feedback[0].content  # the model sees its own bad output
    assert "cited_sources" in feedback[1].content  # ...and what was wrong with it
    assert "invalid" in feedback[1].content.lower()


async def test_out_of_range_citation_is_a_semantic_failure_and_retries(client, doc, scripted):
    # Schema-valid (a positive int) but only 1 chunk was retrieved.
    calls = scripted([{"answerable": True, "answer": "x [99]", "cited_sources": [99]}, GOOD])
    body = (await client.post("/ask/structured", json={"query": "boiling"})).json()
    assert body["attempts"] == 2
    assert "99" in body["validation_errors"][0]
    assert "1..1" in calls[1]["feedback"][1].content
    assert [c["number"] for c in body["citations"]] == [1]


async def test_answerable_without_citations_triggers_retry(client, doc, scripted):
    scripted([{"answerable": True, "answer": "x", "cited_sources": []}, GOOD])
    body = (await client.post("/ask/structured", json={"query": "boiling"})).json()
    assert body["attempts"] == 2


async def test_retries_exhausted_returns_502_invalid_model_output(client, doc, scripted):
    calls = scripted(["garbage"])  # last reply repeats forever
    resp = await client.post("/ask/structured", json={"query": "boiling"})
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["code"] == "invalid_model_output"
    assert err["details"]["attempts"] == 3  # 1 + default 2 retries
    assert len(err["details"]["errors"]) == 3
    assert len(calls) == 3


async def test_validation_retry_budget_is_configurable(client, doc, scripted, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_validation_retries", 0)
    calls = scripted(["garbage", GOOD])
    resp = await client.post("/ask/structured", json={"query": "boiling"})
    assert resp.status_code == 502
    assert len(calls) == 1


async def test_feedback_is_bounded_to_the_last_failed_attempt(client, doc, scripted):
    calls = scripted(["bad one", "bad two", GOOD])
    await client.post("/ask/structured", json={"query": "boiling"})
    assert len(calls) == 3
    assert len(calls[2]["feedback"]) == 2  # not 4: prompt does not grow per retry
    assert "bad two" in calls[2]["feedback"][0].content
    assert "bad one" not in calls[2]["feedback"][0].content


async def test_echoed_bad_output_is_truncated(client, doc, scripted):
    calls = scripted(["x" * 50_000, GOOD])
    await client.post("/ask/structured", json={"query": "boiling"})
    assert len(calls[1]["feedback"][0].content) <= 2000


# ---- refusals, empty corpus, provider failures ----------------------------


async def test_not_answerable_is_normalized_to_the_standard_refusal(client, doc, scripted):
    scripted([{"answerable": False, "answer": "Hmm, maybe 42?", "cited_sources": []}])
    body = (await client.post("/ask/structured", json={"query": "meaning of life"})).json()
    assert body["answerable"] is False
    assert body["answer"] == "I don't know based on the provided documents."
    assert body["citations"] == []
    assert body["attempts"] == 1


async def test_empty_corpus_refuses_without_calling_llm(client, scripted):
    calls = scripted([GOOD])
    body = (await client.post("/ask/structured", json={"query": "anything"})).json()
    assert body["retrieved_count"] == 0
    assert body["attempts"] == 0
    assert body["answerable"] is False
    assert calls == []


async def test_provider_error_is_502_provider_error_and_not_retried_here(client, doc, scripted):
    calls = scripted([APIConnectionError(request=httpx.Request("POST", "http://test"))])
    resp = await client.post("/ask/structured", json={"query": "boiling"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "provider_error"
    # Transport retries belong to ChatOpenAI; the validation loop must not
    # multiply them by re-asking after an APIError.
    assert len(calls) == 1


async def test_empty_reply_is_invalid_and_retries(client, doc, scripted):
    calls = scripted(["", GOOD])
    body = (await client.post("/ask/structured", json={"query": "boiling"})).json()
    assert body["attempts"] == 2
    assert "empty" in body["validation_errors"][0]
    assert calls[1]["feedback"][0].content == "(empty)"


async def test_json_wrapped_in_a_code_fence_is_accepted_without_retry(client, doc, scripted):
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    calls = scripted([fenced])
    body = (await client.post("/ask/structured", json={"query": "boiling"})).json()
    assert body["attempts"] == 1
    assert len(calls) == 1


async def test_extra_unknown_fields_from_the_model_are_ignored(client, doc, scripted):
    scripted([{**GOOD, "confidence": 0.9}])
    resp = await client.post("/ask/structured", json={"query": "boiling"})
    assert resp.status_code == 200


# ---- request validation + filters -----------------------------------------


async def test_rejects_empty_query(client):
    assert (await client.post("/ask/structured", json={"query": ""})).status_code == 422


async def test_respects_document_id_filter(client, scripted):
    d1 = (await client.post("/documents", files={"file": ("a.txt", b"The sky is blue.")},
                            params={"chunk_size": 500, "chunk_overlap": 0})).json()
    d2 = (await client.post("/documents", files={"file": ("b.txt", b"The ocean is blue too.")},
                            params={"chunk_size": 500, "chunk_overlap": 0})).json()
    scripted([GOOD])
    body = (await client.post("/ask/structured",
                              json={"query": "blue", "document_id": d2["id"]})).json()
    assert body["retrieved_count"] >= 1
    assert {c["document_id"] for c in body["citations"]} <= {d2["id"]}
    assert d1["id"] not in {c["document_id"] for c in body["citations"]}


# ---- Week 6 (Sat): token accounting ---------------------------------------------------

async def test_structured_usage_sums_every_attempt_including_failed_ones(client, monkeypatch):
    """A retry is real spend: 2 attempts must report 2 calls and the SUM of their tokens."""
    from app.langchain_port import structured as mod

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )
    replies = [
        AIMessage(content="not json at all", usage_metadata={"input_tokens": 300, "output_tokens": 10, "total_tokens": 310}),
        AIMessage(content=json.dumps(GOOD), usage_metadata={"input_tokens": 340, "output_tokens": 25, "total_tokens": 365}),
    ]
    calls = []

    async def fn(input_dict):
        calls.append(input_dict)
        return replies[len(calls) - 1]

    monkeypatch.setattr(mod, "_structured_chain", FakeLangChainChain(fn))
    body = (await client.post("/ask/structured", json={"query": "boiling point of water"})).json()
    assert body["attempts"] == 2
    assert body["usage"] == {"input_tokens": 640, "output_tokens": 35, "model_calls": 2}


async def test_structured_no_retrieval_reports_zero_usage(client):
    body = (await client.post("/ask/structured", json={"query": "anything at all"})).json()
    assert body["attempts"] == 0
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
