"""Days 36/38: the agent loop -- tool calling, streaming, termination."""
import httpx
import pytest
from openai import APIConnectionError

from tests.agent_helpers import multi_tool_turn, parse_sse, text_turn, tool_turn


async def upload(client, name: str, body: str) -> int:
    resp = await client.post("/documents", files={"file": (name, body.encode())},
                             params={"chunk_size": 500, "chunk_overlap": 0})
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


@pytest.fixture
async def doc_id(client):
    return await upload(client, "policy.txt", "Employees get 25 days of annual leave per year.")


# ---- no tools needed --------------------------------------------------------


async def test_direct_answer_without_tools(client, scripted_agent):
    model = scripted_agent([text_turn("Hello", " there.")])
    resp = await client.post("/agent", json={"query": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Hello there."
    assert body["steps"] == 1 and body["tool_calls_used"] == 0
    assert body["trace"] == [] and body["budget_exhausted"] is False
    assert len(model.calls) == 1


async def test_all_three_tools_offered_by_default(client, scripted_agent):
    model = scripted_agent([text_turn("ok")])
    await client.post("/agent", json={"query": "hi"})
    assert model.tools_offered[0] == ["calculator", "document_search", "query_documents"]


async def test_allowed_tools_restricts_what_is_offered(client, scripted_agent):
    model = scripted_agent([text_turn("ok")])
    await client.post("/agent", json={"query": "hi", "allowed_tools": ["calculator"]})
    assert model.tools_offered[0] == ["calculator"]


async def test_unknown_tool_name_in_request_is_422(client):
    resp = await client.post("/agent", json={"query": "hi", "allowed_tools": ["shell"]})
    assert resp.status_code == 422


# ---- the loop ------------------------------------------------------------------


async def test_calculator_round_trip_and_trace(client, scripted_agent):
    model = scripted_agent([
        tool_turn("calculator", {"expression": "(12.5 * 4) / 3 + 2 ** 3"}),
        text_turn("The result is ", "about 24.67."),
    ])
    body = (await client.post("/agent", json={"query": "compute"})).json()
    assert body["answer"] == "The result is about 24.67."
    assert body["steps"] == 2 and body["tool_calls_used"] == 1
    kinds = [e["event"] for e in body["trace"]]
    assert kinds == ["tool_call", "tool_result"]
    assert body["trace"][0]["args"] == {"expression": "(12.5 * 4) / 3 + 2 ** 3"}
    assert body["trace"][1]["ok"] is True and body["trace"][1]["preview"] == "24.66666667"
    # The tool result really was fed back to the model, as a tool message.
    second_call = model.calls[1]
    assert second_call[-1].tool_call_id == "call_1" and second_call[-1].content == "24.66666667"
    assert second_call[-2].tool_calls[0]["name"] == "calculator"  # the assistant turn is replayed


async def test_document_search_returns_numbered_wrapped_untrusted_text(client, doc_id, scripted_agent):
    model = scripted_agent([
        tool_turn("document_search", {"query": "annual leave"}),
        text_turn("Employees get 25 days [1]."),
    ])
    body = (await client.post("/agent", json={"query": "leave?"})).json()
    tool_msg = model.calls[1][-1].content
    assert '<retrieved_document source="[1] policy.txt"' in tool_msg
    assert 'trust="untrusted"' in tool_msg and "25 days" in tool_msg
    assert body["all_citations_valid"] is True
    assert [c["number"] for c in body["citations"]] == [1]
    assert body["citations"][0]["filename"] == "policy.txt"


async def test_fabricated_citation_is_flagged_and_not_returned(client, doc_id, scripted_agent):
    scripted_agent([tool_turn("document_search", {"query": "leave"}),
                    text_turn("Per [1] and [7], 25 days.")])
    body = (await client.post("/agent", json={"query": "leave?"})).json()
    assert body["all_citations_valid"] is False
    assert [c["number"] for c in body["citations"]] == [1]  # [7] never retrieved


async def test_source_numbers_are_stable_across_calls(client, scripted_agent):
    a = await upload(client, "a.txt", "Alpha document text.")
    b = await upload(client, "b.txt", "Beta document text.")
    model = scripted_agent([
        tool_turn("document_search", {"query": "x", "document_id": a}, "c1"),
        tool_turn("document_search", {"query": "x", "document_id": b}, "c2"),
        text_turn("Done [1][2]."),
    ])
    body = (await client.post("/agent", json={"query": "q"})).json()
    assert [c["filename"] for c in body["citations"]] == ["a.txt", "b.txt"]
    assert 'source="[2] b.txt"' in model.calls[2][-1].content


async def test_query_documents_counts_metadata(client, doc_id, scripted_agent):
    model = scripted_agent([
        tool_turn("query_documents", {"sql": "select filename, chunk_count from documents"}),
        text_turn("One document."),
    ])
    body = (await client.post("/agent", json={"query": "how many docs"})).json()
    result = model.calls[1][-1].content
    assert "policy.txt" in result and '<tool_data' in result
    assert body["trace"][1]["ok"] is True


# ---- recovery from bad tool calls ---------------------------------------------------


async def test_invalid_arguments_are_reported_to_the_model_not_raised(client, scripted_agent):
    model = scripted_agent([tool_turn("calculator", {"expression": ""}), text_turn("sorry")])
    body = (await client.post("/agent", json={"query": "q"})).json()
    assert body["trace"][1]["ok"] is False and "invalid arguments" in body["trace"][1]["error"]
    assert model.calls[1][-1].content.startswith("ERROR: invalid arguments")


async def test_malformed_json_arguments_become_an_error_result(client, scripted_agent):
    model = scripted_agent([tool_turn("calculator", None, raw_args="this is not json at all"),
                            text_turn("sorry")])
    body = (await client.post("/agent", json={"query": "q"})).json()
    assert body["trace"][1]["ok"] is False and "not valid JSON" in body["trace"][1]["error"]
    assert model.calls[1][-1].content.startswith("ERROR")


async def test_unknown_tool_is_denied(client, scripted_agent):
    scripted_agent([tool_turn("send_email", {"to": "x"}), text_turn("ok")])
    body = (await client.post("/agent", json={"query": "q"})).json()
    assert body["trace"][1]["event"] == "tool_denied"
    assert "not available" in body["trace"][1]["reason"]


async def test_calculator_errors_are_results_not_crashes(client, scripted_agent):
    scripted_agent([tool_turn("calculator", {"expression": "1/0"}), text_turn("undefined")])
    body = (await client.post("/agent", json={"query": "q"})).json()
    assert body["trace"][1]["error"] == "division by zero"


# ---- termination -----------------------------------------------------------------------------


async def test_duplicate_call_is_rejected(client, scripted_agent):
    model = scripted_agent([
        tool_turn("calculator", {"expression": "1+1"}, "c1"),
        tool_turn("calculator", {"expression": "1+1"}, "c2"),
        text_turn("2"),
    ])
    body = (await client.post("/agent", json={"query": "q"})).json()
    results = [e for e in body["trace"] if e["event"] == "tool_result"]
    assert results[0]["ok"] is True and results[1]["ok"] is False
    assert "duplicate" in results[1]["error"]
    assert len(model.calls) == 3


async def test_step_budget_forces_a_final_answer_with_tools_unbound(client, scripted_agent):
    counter = iter(range(100))

    class Endless(list):  # a model that never stops calling tools, with fresh args each time
        pass

    turns = [tool_turn("calculator", {"expression": f"{i}+1"}, f"c{i}") for i in range(10)]
    model = scripted_agent(turns[:2] + [text_turn("Best answer so far.")])
    # 2 steps allowed: step1 call, step2 call, then forced final with no tools.
    body = (await client.post("/agent", json={"query": "q", "max_steps": 2})).json()
    assert body["budget_exhausted"] is True
    assert body["answer"] == "Best answer so far."
    assert model.tools_offered[-1] == []  # the forced call offers no tools
    assert "Tool budget reached" in model.calls[-1][-1].content
    assert body["trace"][-1]["event"] == "budget_exhausted"


async def test_total_tool_call_budget_denies_excess_calls(client, scripted_agent, monkeypatch):
    from app.agent import loop  # noqa: F401
    scripted_agent([
        multi_tool_turn([("calculator", {"expression": f"{i}+0"}, f"c{i}") for i in range(3)]),
        multi_tool_turn([("calculator", {"expression": f"{i}+9"}, f"d{i}") for i in range(3)]),
        multi_tool_turn([("calculator", {"expression": f"{i}+8"}, f"e{i}") for i in range(3)]),
        text_turn("done"),
    ])
    body = (await client.post("/agent", json={"query": "q", "max_steps": 6})).json()
    denied = [e for e in body["trace"] if e["event"] == "tool_denied"]
    assert any("budget" in d["reason"] for d in denied)  # >6 total calls were refused


async def test_per_step_call_cap(client, scripted_agent):
    scripted_agent([
        multi_tool_turn([("calculator", {"expression": f"{i}+1"}, f"c{i}") for i in range(5)]),
        text_turn("done"),
    ])
    body = (await client.post("/agent", json={"query": "q"})).json()
    denied = [e for e in body["trace"] if e["event"] == "tool_denied"]
    assert len(denied) == 2 and all("too many" in d["reason"] for d in denied)


async def test_oversized_tool_output_is_truncated_for_the_model(client, scripted_agent):
    big = "word " * 5000
    await upload(client, "big.txt", big)
    model = scripted_agent([tool_turn("document_search", {"query": "word", "top_k": 5}),
                            text_turn("ok")])
    await client.post("/agent", json={"query": "q"})
    fed_back = model.calls[1][-1].content
    assert len(fed_back) <= 4000 + 40 and fed_back.endswith("...[truncated]")


# ---- provider failure ------------------------------------------------------------------------------


async def test_provider_error_is_a_502(client, scripted_agent):
    scripted_agent([APIConnectionError(request=httpx.Request("POST", "http://x"))])
    resp = await client.post("/agent", json={"query": "q"})
    assert resp.status_code == 502 and resp.json()["error"]["code"] == "provider_error"


# ---- streaming endpoint ---------------------------------------------------------------------------------


async def test_stream_event_order_and_content(client, scripted_agent):
    scripted_agent([tool_turn("calculator", {"expression": "6*7"}),
                    text_turn("The ", "answer ", "is 42.")])
    resp = await client.post("/agent/stream", json={"query": "q"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    events = parse_sse(resp.text)
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["tool_call", "tool_result"]
    assert kinds[-2:] == ["final", "done"]
    assert "".join(e["text"] for e in events if e["event"] == "token") == "The answer is 42."
    assert events[-2]["answer"] == "The answer is 42."


async def test_stream_tokens_arrive_incrementally(client, scripted_agent):
    scripted_agent([text_turn("a", "b", "c", "d")])
    events = parse_sse((await client.post("/agent/stream", json={"query": "q"})).text)
    assert len([e for e in events if e["event"] == "token"]) >= 2  # not one big blob


async def test_stream_reports_provider_failure_as_an_error_event(client, scripted_agent):
    scripted_agent([APIConnectionError(request=httpx.Request("POST", "http://x"))])
    resp = await client.post("/agent/stream", json={"query": "q"})
    assert resp.status_code == 200  # status was already sent; error travels in-band
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "error" and events[-1]["code"] == "provider_error"
    assert not any(e["event"] == "done" for e in events)


async def test_stream_and_non_stream_agree(client, doc_id, scripted_agent):
    turns = [tool_turn("document_search", {"query": "leave"}), text_turn("25 days [1].")]
    scripted_agent(turns)
    plain = (await client.post("/agent", json={"query": "q"})).json()
    scripted_agent(turns)
    final = [e for e in parse_sse((await client.post("/agent/stream", json={"query": "q"})).text)
             if e["event"] == "final"][0]
    assert final["answer"] == plain["answer"]
    assert [c["chunk_id"] for c in final["citations"]] == [c["chunk_id"] for c in plain["citations"]]
