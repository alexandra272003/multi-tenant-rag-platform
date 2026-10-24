"""
Week 6: the agent against the REAL LangChain ChatOpenAI client and OpenAI
SDK streaming parser, with only the network replaced by httpx.MockTransport.
The other agent tests script the model; this one proves what they skip:
the tool schemas that actually go over the wire, how tool-call deltas are
reassembled from an SSE stream, and that tool results go back in the
format the provider expects. (Same reason test_structured_wire.py exists:
mocked tests passed while the real client behaved differently.)
"""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from tests.agent_helpers import parse_sse


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _sse_body(*chunks: dict) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


TOOL_CALL_TURN = _sse_body(
    _chunk({"role": "assistant", "content": None, "tool_calls": [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "calculator", "arguments": ""}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"expression": "2+'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '2"}'}}]}),
    _chunk({}, "tool_calls"),
)
ANSWER_TURN = _sse_body(
    _chunk({"role": "assistant", "content": "The answer"}),
    _chunk({"content": " is 4."}),
    _chunk({}, "stop"),
)


@pytest.fixture
def wire(monkeypatch):
    from app.agent import loop

    def install(turns: list[bytes]):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            body = turns[min(len(requests) - 1, len(turns) - 1)]
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        monkeypatch.setattr(loop, "llm", ChatOpenAI(
            model="m", api_key="k", base_url="http://mock/v1", max_retries=0,
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
        return requests

    return install


async def test_real_client_round_trip_through_a_tool(client, wire):
    requests = wire([TOOL_CALL_TURN, ANSWER_TURN])
    resp = await client.post("/agent", json={"query": "what is 2+2?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The answer is 4."
    assert body["trace"][0]["args"] == {"expression": "2+2"}  # reassembled from split deltas
    assert body["trace"][1]["preview"] == "4"

    first, second = requests
    assert first["stream"] is True
    assert {t["function"]["name"] for t in first["tools"]} == {
        "calculator", "document_search", "query_documents"}
    calc = next(t for t in first["tools"] if t["function"]["name"] == "calculator")["function"]
    assert calc["parameters"]["required"] == ["expression"]
    assert calc["parameters"]["additionalProperties"] is False  # extra="forbid" reaches the provider

    msgs = second["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1" and tool_msg["content"] == "4"
    assert msgs[0]["role"] == "system" and "UNTRUSTED" in msgs[0]["content"]


async def test_real_client_streams_tokens_over_sse(client, wire):
    wire([TOOL_CALL_TURN, ANSWER_TURN])
    events = parse_sse((await client.post("/agent/stream", json={"query": "2+2?"})).text)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "tool_call" and kinds[-2:] == ["final", "done"]
    assert "".join(e["text"] for e in events if e["event"] == "token") == "The answer is 4."


async def test_only_allowed_tools_go_over_the_wire(client, wire):
    requests = wire([ANSWER_TURN])
    await client.post("/agent", json={"query": "hi", "allowed_tools": ["calculator"]})
    assert [t["function"]["name"] for t in requests[0]["tools"]] == ["calculator"]


async def test_forced_final_call_sends_no_tools(client, wire):
    # Model keeps asking for (different) tool calls; with max_steps=1 the loop
    # must make a second, tool-less call and take whatever it answers.
    other = _sse_body(
        _chunk({"role": "assistant", "content": None, "tool_calls": [
            {"index": 0, "id": "call_9", "type": "function",
             "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}]}),
        _chunk({}, "tool_calls"))
    requests = wire([other, ANSWER_TURN])
    body = (await client.post("/agent", json={"query": "q", "max_steps": 1})).json()
    assert body["budget_exhausted"] is True and body["answer"] == "The answer is 4."
    assert "tools" not in requests[1]
