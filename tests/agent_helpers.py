"""Shared helpers for the agent tests: a scripted fake chat model that
streams AIMessageChunks exactly the way a real provider does (tool-call
arguments arrive in PIECES, which is what makes chunk accumulation worth
testing), plus small builders."""
import json

from langchain_core.messages import AIMessageChunk


def text_turn(*parts: str) -> list:
    return [AIMessageChunk(content=p) for p in parts]


def tool_turn(name: str, args, call_id: str = "call_1", raw_args: str | None = None) -> list:
    """One model turn that calls a tool. Args are split mid-JSON across two
    chunks. Pass raw_args to send deliberately malformed JSON."""
    s = raw_args if raw_args is not None else json.dumps(args)
    half = max(1, len(s) // 2)
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": s[:half], "id": call_id, "index": 0}]),
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": s[half:], "id": None, "index": 0}]),
    ]


def multi_tool_turn(calls: list[tuple[str, dict, str]]) -> list:
    chunks = []
    for i, (name, args, cid) in enumerate(calls):
        chunks.append(AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": json.dumps(args), "id": cid, "index": i}]))
    return chunks


class ScriptedModel:
    """astream() replays the next scripted turn; the last turn repeats.
    Records every messages list it was called with and the tools offered."""

    def __init__(self, turns: list):
        self.turns = turns
        self.calls: list[list] = []
        self.tools_offered: list[list[str]] = []

    async def astream(self, messages):
        self.calls.append(list(messages))
        turn = self.turns[min(len(self.calls) - 1, len(self.turns) - 1)]
        if isinstance(turn, Exception):
            raise turn
        for chunk in turn:
            yield chunk


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        if "data" in lines:
            events.append(json.loads(lines["data"]))
    return events
