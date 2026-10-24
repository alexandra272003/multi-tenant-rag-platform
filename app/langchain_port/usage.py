"""
Week 6 (Sat): token accounting for the direct and structured endpoints.

The agent already sums provider usage across its steps (agent/loop.py).
This module gives /ask/langchain and /ask/structured the same numbers so
the three modes can be compared on cost, not just on answer quality.

Best-effort by design: LangChain exposes `usage_metadata` on an AIMessage
when the provider reports it. A provider (or a test fake) that returns a
plain string, or a message without usage, counts as 0 tokens -- the
`model_calls` field still says how many calls were made, so "0 tokens,
1 call" reads as "not reported", never as "free".
"""


def new_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}


def add_message_usage(total: dict, message) -> None:
    """Add one model call (an AIMessage or a plain string) to `total`."""
    total["model_calls"] += 1
    meta = getattr(message, "usage_metadata", None) or {}
    total["input_tokens"] += meta.get("input_tokens", 0) or 0
    total["output_tokens"] += meta.get("output_tokens", 0) or 0
