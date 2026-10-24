"""CITATION_RE is shared by /ask, /ask/langchain and the agent. gpt-oss models sometimes cite
with fullwidth brackets (【1】) instead of [1]; that must count as a citation, not as 'none'."""
import pytest
from langchain_core.messages import AIMessage

from app.services.generation_service import CITATION_RE
from tests.conftest import FakeLangChainChain


@pytest.mark.parametrize("text,expected", [
    ("20 days [1].", {1}),
    ("Both apply [1][2] and [12].", {1, 2, 12}),
    ("20 days 【1】.", {1}),
    ("Cited 【2†source】 and 【3†L5-L9】.", {2, 3}),
    ("Mixed [1] and 【4】.", {1, 4}),
    ("A [1]", {1}),                       # narrow no-break space before the bracket
    ("not a citation [abc] or [1a] or 【x】", set()),
    ("no citation at all", set()),
])
def test_citation_regex_accepts_ascii_and_fullwidth_brackets(text, expected):
    assert {int(n) for n in CITATION_RE.findall(text)} == expected


async def test_langchain_endpoint_counts_a_fullwidth_citation(client, monkeypatch):
    from app.langchain_port import chain as chain_module

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake_fn(input_dict):
        return AIMessage(content="Water boils at **100 C**【1】.")

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))
    body = (await client.post("/ask/langchain", json={"query": "boiling point of water"})).json()
    assert [c["filename"] for c in body["citations"]] == ["notes.txt"]
    assert body["all_citations_valid"] is True


async def test_fullwidth_fabricated_citation_is_still_flagged(client, monkeypatch):
    """Being tolerant about the bracket style must not make fabricated numbers pass."""
    from app.langchain_port import chain as chain_module

    await client.post(
        "/documents",
        files={"file": ("notes.txt", b"Water boils at 100 degrees Celsius at sea level.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    async def fake_fn(input_dict):
        return AIMessage(content="Real claim 【1】 and invented claim 【99】.")

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))
    body = (await client.post("/ask/langchain", json={"query": "boiling point of water"})).json()
    assert body["all_citations_valid"] is False
