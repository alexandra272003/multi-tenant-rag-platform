"""Week 6 (Sat): the comparison runner -- scoring and aggregation are pure functions,
and the whole runner is exercised in-process against the real app with scripted models."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from tests.agent_helpers import text_turn, tool_turn
from tests.conftest import FakeLangChainChain

spec = importlib.util.spec_from_file_location("run_compare", Path(__file__).resolve().parent.parent / "eval" / "run_compare.py")
rc = importlib.util.module_from_spec(spec)
sys.modules["run_compare"] = rc
spec.loader.exec_module(rc)

LABELED = {"id": "f1", "type": "factoid", "question": "How many days?", "gold_document": "leave.txt", "gold_span": "20 days"}
NO_ANSWER = {"id": "n1", "type": "no_answer", "question": "Who won 1998?", "gold_document": None, "gold_span": None}


def body(answer="x [1]", cites=("leave.txt",), valid=True, **extra):
    return {"answer": answer, "citations": [{"filename": f} for f in cites], "all_citations_valid": valid,
            "retrieved_count": 3, **extra}


# ---- pure scoring -------------------------------------------------------------------


def test_labeled_question_answered_from_the_gold_document():
    r = rc.score_response(LABELED, "direct", 200, body(), 1.5)
    assert r["answered"] and r["source_correct"] is True and r["correct_refusal"] is None and not r["error"]


def test_answer_citing_the_wrong_document_is_not_source_correct():
    r = rc.score_response(LABELED, "direct", 200, body(cites=("faq.txt",)), 1.0)
    assert r["answered"] and r["source_correct"] is False


def test_refusing_an_answerable_question_counts_against_answer_rate():
    r = rc.score_response(LABELED, "direct", 200, body(answer="I don't know based on the provided documents.", cites=()), 1.0)
    assert r["refused"] and not r["answered"] and r["source_correct"] is False


def test_no_answer_question_is_scored_on_refusal_not_sources():
    refused = rc.score_response(NO_ANSWER, "direct", 200, body(answer="I don't know based on the provided documents.", cites=()), 1.0)
    hallucinated = rc.score_response(NO_ANSWER, "direct", 200, body(answer="Argentina [1]", cites=("faq.txt",)), 1.0)
    assert refused["correct_refusal"] is True and refused["source_correct"] is None
    assert hallucinated["correct_refusal"] is False


def test_structured_mode_refusal_uses_the_answerable_flag():
    b = {"answerable": False, "answer": "I don't know based on the provided documents.", "citations": [], "attempts": 2}
    r = rc.score_response(NO_ANSWER, "structured", 200, b, 1.0)
    assert r["correct_refusal"] is True and r["model_calls"] == 2


def test_agent_mode_reports_tool_calls_and_tokens():
    b = body(tool_calls_used=2, steps=3, usage={"input_tokens": 900, "output_tokens": 100, "model_calls": 3})
    r = rc.score_response(LABELED, "agent", 200, b, 4.0)
    assert (r["tool_calls"], r["model_calls"], r["tokens"]) == (2, 3, 1000)


def test_invalid_citation_and_http_errors_are_recorded():
    assert rc.score_response(LABELED, "direct", 200, body(valid=False), 1.0)["invalid_citation"] is True
    err = rc.score_response(LABELED, "direct", 502, {}, 1.0)
    assert err["error"] and "answered" not in err


def test_aggregate_rates_and_latency():
    rows = [
        rc.score_response(LABELED, "direct", 200, body(), 1.0),
        rc.score_response({**LABELED, "id": "f2"}, "direct", 200, body(cites=("faq.txt",)), 3.0),
        rc.score_response(NO_ANSWER, "direct", 200, body(answer="I don't know based on the provided documents.", cites=()), 2.0),
        rc.score_response({**LABELED, "id": "f3"}, "direct", 500, {}, 9.0),
    ]
    a = rc.aggregate(rows)["direct"]
    assert (a["n"], a["errors"]) == (4, 1)
    assert a["answer_rate"] == 1.0 and a["source_correct_rate"] == 0.5 and a["correct_refusal_rate"] == 1.0
    assert a["latency_p50_s"] == 2.0 and a["latency_p95_s"] == 3.0  # the errored 9s call is excluded


def test_format_table_has_one_row_per_mode():
    rows = [rc.score_response(LABELED, m, 200, body(tool_calls_used=1, steps=2, usage={}), 1.0) for m in ("direct", "agent")]
    table = rc.format_table(rc.aggregate(rows))
    assert table.count("\n") == 3 and "| direct |" in table and "| agent |" in table


# ---- the runner, end to end, in-process ---------------------------------------------------


async def test_runner_end_to_end_with_scripted_models(client, scripted_agent, monkeypatch, tmp_path):
    (tmp_path / "leave.txt").write_text("Employees get 20 days of annual leave per calendar year.")
    await rc.ensure_corpus(client, tmp_path)
    await rc.ensure_corpus(client, tmp_path)  # idempotent: the 409 on re-upload is fine

    async def direct(inputs):
        return AIMessage(content="20 days [1].")

    async def structured(inputs):
        return AIMessage(content=json.dumps({"answerable": True, "answer": "20 days", "cited_sources": [1]}))

    monkeypatch.setattr("app.langchain_port.chain._chain", FakeLangChainChain(direct))
    monkeypatch.setattr("app.langchain_port.structured._structured_chain", FakeLangChainChain(structured))
    scripted_agent([tool_turn("document_search", {"query": "leave"}), text_turn("20 days [1].")] * 4)

    questions = [LABELED, {**LABELED, "id": "f2"}]
    rows = await rc.run_compare(client, questions, ["direct", "structured", "agent"])
    assert len(rows) == 6 and not any(r["error"] for r in rows)
    agg = rc.aggregate(rows)
    assert set(agg) == {"direct", "structured", "agent"}
    assert all(agg[m]["source_correct_rate"] == 1.0 for m in agg)
    assert agg["agent"]["avg_tool_calls"] == 1.0 and agg["direct"]["avg_tool_calls"] == 0.0


# ---- Week 6 (Sat): tokens now scored for every mode ------------------------------------

def test_direct_and_structured_rows_now_carry_tokens_from_reported_usage():
    usage = {"input_tokens": 400, "output_tokens": 20, "model_calls": 1}
    d = rc.score_response(LABELED, "direct", 200, body(usage=usage), 1.0)
    s = rc.score_response(LABELED, "structured", 200, body(usage={**usage, "model_calls": 2}, attempts=2, answerable=True), 1.0)
    assert d["tokens"] == 420 and d["model_calls"] == 1
    assert s["tokens"] == 420 and s["model_calls"] == 2   # attempts stays the call count


def test_missing_usage_gives_no_tokens_rather_than_zero():
    r = rc.score_response(LABELED, "direct", 200, body(), 1.0)
    assert r["tokens"] is None


# ---- small-run helpers for rate-limited keys --------------------------------------------

def test_sample_per_type_takes_first_k_of_every_type_in_file_order():
    qs = [{"id": f"{t}{i}", "type": t} for t in ("factoid", "no_answer") for i in range(4)]
    out = rc.sample_per_type(qs, 2)
    assert [q["id"] for q in out] == ["factoid0", "factoid1", "no_answer0", "no_answer1"]


def test_real_eval_set_per_type_sample_includes_no_answer_questions():
    import json as _json
    from pathlib import Path as _P
    qs = _json.loads((_P(rc.__file__).parent / "eval_set.json").read_text())
    out = rc.sample_per_type(qs, 3)
    assert {q["type"] for q in out} == {"factoid", "paraphrase", "ambiguous", "no_answer"}
    assert len(out) == 12


async def test_run_stops_early_after_consecutive_errors_and_keeps_partial_rows():
    class AlwaysBad:
        calls = 0
        async def post(self, *a, **k):
            AlwaysBad.calls += 1
            import httpx
            return httpx.Response(502, json={"error": "rate limited"})

    qs = [{"id": f"q{i}", "type": "factoid", "question": "x", "gold_document": None} for i in range(10)]
    rows = await rc.run_compare(AlwaysBad(), qs, ["direct"], max_consecutive_errors=3)
    assert len(rows) == 3 and AlwaysBad.calls == 3          # stopped, did not hammer all 10
