"""
Tests the pure scoring logic in eval/run_eval.py -- no database, no
embedding model, no Docker needed. This is what stands in for actually
running the eval script in CI, since the script itself needs a real
Postgres + pgvector + a downloaded embedding model to run for real.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_eval import score_question  # noqa: E402


def test_hit_at_rank_one_gives_perfect_mrr():
    result = score_question(
        gold_span="Pro plan costs 49 dollars",
        retrieved_contents=["The Pro plan costs 49 dollars per month.", "unrelated text"],
        top_k=5,
    )
    assert result["recall"] == 1.0
    assert result["mrr"] == 1.0
    assert result["hit_ranks"] == [1]


def test_hit_at_rank_three_gives_mrr_one_third():
    result = score_question(
        gold_span="100 degrees Celsius",
        retrieved_contents=["irrelevant", "also irrelevant", "Water boils at 100 degrees Celsius."],
        top_k=5,
    )
    assert result["recall"] == 1.0
    assert result["mrr"] == 1.0 / 3
    assert result["hit_ranks"] == [3]


def test_no_hit_scores_zero_everywhere():
    result = score_question(
        gold_span="this phrase never appears",
        retrieved_contents=["something else entirely", "and this too"],
        top_k=5,
    )
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
    assert result["mrr"] == 0.0
    assert result["hit_ranks"] == []


def test_matching_is_case_insensitive():
    result = score_question(
        gold_span="Pro Plan Costs",
        retrieved_contents=["the pro plan costs very little"],
        top_k=5,
    )
    assert result["recall"] == 1.0


def test_precision_counts_every_matching_chunk_not_just_the_first():
    """
    Overlap (Day 29) means a gold sentence can legitimately land in more
    than one neighboring chunk -- precision should reflect that, not cap
    at a single hit.
    """
    result = score_question(
        gold_span="100 degrees Celsius",
        retrieved_contents=[
            "Water boils at 100 degrees Celsius at sea level.",
            "...100 degrees Celsius, which is 212 Fahrenheit...",
            "completely unrelated chunk",
        ],
        top_k=3,
    )
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0


def test_empty_eval_set_file_is_valid_json_with_expected_shape():
    import json

    data = json.loads((Path(__file__).resolve().parent.parent / "eval" / "eval_set.json").read_text())
    assert len(data) == 30
    for q in data:
        assert set(q.keys()) == {"id", "type", "question", "gold_document", "gold_span"}
        if q["type"] == "no_answer":
            assert q["gold_span"] is None
        else:
            assert isinstance(q["gold_span"], str) and len(q["gold_span"]) > 0
