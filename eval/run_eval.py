"""
Day 32: retrieval evaluation. Run this INSIDE the api container, where it
has access to the real Postgres + pgvector database and the local
embedding model:

    docker compose cp eval api:/code/eval
    docker compose exec api python eval/run_eval.py

What it does, in order:
  1. Ingests the 3 synthetic documents in eval/corpus/ (skips any already
     ingested with the same chunk settings -- safe to re-run).
  2. For every labeled question in eval/eval_set.json, runs retrieval
     scoped to just this eval corpus (so results aren't diluted by
     whatever else is sitting in the same database from manual testing).
  3. Checks whether the question's gold_span (an exact short phrase from
     the correct source document) appears in any of the top-k retrieved
     chunks, and prints Precision@k, Recall@k, and MRR -- per question
     type and overall.
  4. Reports no_answer questions separately (there is no "gold chunk" to
     check them against -- see the note in that section).

Metric definitions, and the simplification behind them (read this before
quoting a number in an interview): each labeled question has exactly ONE
relevant passage in the whole corpus (the sentence containing gold_span).
That's a deliberate simplification -- real corpora often have more than
one passage that could answer a question -- made so these metrics can be
computed by simple substring matching without a human re-labeling every
chunk. Under this single-relevant-item model:
  - Recall@k = 1 if ANY of the top-k chunks contains the gold_span, else 0.
  - Precision@k = (chunks among the top-k that contain the gold_span) / k.
    Overlap (Day 29) means more than one neighboring chunk can legitimately
    contain the same sentence, so this can be > 1/k.
  - MRR = 1 / (rank of the first chunk that contains the gold_span),
    0 if none of the top-k do.

Reusable for Day 33: every chunking/top-k setting this script touches is a
command-line flag, specifically so the Day 33 chunk-size x k grid can call
this same script repeatedly with different values instead of being
rewritten from scratch.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth.access import AccessFilter  # noqa: E402
from app.auth.rbac import Principal  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.core.errors import DuplicateDocumentError  # noqa: E402
from app.ingestion.chunking import count_tokens  # noqa: E402
from app.services import ingestion_service, retrieval_service  # noqa: E402
from app.services.generation_service import SYSTEM_PROMPT  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile -- no numpy dependency needed for 22 data points."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(pct / 100 * (len(s) - 1))))
    return s[k]

EVAL_TENANT_ID = 1  # the "default" tenant created by migration c3a7e1d92f10
EVAL_PRINCIPAL = Principal.system(EVAL_TENANT_ID)
EVAL_ACCESS = AccessFilter.from_principal(EVAL_PRINCIPAL)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


async def ingest_corpus(session, chunk_size: int, chunk_overlap: int) -> dict[str, int]:
    """
    Returns {filename: document_id} for the eval corpus, ingesting any
    file not already present with these exact chunk settings. Re-running
    this script with the same chunk_size/overlap is safe and idempotent
    (Day 28's duplicate detection does the work here); a DIFFERENT
    chunk_size (as Day 33's grid will use) ingests a fresh copy, exactly
    as the duplicate-detection design intended.
    """
    ids: dict[str, int] = {}
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        data = path.read_bytes()
        try:
            doc = await ingestion_service.ingest_document(
                session, EVAL_PRINCIPAL, path.name, data, chunk_size, chunk_overlap
            )
            ids[path.name] = doc.id
        except DuplicateDocumentError as exc:
            ids[path.name] = exc.details["existing_document_id"]
    return ids


def score_question(gold_span: str, retrieved_contents: list[str], top_k: int) -> dict:
    """
    Pure scoring logic, kept separate from the DB/embedding calls above so
    it can be unit-tested directly (see tests/test_eval_scoring.py) without
    needing Postgres, pgvector, or a real embedding model. Matching is
    case-insensitive substring containment, per the single-relevant-item
    simplification documented in this file's module docstring.
    """
    gold = gold_span.lower()
    hit_ranks = [
        rank for rank, content in enumerate(retrieved_contents, start=1)
        if gold in content.lower()
    ]
    recall = 1.0 if hit_ranks else 0.0
    precision = len(hit_ranks) / top_k if top_k else 0.0
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    return {"hit_ranks": hit_ranks, "recall": recall, "precision": precision, "mrr": mrr}


async def run(chunk_size: int, chunk_overlap: int, top_k: int) -> None:
    questions = json.loads(EVAL_SET_PATH.read_text())

    async with SessionLocal() as session:
        # Week 7: every query and ingest is tenant-scoped. The eval runs as the
        # system principal of the default tenant (the one the tenancy migration
        # created to adopt pre-existing data); the session hook applies it to
        # Postgres row-level security as well.
        session.info["tenant_id"] = EVAL_TENANT_ID
        doc_ids_by_name = await ingest_corpus(session, chunk_size, chunk_overlap)
        corpus_doc_ids = list(doc_ids_by_name.values())

        from sqlalchemy import func, select as sa_select
        from app.models import Chunk
        total_chunks = (
            await session.execute(
                sa_select(func.count()).select_from(Chunk).where(Chunk.document_id.in_(corpus_doc_ids))
            )
        ).scalar()

        print(f"Eval corpus ready: {doc_ids_by_name} ({total_chunks} chunks total)")
        print(f"Settings: chunk_size={chunk_size} chunk_overlap={chunk_overlap} top_k={top_k}")
        if total_chunks < top_k:
            print(
                f"NOTE: corpus has only {total_chunks} chunks total, fewer than top_k={top_k} -- "
                f"every Precision@{top_k} figure below is mechanically capped at {total_chunks}/{top_k}, "
                f"not a retrieval failure."
            )
        print()

        labeled = [q for q in questions if q["gold_span"] is not None]
        no_answer = [q for q in questions if q["gold_span"] is None]

        per_type_scores: dict[str, list[dict]] = {}
        all_scores: list[dict] = []
        latencies_ms: list[float] = []
        prompt_token_counts: list[int] = []
        system_prompt_tokens = count_tokens(SYSTEM_PROMPT)

        for q in labeled:
            t0 = time.perf_counter()
            results = await retrieval_service.retrieve(
                session, q["question"], access=EVAL_ACCESS, top_k=top_k, document_ids=corpus_doc_ids
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            contents = [chunk.content for chunk, _filename, _distance in results]
            s = score_question(q["gold_span"], contents, top_k)

            # Approximates the Day 31 prompt: system instructions + the
            # question + every retrieved chunk's own stored token_count
            # (captured once at ingestion, in app/ingestion/chunking.py).
            # This is what a bigger chunk_size or top_k actually costs.
            retrieved_tokens = sum(chunk.token_count for chunk, _f, _d in results)
            prompt_token_counts.append(
                system_prompt_tokens + count_tokens(q["question"]) + retrieved_tokens
            )

            score = {"id": q["id"], "type": q["type"], "recall": s["recall"],
                      "precision": s["precision"], "mrr": s["mrr"]}
            all_scores.append(score)
            per_type_scores.setdefault(q["type"], []).append(score)

            status = "HIT " if s["hit_ranks"] else "MISS"
            rank_display = s["hit_ranks"][0] if s["hit_ranks"] else "-"
            print(f"[{status}] {q['id']:4} ({q['type']:10}) rank={rank_display:<3}  {q['question']}")

        def _avg(scores, field):
            return sum(s[field] for s in scores) / len(scores) if scores else 0.0

        print("\n" + "=" * 72)
        print("RESULTS BY QUESTION TYPE")
        print("=" * 72)
        for qtype, scores in per_type_scores.items():
            print(
                f"  {qtype:12} n={len(scores):<3} "
                f"Precision@{top_k}={_avg(scores, 'precision'):.3f}  "
                f"Recall@{top_k}={_avg(scores, 'recall'):.3f}  "
                f"MRR={_avg(scores, 'mrr'):.3f}"
            )

        print("-" * 72)
        print(
            f"  {'OVERALL':12} n={len(all_scores):<3} "
            f"Precision@{top_k}={_avg(all_scores, 'precision'):.3f}  "
            f"Recall@{top_k}={_avg(all_scores, 'recall'):.3f}  "
            f"MRR={_avg(all_scores, 'mrr'):.3f}"
        )

        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        p95_latency_ms = _percentile(latencies_ms, 95)
        avg_prompt_tokens = (
            sum(prompt_token_counts) / len(prompt_token_counts) if prompt_token_counts else 0.0
        )
        print(
            f"\n  Cost/speed: avg_retrieval_latency={avg_latency_ms:.1f}ms  "
            f"p95_retrieval_latency={p95_latency_ms:.1f}ms  "
            f"avg_prompt_tokens={avg_prompt_tokens:.1f}  "
            f"(system_prompt alone = {system_prompt_tokens} tokens)"
        )

        if no_answer:
            print("\n" + "=" * 72)
            print(f"NO-ANSWER QUESTIONS ({len(no_answer)}) -- reported separately, not in the")
            print("averages above: there is no single correct chunk to check these against.")
            print("What to look at instead: does the TOP-1 distance come back noticeably")
            print("WORSE (larger) than the distances on the labeled hits above? If so, that")
            print("gap is a candidate similarity threshold for an abstention rule later.")
            print("=" * 72)
            for q in no_answer:
                results = await retrieval_service.retrieve(
                    session, q["question"], access=EVAL_ACCESS, top_k=1, document_ids=corpus_doc_ids
                )
                top1 = f"{results[0][2]:.3f}" if results else "n/a (no chunks at all)"
                print(f"  {q['id']:4} top-1 distance={top1:<8}  {q['question']}")

        out_path = Path(__file__).resolve().parent / "results_latest.json"
        out_path.write_text(json.dumps({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "total_chunks_in_corpus": total_chunks,
            "overall": {
                "precision": _avg(all_scores, "precision"),
                "recall": _avg(all_scores, "recall"),
                "mrr": _avg(all_scores, "mrr"),
                "n_labeled": len(all_scores),
                "n_no_answer": len(no_answer),
                "avg_retrieval_latency_ms": avg_latency_ms,
                "p95_retrieval_latency_ms": p95_latency_ms,
                "avg_prompt_tokens": avg_prompt_tokens,
            },
            "by_type": {
                qtype: {
                    "precision": _avg(scores, "precision"),
                    "recall": _avg(scores, "recall"),
                    "mrr": _avg(scores, "mrr"),
                    "n": len(scores),
                }
                for qtype, scores in per_type_scores.items()
            },
        }, indent=2))
        print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 32 retrieval evaluation")
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.chunk_size, args.chunk_overlap, args.top_k))


if __name__ == "__main__":
    main()
