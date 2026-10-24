"""
Day 33: chunk-size x top-k experiment grid.

Calls Day 32's run_eval.run() once per (chunk_size, top_k) combination,
changing exactly those two variables and nothing else -- same eval set
(eval/eval_set.json), same embedding model, same prompt -- which is the
one rule Day 33 actually insists on: change one thing at a time (or run
the full grid) or you can't attribute a result to a cause.

Run inside the api container, same as run_eval.py:

    docker compose cp eval api:/code/eval
    docker compose exec api python eval/run_grid.py

Each document in the eval corpus gets ingested fresh at each new
chunk_size (Day 28's dedupe key is content hash + chunk_size +
chunk_overlap, so a different chunk_size is correctly treated as a new,
separate ingestion, not a duplicate -- this is the intended behavior, not
a workaround).

Overlap is fixed at 10% of chunk_size for every cell in the grid, so
overlap isn't a hidden third variable confounding the chunk_size
comparison.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_eval import run as run_eval_once  # noqa: E402

CHUNK_SIZES = [250, 500, 1000]
TOP_KS = [3, 5, 10]
RESULTS_PATH = Path(__file__).resolve().parent / "results_latest.json"
GRID_PATH = Path(__file__).resolve().parent / "grid_results.json"


async def main() -> None:
    grid = []
    for chunk_size in CHUNK_SIZES:
        chunk_overlap = max(10, chunk_size // 10)  # 10% overlap, held fixed across the grid
        for top_k in TOP_KS:
            print(f"\n{'#' * 78}")
            print(f"# chunk_size={chunk_size}  chunk_overlap={chunk_overlap}  top_k={top_k}")
            print("#" * 78)
            await run_eval_once(chunk_size, chunk_overlap, top_k)
            grid.append(json.loads(RESULTS_PATH.read_text()))

    GRID_PATH.write_text(json.dumps(grid, indent=2))

    print("\n\n" + "=" * 100)
    print("DAY 33 GRID SUMMARY -- same eval set, same model, same prompt throughout;")
    print("only chunk_size and top_k change between rows.")
    print("=" * 100)
    header = (
        f"{'chunk':>6} {'ovlp':>5} {'k':>3} {'chunks':>7} "
        f"{'Prec@k':>7} {'Recall@k':>9} {'MRR':>6} {'avg tok':>8} {'p95 ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in grid:
        o = r["overall"]
        print(
            f"{r['chunk_size']:>6} {r['chunk_overlap']:>5} {r['top_k']:>3} "
            f"{r['total_chunks_in_corpus']:>7} "
            f"{o['precision']:>7.3f} {o['recall']:>9.3f} {o['mrr']:>6.3f} "
            f"{o['avg_prompt_tokens']:>8.1f} {o['p95_retrieval_latency_ms']:>8.1f}"
        )
    print(f"\nWrote {GRID_PATH}")
    print(
        "\nReminder: Precision@k is mechanically capped at (total chunks)/k whenever "
        "the corpus has fewer chunks than k (see the per-run NOTE lines above) -- "
        "expect this at chunk_size=1000, where each short eval document becomes "
        "only 1-2 chunks. Read precision at that row as a corpus-size artifact, not "
        "a retrieval quality signal."
    )


if __name__ == "__main__":
    asyncio.run(main())
