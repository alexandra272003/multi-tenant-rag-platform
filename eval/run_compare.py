"""
Week 6 (Sat): direct RAG vs structured output vs agent, on the same 30 questions.

Runs from your machine against a RUNNING API (it uses the real endpoints, so it
measures what a client would see, including auth, retrieval, the LLM and the
network). Needs `httpx` (already in requirements.txt):

    python eval/run_compare.py --email you@example.com --password '...'
    python eval/run_compare.py --register-tenant "Eval Co" --email eval@x.test --password '...' --limit 10

It uploads eval/corpus/*.txt into YOUR tenant (a duplicate upload is fine),
sends every question in eval/eval_set.json to each mode, and reports per mode:

    answer rate            share of ANSWERABLE questions that were answered, not refused
    source-correct rate    answered AND cited the gold document   (the grounding signal)
    correct-refusal rate   share of NO-ANSWER questions that were refused  (the honesty signal)
    invalid-citation rate  answers citing a source number that was never retrieved
    latency p50 / p95      wall-clock seconds per question
    model calls / tool calls / tokens   cost proxies (agent: from the run's own accounting)

What these numbers can and cannot tell you (read before quoting one):
  * "source-correct" is a PROXY for answer quality: it checks the cited document,
    not whether the sentence is right. Paraphrased answers make exact-match scoring
    unreliable, so this avoids it -- at the cost of not catching a wrong claim that
    cites the right file. Spot-check a sample by hand.
  * 30 questions on a 3-document corpus is a smoke test, not a benchmark. Differences
    of a question or two are noise. Latency depends on your network and the provider.
  * Token counts come from provider-reported usage on every mode (the agent loop sums
    its steps; direct and structured report their own). A provider that omits usage
    shows '-' for tokens; compare model calls in that case.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REFUSAL = "i don't know based on the provided documents"

MODES = {
    "direct": ("/ask/langchain", lambda q: {"query": q, "top_k": 5}),
    "structured": ("/ask/structured", lambda q: {"query": q, "top_k": 5}),
    "agent": ("/agent", lambda q: {"query": q}),
}


def is_refusal(mode: str, body: dict) -> bool:
    if mode == "structured" and body.get("answerable") is False:
        return True
    return REFUSAL in (body.get("answer") or "").lower()


def score_response(question: dict, mode: str, status: int, body: dict, latency_s: float) -> dict:
    """Pure scoring (unit-tested, no network). One row per (question, mode)."""
    row = {"id": question["id"], "type": question["type"], "mode": mode, "status": status,
           "latency_s": latency_s, "error": status != 200}
    if status != 200:
        return row
    refused = is_refusal(mode, body)
    cited = {c["filename"] for c in body.get("citations", [])}
    gold = question.get("gold_document")
    row.update(
        refused=refused,
        has_gold=gold is not None,
        answered=not refused,
        source_correct=(gold in cited) if gold else None,
        correct_refusal=refused if gold is None else None,
        invalid_citation=body.get("all_citations_valid") is False,
    )
    usage = body.get("usage") or {}
    tokens = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None
    if mode == "agent":
        row.update(model_calls=usage.get("model_calls", body.get("steps")),
                   tool_calls=body.get("tool_calls_used", 0), tokens=tokens)
    elif mode == "structured":
        # `attempts` is the authoritative call count; tokens cover ALL attempts.
        row.update(model_calls=body.get("attempts"), tool_calls=0, tokens=tokens)
    else:
        row.update(model_calls=1 if body.get("retrieved_count") else 0, tool_calls=0, tokens=tokens)
    return row


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))]


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        ok = [r for r in rows if r["mode"] == mode and not r["error"]]
        labeled = [r for r in ok if r["has_gold"]]
        lat = [r["latency_s"] for r in ok]
        tokens = [r["tokens"] for r in ok if r.get("tokens")]
        out[mode] = {
            "n": sum(1 for r in rows if r["mode"] == mode),
            "errors": sum(1 for r in rows if r["mode"] == mode and r["error"]),
            "answer_rate": _rate(labeled, "answered"),
            "source_correct_rate": _rate(labeled, "source_correct"),
            "correct_refusal_rate": _rate([r for r in ok if not r["has_gold"]], "correct_refusal"),
            "invalid_citation_rate": _rate(ok, "invalid_citation"),
            "latency_p50_s": statistics.median(lat) if lat else 0.0,
            "latency_p95_s": _pct(lat, 95),
            "avg_model_calls": statistics.mean([r["model_calls"] for r in ok if r.get("model_calls") is not None] or [0]),
            "avg_tool_calls": statistics.mean([r["tool_calls"] for r in ok if r.get("tool_calls") is not None] or [0]),
            "avg_tokens": statistics.mean(tokens) if tokens else None,
        }
    return out


async def wait_until_ready(client: httpx.AsyncClient, document_id: int, name: str,
                           timeout_s: float = 180.0) -> None:
    """Uploads are asynchronous (202): questions asked before the worker has finished
    would find no chunks and 'refuse', which would look like a quality problem."""
    deadline = time.monotonic() + timeout_s
    while True:
        r = await client.get(f"/documents/{document_id}")
        doc = r.json() if r.status_code == 200 else {}
        if doc.get("status") == "ready":
            return
        if doc.get("status") == "failed":
            raise SystemExit(f"ingestion of {name} failed: {doc.get('error_message')}")
        if time.monotonic() > deadline:
            raise SystemExit(f"{name} was still '{doc.get('status')}' after {timeout_s:.0f}s -- is the worker running?")
        await asyncio.sleep(0.5)


async def ensure_corpus(client: httpx.AsyncClient, corpus_dir: Path) -> None:
    for path in sorted(corpus_dir.glob("*.txt")):
        r = await client.post("/documents", files={"file": (path.name, path.read_bytes())},
                              params={"chunk_size": 500, "chunk_overlap": 0})
        if r.status_code == 202:
            document_id = r.json()["id"]
        elif r.status_code == 409:  # already uploaded: fine, re-runs are safe (but wait for it)
            document_id = r.json()["error"]["details"]["existing_document_id"]
        else:
            raise SystemExit(f"could not upload {path.name}: {r.status_code} {r.text}")
        await wait_until_ready(client, document_id, path.name)


def sample_per_type(questions: list[dict], k: int) -> list[dict]:
    """First k questions of EACH type, in file order. A small run then still covers
    factoid, paraphrase, ambiguous AND no-answer questions (`--limit` takes the first
    N overall, which are all one type)."""
    seen: dict[str, int] = {}
    out = []
    for q in questions:
        if seen.get(q["type"], 0) < k:
            out.append(q)
            seen[q["type"]] = seen.get(q["type"], 0) + 1
    return out


async def run_compare(client: httpx.AsyncClient, questions: list[dict], modes: list[str],
                      max_consecutive_errors: int = 0) -> list[dict]:
    """max_consecutive_errors > 0: stop early after that many failures in a row (a
    rate-limited key would otherwise spend minutes retrying 30 questions that all
    fail). The rows gathered so far are still returned and reported."""
    rows = []
    streak = 0
    for q in questions:
        for mode in modes:
            path, build = MODES[mode]
            t0 = time.perf_counter()
            try:
                resp = await client.post(path, json=build(q["question"]), timeout=120)
                status, body = resp.status_code, (resp.json() if resp.content else {})
            except httpx.HTTPError as exc:
                status, body = 599, {"error": str(exc)}
            rows.append(score_response(q, mode, status, body, time.perf_counter() - t0))
            streak = streak + 1 if rows[-1]["error"] else 0
            if max_consecutive_errors and streak >= max_consecutive_errors:
                print(f"\nSTOPPING EARLY: {streak} failed requests in a row (last status {status}). "
                      "Usually a provider rate limit or a bad key -- the table below covers only "
                      "what finished and must not be quoted as a full run.")
                return rows
    return rows


def format_table(agg: dict) -> str:
    def f(v, pct=False, nd=2):
        if v is None:
            return "n/a"
        return f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"
    head = ("| mode | n | err | answer | source-correct | correct-refusal | bad-cite | p50 s | p95 s "
            "| model calls | tool calls | tokens |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for mode, a in agg.items():
        lines.append(
            f"| {mode} | {a['n']} | {a['errors']} | {f(a['answer_rate'], True)} | {f(a['source_correct_rate'], True)} "
            f"| {f(a['correct_refusal_rate'], True)} | {f(a['invalid_citation_rate'], True)} "
            f"| {f(a['latency_p50_s'])} | {f(a['latency_p95_s'])} | {f(a['avg_model_calls'], nd=1)} "
            f"| {f(a['avg_tool_calls'], nd=1)} | {f(a['avg_tokens'], nd=0)} |")
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--register-tenant", help="create a new tenant with this name first (instead of logging in)")
    p.add_argument("--modes", default="direct,structured,agent")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (cheap trial run)")
    p.add_argument("--per-type", type=int, default=0,
                   help="first K questions of EACH type (factoid/paraphrase/ambiguous/no_answer): a small "
                        "run that still includes no-answer questions. Use this on a rate-limited free key")
    p.add_argument("--stop-after-errors", type=int, default=3,
                   help="stop early after N failed requests in a row (0 = never stop)")
    args = p.parse_args()
    modes = [m.strip() for m in args.modes.split(",")]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        sys.exit(f"unknown mode(s): {unknown}; choose from {list(MODES)}")

    questions = json.loads((HERE / "eval_set.json").read_text())
    if args.per_type:
        questions = sample_per_type(questions, args.per_type)
    elif args.limit:
        questions = questions[: args.limit]

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        if args.register_tenant:
            r = await client.post("/auth/register", json={"tenant_name": args.register_tenant,
                                                         "email": args.email, "password": args.password})
        else:
            r = await client.post("/auth/login", json={"email": args.email, "password": args.password})
        if r.status_code not in (200, 201):
            sys.exit(f"auth failed: {r.status_code} {r.text}")
        client.headers["Authorization"] = "Bearer " + r.json()["access_token"]
        await ensure_corpus(client, HERE / "corpus")
        print(f"Running {len(questions)} questions x {modes} (sequential, so latency is comparable)...")
        rows = await run_compare(client, questions, modes, args.stop_after_errors)

    agg = aggregate(rows)
    print("\n" + format_table(agg))
    (HERE / "results_compare.json").write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))
    print(f"\nWrote {HERE / 'results_compare.json'}  (paste the table into docs/WEEK6_AGENT_VS_RAG.md)")


if __name__ == "__main__":
    asyncio.run(main())
