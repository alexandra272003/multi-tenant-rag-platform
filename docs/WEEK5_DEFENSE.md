# Week 5 — Failure Analysis & Oral Defense

**Day 34 deliverable.** Everything in the failure table below actually happened
while building this project (Days 28-33) — nothing here is invented for the
exercise. That's deliberate: a real bug you found and fixed is a far stronger
interview story than a textbook example you've only read about.

---

## Part 1 — Failure analysis: symptom → cause → fix

| # | Symptom | Cause | Fix | Day found |
|---|---|---|---|---|
| 1 | `/retrieve` returned **zero results** against a real, populated Postgres database — no error, no exception, just an empty list. | `ivfflat`'s default search probes only **1** of the `lists = 100` clusters the Day 29 migration created. On a small corpus (far fewer rows than 100 clusters' worth), that one probed cluster can easily contain none of the real nearest vectors. | `SET LOCAL ivfflat.probes = 100` before the similarity query, scoped to the one transaction so it can't leak onto a pooled connection reused by another request. | 30 |
| 2 | The eval question `a1` ("probation length") retrieved its correct chunk at **rank 2**, not rank 1 — the only miss-from-the-top among 22 labeled questions. | A short, ambiguous, keyword-style query ("probation length") embeds less precisely than a full natural-language question, so its nearest neighbor in vector space isn't always the single best chunk. | None applied — this is normal behavior, not a bug, and is exactly the kind of result the eval set exists to surface. A real fix (if it mattered at this scale) would be query rewriting/expansion (Chapter 10 of the interview guide) before embedding the query. | 32 |
| 3 | No-answer question `n3` ("refund policy for the Enterprise plan") returned a **deceptively low** top-1 distance (0.249) — closer than you'd want for a question the corpus genuinely can't answer (there is no Enterprise plan, only Starter and Pro). | The question is topically adjacent to real content (refunds *are* covered, just not for a plan that doesn't exist), so embedding similarity alone can't distinguish "close topic" from "actually answerable." | Not yet applied — this is the concrete evidence that a naive fixed similarity threshold for abstention would misfire on this exact question. A real fix needs either a higher threshold tuned against more no-answer examples, or an explicit "does this chunk actually name the thing asked about" check, which is a generation-layer problem, not a retrieval-layer one. | 32 |
| 4 | Every retrieval-side test was silently making a real call to the embedding model (and, in a network-restricted environment, failing outright) even though a `mock_embeddings` fixture was supposedly active. | The autouse test fixture patched `embed_texts` only at its **ingestion** call site (`app.services.ingestion_service.embed_texts`). Retrieval imports the same function separately (`app.services.retrieval_service.embed_texts`), and patching one import binding doesn't patch the other. | Extended the fixture to patch both call sites. Side effect: the full 66-test suite dropped from minutes (real model inference) to under a second. | 32 (caught while verifying Day 32's own work) |
| 5 | A chunk_size=1000 run in the Day 33 grid shows Precision@10 that looks poor at a glance. | The eval corpus is 3 short synthetic documents. At `chunk_size=1000`, each document collapses into only 1-2 chunks — fewer than `top_k=10` can even return, so Precision@10 is mechanically capped at (total chunks)/10 regardless of retrieval quality. | Not a bug to fix — a reporting discipline to maintain. `run_eval.py` prints an explicit `NOTE` whenever total corpus chunks < top_k, so the number is never read without its denominator context. | 33 |
| 6 | (From earlier in this sprint, a different project, included here because the lesson generalizes) An LLM provider call failed with `401 Incorrect API key` despite a correct-looking key in `.env`. | `docker-compose.yml` only passed `DATABASE_URL` into the container's environment — the other `.env` variables were never wired through via `${VAR}` substitution, so the container always saw the hardcoded placeholder default regardless of what `.env` said. | Added explicit `KEY: ${KEY}` lines to the `environment:` block for every setting the app actually needs at runtime. **Lesson applied here:** this project's `docker-compose.yml` was written with that lesson already in mind from Day 31 onward. | Week 4 (chatbot project), reapplied as a design habit in this one |

### What these six have in common

Four of the six (#1, #3, #4, #6) share the same shape: **the system ran successfully and produced a plausible-looking result that was actually wrong or misleading, with no error to flag it.** That's a harder class of bug than a crash, because nothing tells you to go looking. The concrete defense against it, demonstrated across this project: automated tests that assert on *content*, not just status codes (#4 was only caught because a test suite existed to run at all); an eval set with labeled no-answer questions specifically to catch #3-shaped problems; and treating "it returned 200 with an empty/wrong-looking result" as seriously as a 500 during manual testing (#1 was caught that way, not by a test).

### Gap, named honestly: conflicting documents

The sprint's Day 34 scope also asks about conflicting-document failures (two sources disagreeing). **This project's eval corpus doesn't include a conflicting-document case** — the three documents are deliberately topic-disjoint, which was the right choice for testing retrieval precision cleanly, but means this specific failure mode was never exercised. The honest answer for an interview: the fix would be a document-level `version`/`date` metadata column (not yet in the schema), preferring the newest or most authoritative source, and surfacing both with citations when genuinely unresolved — described but not built, which is a legitimate thing to say plainly rather than imply it was tested.

---

## Part 2 — RAG vs. fine-tuning, grounded in what was actually built

This project is a clean real-world case for the standard comparison:

- **Facts changed by re-ingestion, not retraining.** Uploading a new or edited document (Day 28) takes effect immediately on the next query — no training run, no downtime.
- **Every claim is citable.** Day 31's numbered-context prompt plus citation verification means an answer can point to exactly which chunk it came from, something a fine-tuned model's weights fundamentally cannot do.
- **Deletion works cleanly.** Removing a document removes its chunks via cascade delete (Day 28's schema) and it's gone from every future answer — fine-tuning has no equivalent "unlearn this" operation.
- **What RAG here does *not* do:** change the model's tone, style, or general reasoning ability. If this project needed a consistently different voice or output format, that's a fine-tuning or system-prompt problem, not something more retrieval solves.

---

## Part 3 — The 10-minute oral defense

Five segments, two minutes each, matching the sprint's own required format.

### Architecture (2 min)

> "This is RAG built from scratch, no framework. Two pipelines. Indexing, offline, once per document: parse bytes into pages, clean the text — Unicode normalization, hyphenation repair, line re-flow — then pack it into overlapping, sentence-aligned chunks, embed each chunk locally with fastembed, and store the vector in a real pgvector column with an ivfflat cosine index. Querying, online, every request: embed the question with that same local model, search the index for the nearest chunks, optionally filtered by document or source type, build a prompt that numbers and labels each chunk by source, generate an answer, and verify every citation the model produced actually points to a chunk that was really retrieved. Day 32 adds a frozen 30-question eval set with Precision@k, Recall@k and MRR; Day 33 runs that same eval across a 3×3 chunk-size by top-k grid to replace opinion with a measured table."

### Schema (2 min)

> "Two tables. `documents` holds one row per upload — filename, source type, a SHA-256 of the content, and the chunk settings it was ingested with. `chunks` holds one row per piece, with a foreign key back to its document, its position, page number where applicable, character offsets into the cleaned text, a token count, and — since Day 29 — a `vector(384)` embedding column. The unique constraint on `(content_sha256, chunk_size, chunk_overlap)` is the duplicate-detection key: the same file at the same chunk settings is rejected as a 409, but the same file at a *different* chunk size is correctly allowed as a fresh ingestion — which is exactly what Day 33's grid experiment relies on."

### Scaling limit (2 min)

> "Two honest ones. First: parsing, cleaning, and chunking are synchronous, CPU-bound Python running directly inside an async request handler — one very large document would stall the event loop for every other concurrent request. The fix, in increasing effort, is `asyncio.to_thread` as a two-line change, then a queue-and-worker model with a status-polling endpoint for real scale. Second: the eval corpus itself is intentionally tiny — three short documents — which is correct for testing retrieval *correctness* cheaply and deterministically, but Day 30's `ivfflat.probes` bug is a direct, lived example of a retrieval index that behaves differently at real scale than it does on a handful of test rows. Both limits are named, not hidden."

### Failure mode (2 min)

> "The ivfflat zero-results bug, item 1 in the failure table. Index built for 100 clusters, small table, default search checks only one cluster, query runs successfully and returns nothing — no exception anywhere in the stack to flag it. Found by manually checking an empty result against a database I'd just confirmed had real rows in it, not by a test, which is itself the lesson: a result that's wrong-but-plausible needs a human checking content, not just an automated check for a 200 status code. Fixed with `SET LOCAL ivfflat.probes = 100`, which is a real speed-for-recall trade-off, not a free fix — the right probe count for a real corpus depends on its actual size and would get tuned the same way chunk size is tuned, by measuring, not guessing."

### One trade-off (2 min)

> "Local embeddings over a hosted API. No key, no billing, no network dependency at request time, no provider outage to handle — at the cost of a smaller, somewhat lower-quality model than a large hosted one, and a one-time model download on first use. This decision itself came from a failed first attempt: Groq was the original plan for embeddings, discovered mid-build to have no embeddings endpoint at all despite some third-party client libraries implying otherwise — chat, audio, and TTS only. Switching to local embeddings for the *embedding* step while keeping Groq for the *generation* step (Day 31) is the actual, working split in this project today."

---

## Quick numbers to have ready

From the real Day 32 run against actual Postgres + pgvector (not the mocked sandbox runs):

- **Recall@5 = 1.000** across all 22 labeled questions (factoid, paraphrase, ambiguous) — every gold passage was found somewhere in the top 5.
- **MRR = 0.977 overall** — all but one question's correct chunk landed at rank 1; the one exception (`a1`) landed at rank 2.
- **Precision@5 ≈ 0.2** — expected and explainable, not a quality signal: with exactly one relevant chunk per question and k=5, 1/5 is the ceiling.
- **8 no-answer questions**, correctly excluded from the averages above and reported separately by top-1 distance instead.

---

## Carried into Week 6

The sprint's own note for today: *"merge learnings into the Week 6 LangChain port."* Concretely, from this week:

- The numbered-citation-with-verification prompt pattern (Day 31) is worth porting as-is — it's framework-agnostic.
- The eval harness (Day 32-33) should keep running against whatever Week 6 builds, unchanged, as the regression check that the LangChain port didn't quietly make retrieval worse.
- The `ivfflat.probes` lesson applies to any vector-index choice Week 6 makes, LangChain or not.
- The "Groq has no embeddings endpoint" fact is now known cold and won't cost debugging time again.
