# multi-tenant-rag-platform

A multi-tenant document and RAG (retrieval-augmented generation) platform. Organisations upload documents, a
background worker ingests them, and users ask grounded, cited questions in chat. Tenant isolation is enforced in
three independent layers, including Postgres row-level security.

Built as a backend + GenAI engineering capstone. The focus is the parts that make a RAG system trustworthy to
run for several customers at once: isolation, authorization, background jobs that survive failure, quotas, and
observability. The model call is the easy part.

**Stack:** Python 3.12 · FastAPI · PostgreSQL 16 + pgvector · SQLAlchemy 2 (async) · Alembic · Celery + Redis ·
fastembed (local embeddings) · OpenAI-compatible LLM (Groq by default) · LangChain · Prometheus · Docker Compose

---

## Contents
1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Using the API](#using-the-api)
5. [Design decisions](#design-decisions)
6. [Verification](#verification)
7. [Configuration](#configuration)
8. [Project layout](#project-layout)
9. [Known limitations](#known-limitations)
10. [Documentation](#documentation)

---

## What it does

| Area | Capability |
|---|---|
| **Multi-tenancy** | Self-service signup creates a tenant and its owner. Every row carries `tenant_id`. Another tenant's ids return 404, never 403, so they don't confirm an id exists. |
| **Auth and RBAC** | JWT login. Roles: owner, admin, editor, viewer. Role checks and object-level checks are separate (an editor may edit only their own documents). The caller is rebuilt from the database on every request, so role changes and deactivation apply immediately. |
| **Async ingestion** | `POST /documents` returns `202 Accepted`. A Celery worker parses, chunks, embeds and publishes. Status machine: `uploaded → processing → ready \| failed`. |
| **Retrieval** | pgvector cosine search, always filtered by tenant and visibility before ranking. Private documents are visible only to their owner and to admins. |
| **Grounded Q&A** | Numbered context, cited answers, and every citation is checked against what was actually retrieved. With no evidence the model is not called. |
| **Conversational RAG** | Private chat threads. Each follow-up is rewritten into a standalone question, retrieved fresh under the caller's current access, and answered with recent history. |
| **Agent** | Tool-calling agent with document search and a guarded, read-only SQL tool confined to the caller's tenant (see the agent threat model). |
| **Structured output** | Schema-validated JSON answers with validation retries (LangChain). |
| **Quotas and rate limits** | Daily per-tenant quotas (atomic, persisted) plus per-minute rate limits (memory or Redis). |
| **Observability** | Prometheus `/metrics`, JSON logs carrying request, tenant and user ids, and an `X-Request-ID` header on every response. |
| **Data safety** | Expand-only migrations, soft delete, optimistic locking, and an append-only audit log written in the same transaction as the change. |

---

## Architecture

```mermaid
flowchart LR
    C[Client] -->|JWT| API[FastAPI API]
    API -->|"RBAC → rate limit → quota"| PG[("PostgreSQL + pgvector<br/>row-level security")]
    API -->|enqueue job| R[(Redis)]
    R --> W["Celery worker<br/>+ beat sweeper"]
    W -->|"claim · publish chunks"| PG
    W --> E["fastembed<br/>(local embeddings)"]
    API --> E
    API -->|chat completions| L["LLM provider<br/>(OpenAI-compatible)"]
    API -->|"/metrics"| P[Prometheus]
```

### Document lifecycle

```mermaid
stateDiagram-v2
    [*] --> uploaded: POST /documents (202)
    uploaded --> processing: worker claims (attempts += 1)
    processing --> ready: chunks + status + audit commit atomically
    processing --> uploaded: transient error (embedding provider), retry with backoff
    processing --> failed: permanent error (corrupt file, no text)
    failed --> uploaded: POST /documents/{id}/retry
    processing --> processing: lease expired, another worker takes over
```

### One conversational turn

```
user message
   ├─ load last N messages of THIS user's conversation       (history window)
   ├─ condense follow-up into a standalone question          (falls back to the raw query on any problem)
   ├─ retrieve under the caller's CURRENT access filter      (never reuses earlier turns' chunks)
   ├─ generate with history + freshly numbered context       ([n] markers stripped from replayed history)
   ├─ verify every citation against what was retrieved
   └─ save question + answer together, only after the answer exists
```

### Tenant isolation, three layers

| Layer | Mechanism |
|---|---|
| API | The caller is built from the database. Every read goes through one `AccessFilter` (tenant, soft delete, visibility). |
| Query | `retrieve()` takes a required `access` argument, so a caller that forgets to scope cannot call it. |
| Database | Postgres `FORCE ROW LEVEL SECURITY` with a fail-closed policy. A hook sets `SET LOCAL ROLE rag_app` and the tenant on **every transaction**. With no tenant set, the query returns zero rows. |

---

## Quick start

**Requirements:** Docker Desktop (with WSL 2 on Windows) and a free [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/<your-username>/multi-tenant-rag-platform.git
cd multi-tenant-rag-platform
cp .env.example .env        # then put your key in LLM_API_KEY
docker compose up --build
```

On Windows PowerShell, create `.env` like this instead of `cp`:

```powershell
@"
LLM_API_KEY=your_groq_key
LLM_MODEL=openai/gpt-oss-120b
LLM_STRUCTURED_METHOD=json_schema
"@ | Set-Content -Encoding ascii .env
```

This starts four services:

| Service | Role |
|---|---|
| `postgres` | PostgreSQL 16 with pgvector |
| `redis` | Job broker and shared rate-limit counters |
| `api` | FastAPI on http://localhost:8000. Migrations run on startup. |
| `worker` | Celery worker with the stuck-job sweeper (beat) |

Interactive API docs: http://localhost:8000/docs

The first document you upload takes about 20 seconds because the worker downloads the embedding model once.
The model is cached in a Docker volume after that.

---

## Using the API

```bash
# 1. Sign up (creates a tenant and its owner) and keep the token
curl -s -X POST localhost:8000/auth/register -H 'content-type: application/json' \
  -d '{"tenant_name":"Acme","email":"me@acme.test","password":"password-123"}'

# 2. Upload a document: returns 202 with status "uploaded"
curl -s -X POST "localhost:8000/documents?chunk_size=200&chunk_overlap=20" \
  -H "Authorization: Bearer $TOKEN" -F "file=@notes.txt"

# 3. Poll until status is "ready" (or "failed", then POST /documents/{id}/retry)
curl -s localhost:8000/documents/1 -H "Authorization: Bearer $TOKEN"

# 4. Ask a grounded question
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"Who approved Project Atlas?"}'

# 5. Or hold a conversation
curl -s -X POST localhost:8000/conversations -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{}'
curl -s -X POST localhost:8000/conversations/1/messages -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"and what is its budget?"}'
```

### Endpoints

| Method and path | Purpose | Minimum role |
|---|---|---|
| `POST /auth/register` · `POST /auth/login` · `GET /me` | Signup, login, current user | public / any |
| `GET/POST /users` · `PATCH /users/{id}` | Manage users in your tenant | admin |
| `POST /documents` | Upload (202, async ingestion) | editor |
| `GET /documents` · `GET /documents/{id}` · `GET /documents/{id}/chunks` | List, status, chunks | viewer |
| `PATCH /documents/{id}` · `DELETE /documents/{id}` | Edit (optimistic lock) · soft delete | owner of doc, or admin |
| `POST /documents/{id}/retry` | Re-run a failed ingestion | owner of doc, or admin |
| `POST /retrieve` | Vector search, no LLM | viewer |
| `POST /ask` · `/ask/langchain` · `/ask/structured` | Grounded Q&A (plain, LangChain, schema-validated) | viewer |
| `POST /agent` · `/agent/stream` | Tool-calling agent | viewer |
| `POST /conversations` · `GET /conversations[/{id}]` · `DELETE /conversations/{id}` | Your chat threads (private to you) | viewer |
| `POST /conversations/{id}/messages` | One conversational turn | viewer |
| `GET /usage` | Today's quota use and limits for your tenant | admin |
| `GET /audit` | Your tenant's audit trail | admin |
| `GET /metrics` · `GET /ping` | Prometheus metrics · liveness | public (metrics optionally token-protected) |

### Roles

| | viewer | editor | admin | owner |
|---|:-:|:-:|:-:|:-:|
| Read documents, query, chat | ✓ | ✓ | ✓ | ✓ |
| Upload documents, edit and delete own | | ✓ | ✓ | ✓ |
| Edit and delete any document | | | ✓ | ✓ |
| Manage users, read audit log and usage | | | ✓ | ✓ |
| Manage admins and owners | | | | ✓ |

Rejected requests return `429` with `Retry-After`. The error code is `rate_limited` for a burst and
`quota_exceeded` for a spent daily budget, so a client can tell "wait a few seconds" from "come back tomorrow".

---

## Design decisions

The reasoning is in the docs. The short version:

- **Why 404 and not 403 for another tenant's id?** A 403 confirms the id exists. Both cases return the same response.
- **Why is upload `202`?** Ingestion can outlast a request and can fail in ways the client shouldn't hold a
  connection for. The document is a status resource the client polls.
- **Why is the worker safe under at-least-once delivery?** The handler is idempotent. A claim is one atomic
  conditional `UPDATE`, with a lease. `attempts` is a fencing token, so a stalled worker that wakes after
  another took over cannot publish. Chunks, status, deleting the staged bytes and the audit row commit in one
  transaction.
- **What if Redis is down at upload time?** The document is already durable as `uploaded`. A scheduled sweeper
  re-enqueues anything stuck. (A transactional outbox would be the next step.)
- **Why is the quota a single conditional `UPDATE … WHERE value < limit RETURNING`?** Read-then-write over-admits
  under concurrency. Fifteen simultaneous requests with 5 left admit exactly 5, tested on real Postgres.
- **Why does a follow-up not reuse the previous turn's chunks?** Access can change between turns. A document
  made private or deleted since then must stop being quotable.
- **Why no tenant label on Prometheus metrics?** Unbounded cardinality. Per-tenant numbers come from `/usage`
  and the structured logs.
- **How would this scale beyond one Postgres?** Read replicas first, then sharding by tenant. The design memo
  covers the order, the hot spots and the failure scenarios.

---

## Verification

Three layers, each answering a different question:

| Check | Command | What it proves |
|---|---|---|
| **468 unit and API tests** | `pytest` | Logic, isolation, RBAC, state machine, quotas, logs. Run on SQLite with a mocked LLM and embeddings, so no services are needed. |
| **57 Postgres checks** | `docker compose exec api python scripts/verify_postgres.py --admin-url postgresql+asyncpg://rag_user:rag_pass@postgres:5432/postgres` | What SQLite can't show: real row-level security, fail-closed behaviour, migrations up/down/up with no model drift, real concurrency (simultaneous edits, workers, quota). |
| **31 end-to-end checks** | `docker compose exec api python scripts/e2e_smoke.py` | The whole running stack through its public API: real Redis, real worker, real LLM. |

The Postgres script creates and drops its own throwaway database. Run it after any schema change.

---

## Configuration

Copy `.env.example` to `.env`. The main settings (all environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | none (required) | Key for the OpenAI-compatible provider |
| `LLM_BASE_URL` / `LLM_MODEL` | Groq / `llama-3.1-8b-instant` | Provider endpoint and chat model |
| `JWT_SECRET` | dev value | **Change for anything real.** With `ENVIRONMENT` other than `development` the app refuses to start on the default. |
| `DAILY_QUERY_LIMIT` / `DAILY_UPLOAD_LIMIT` | 500 / 100 | Per-tenant daily quotas. A tenant row can override them. |
| `RATE_LIMIT_PER_MINUTE` | 60 | Per-tenant burst limit on LLM endpoints |
| `RATE_LIMIT_BACKEND` | `memory` (`redis` in compose) | `memory`, `redis` or `off`. Redis failures fail open. |
| `INGEST_MAX_RETRIES` / `INGEST_LEASE_SECONDS` | 3 / 300 | Worker retries and claim lease |
| `LOG_FORMAT` | `text` (`json` in compose) | Structured JSON logs |
| `METRICS_TOKEN` | unset | If set, `/metrics` needs `Authorization: Bearer <token>` |

---

## Project layout

```
app/
  auth/            JWT dependency, RBAC table, the single AccessFilter
  core/            config, db, tenancy (RLS hook), embeddings, LLM client, rate limiter, metrics, logging
  ingestion/       parsers, cleaning, chunking
  jobs/            Celery app, tasks, enqueue (dispatch)
  models/          SQLAlchemy models
  repositories/    document and job queries (atomic claim, fenced publish)
  routers/         FastAPI routes
  services/        ingestion, retrieval, generation, chat, quota, auth, audit
  agent/           tool-calling loop, SQL guard, tenant scope, injection defences
  langchain_port/  LangChain chain, retriever, structured output
alembic/versions/  expand-only migrations (tenancy → RLS → async ingestion → conversations → quotas)
scripts/           verify_postgres.py, e2e_smoke.py
tests/             468 tests
eval/              retrieval and answer-quality evaluation harness
docs/              design notes and memos
```

---

## Known limitations

Stated plainly, because a portfolio project should know its own edges:

- `users.email` is globally unique, so one person can't belong to two tenants without a memberships table. It also
  complicates sharding.
- The `queries` quota counts requests, not tokens.
- No login brute-force throttling, and rate limits are not configurable per tenant (quotas are).
- Metrics are per process. The worker's counters are not on the API's `/metrics`.
- Rate limiting uses fixed windows, so a client can send up to 2× the limit across a window boundary.
- Scanned PDFs fail (no OCR). Staged upload bytes live in Postgres, which is fine for MB-sized files, not for large ones.
- The quota counter is one hot row per tenant, metric and day. Shard it at much higher request rates.
- The condense step and answer quality depend on the model you configure. They were tested with mocks and a
  live smoke run, not measured on a benchmark.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/WEEK7_SUMMARY.md`](docs/WEEK7_SUMMARY.md) | Start here: what was built, what is verified, and interview questions |
| [`docs/WEEK7_TENANCY.md`](docs/WEEK7_TENANCY.md) | Tenant model, RLS wiring, RBAC, migrations, audit, soft delete, versioning |
| [`docs/WEEK7_INGESTION.md`](docs/WEEK7_INGESTION.md) | Async ingestion design, status machine, failure modes |
| [`docs/WEEK7_OPERATIONS.md`](docs/WEEK7_OPERATIONS.md) | Quotas, rate limits, metrics, logs, suggested alerts |
| [`docs/WEEK7_DISTRIBUTED_DB_MEMO.md`](docs/WEEK7_DISTRIBUTED_DB_MEMO.md) | How to scale beyond one Postgres: replicas, sharding, consistency, failure scenarios |
| [`docs/AGENT_THREAT_MODEL.md`](docs/AGENT_THREAT_MODEL.md) | Prompt-injection and SQL-tool threat model for the agent |
| [`docs/WEEK6_AGENT_VS_RAG.md`](docs/WEEK6_AGENT_VS_RAG.md) | Agent versus plain RAG comparison |
| [`docs/README_WEEK6.md`](docs/README_WEEK6.md) | The earlier single-tenant version's README |
