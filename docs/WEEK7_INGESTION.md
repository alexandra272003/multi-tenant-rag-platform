# Week 7 Thursday — Async ingestion pipeline

## Contract
`POST /documents` validates and stores the upload, then returns **202 Accepted** with `Location: /documents/{id}`
and `status: "uploaded"`. A Celery worker parses, chunks, embeds and publishes. Clients poll `GET /documents/{id}`.
Failed documents can be retried with `POST /documents/{id}/retry`.

```
uploaded ──claim──> processing ──publish──> ready
   ^                    │  ├─ transient error (embedding) ─> uploaded (retry w/ backoff)
   │                    │  └─ permanent error (parse/chunk) ─> failed
   └──── retry ──── failed
```

## Decisions
| Problem | Decision |
|---|---|
| Queue delivers at-least-once | `process_document` is idempotent: ready → `already_ready`, live lease → `busy` |
| Two workers, one document | Atomic claim: conditional `UPDATE … RETURNING attempts` |
| Worker stalls, another takes over | Lease (`INGEST_LEASE_SECONDS`=300); `attempts` is a **fencing token**; the publish checks it and a stale worker gets `superseded` |
| Partial results | Chunks + status + clearing staged bytes + audit commit in **one transaction** |
| Broker down at upload (dual write) | Upload still returns 202; document stays `uploaded`; the beat **sweeper** re-enqueues |
| Lost message / dead worker | Sweeper re-enqueues `processing` past lease and `uploaded` too long |
| Retry only when it can help | Parse/chunk errors are deterministic → `failed`. Embedding errors → release + exponential backoff, then `failed` |
| Document deleted mid-job | Claim/publish are scoped to live rows; result discarded (`gone`/`superseded`) |
| Raw bytes | Staged in `document_sources` (RLS), nulled when ready, kept while failed so retry works |
| Tenant isolation in the worker | Worker imports `app.core.tenancy` → same `SET LOCAL ROLE rag_app` + tenant as the API |
| Event loops | `asyncio.run` per task ⇒ NullPool engine per task |

## Known limits
- Postgres, real Redis and real Celery paths are verified by `scripts/verify_postgres.py` and the compose stack, not by the SQLite unit suite (which uses an in-process job runner).
- Staged bytes live in the DB: fine for MB-sized files, use object storage beyond that.
- No per-tenant fairness yet (one tenant can fill the queue) — Saturday's quotas.
- Scanned PDFs fail permanently (no OCR).

## Interview answers
- *Why 202?* Work exceeds a request's budget and has failure modes the client shouldn't hold a connection for.
- *Exactly-once?* Not achievable; at-least-once + idempotent + fenced publish gives effectively-once.
- *Why fencing if you have a lease?* A paused worker can outlive its lease; the token stops its late write.
