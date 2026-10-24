# Week 7 — Capstone I: multi-tenant document + RAG platform (summary)

## What exists, by day
| Day | Deliverable | Where |
|---|---|---|
| Mon | Tenant model, JWT auth, tenant context on every request, Postgres RLS (fail-closed) | `app/auth`, `app/core/tenancy.py`, `WEEK7_TENANCY.md` |
| Tue | RBAC (owner/admin/editor/viewer) + object-level checks, role-assignment rules | `app/auth/rbac.py` |
| Wed | Migrations (expand-only), indexes, audit log, soft delete, optimistic-lock versioning | `alembic/versions`, `document_repository.py` |
| Thu | **Async ingestion**: 202 + status machine, Celery/Redis worker, idempotent + fenced, retries, sweeper | `app/jobs`, `ingestion_service.py`, `WEEK7_INGESTION.md` |
| Fri | **Conversational RAG**: private per-user conversations, condense → retrieve → answer, history window | `chat_service.py`, `routers/conversations.py` |
| Sat | **Quotas, rate limits, metrics, structured logs** | `quota_service.py`, `core/ratelimit.py`, `core/metrics.py`, `WEEK7_OPERATIONS.md` |
| Sun | **Distributed-DB design memo** + this write-up | `WEEK7_DISTRIBUTED_DB_MEMO.md` |

## Friday in one page
* `POST /conversations`, `GET /conversations[/{id}]`, `DELETE`, `POST /conversations/{id}/messages`.
* **Private to the author** — a 404 for everyone else, owners and admins included (tenant isolation says *which org*, `user_id`
  says *which person*). Tables have RLS like the rest.
* **A turn:** (1) *condense* the follow-up into a standalone question using the recent history (best effort: any failure or a
  malformed rewrite falls back to the user's own words); (2) *retrieve* with the standalone question under the caller's access **at
  this moment**; (3) *generate* with history as conversation turns and fresh numbered context.
* **Decisions that matter:** earlier turns' chunks are never reused (a document made private or deleted since must stop being
  quotable); citation markers `[n]` are stripped from replayed answers (they pointed at a different context last turn); the system
  prompt says numbers refer only to the current context; the user message and the answer are saved **together, after** the model
  answered (a provider failure leaves no dangling question); stored citations are a snapshot, not a join; history is bounded by
  message count and characters; a conversation is capped (409 when full).
* **Not done:** streaming conversation responses; summarising old history instead of dropping it; message edit/regenerate.

## Verification — what is and is not proven
| | Evidence |
|---|---|
| 468 automated tests pass | SQLite, in-process job runner, mocked LLM and embeddings |
| Tenant isolation, RBAC, quotas, rate limits, conversation privacy, ingestion state machine, fencing, sweeper, metrics, logs | covered by those tests |
| Row-level security, sequence grants, the Postgres upsert path of the quota, real concurrency | `scripts/verify_postgres.py` — **written, not run in my sandbox (no Postgres there); run it** |
| Real Redis broker, real Celery worker, Redis limiter | compose stack — **not run in my sandbox**; the Celery wrapper is tested with `.apply()`, the Redis limiter only for fail-open |
| Real LLM behaviour of the condense step | mocked; quality of rewrites with your model is unmeasured |

Run it: `docker compose up --build`, then `docker compose exec api python scripts/verify_postgres.py` (it creates and drops its own database).

## Honest limits
* Metrics are per process (the worker's `ingest_jobs_total` is not on the API's `/metrics`).
* `users.email` global uniqueness blocks multi-tenant users and complicates sharding (memo §4).
* `queries` quota counts requests, not tokens.
* No login throttling; no tenant-configurable rate limits.
* One hot counter row per tenant per metric per day.

## Interview questions this week prepares you for (with the short answer)
1. **How do you isolate tenants?** Three independent layers — API filter, required retrieval scope, Postgres RLS fail-closed.
2. **Why 404 not 403 for another tenant's id?** 403 confirms the id exists.
3. **Why is upload 202?** Work exceeds a request; the client polls a status resource.
4. **Queue is at-least-once; how do you avoid double processing?** Idempotent handler, atomic claim, lease, fencing token, atomic publish.
5. **What if Redis is down at upload time?** Document is `uploaded`, durable in Postgres; the sweeper re-enqueues. (Better: outbox.)
6. **How do you do a conversational RAG follow-up?** Condense to a standalone query, retrieve fresh under current access, replay stripped history.
7. **Why not reuse the previous turn's chunks?** Access can change between turns.
8. **Rate limit vs quota?** Burst guard (cheap, disposable state) vs durable budget (atomic DB counter); different 429 codes.
9. **How do you make the quota race-free?** One conditional `UPDATE … WHERE value < limit RETURNING`.
10. **Why no tenant label on Prometheus metrics?** Unbounded cardinality; use logs/`/usage` for per-tenant.
11. **How would you shard?** By tenant via a directory; global ids first; identity split out; read replicas before shards.
12. **Replica lag and security?** Revocation must be immediate → re-check visibility on the primary.

## Where Week 8 picks up (compressed: Oct 26–27)
CRDT/Yjs collaborative-edit demo, locking/concurrency tests, throttling under load, failure injection (kill the worker mid-job; Redis down;
provider 429), and the per-system failure-mode table. Most of the machinery (optimistic locks, fencing, fail-open limiter) already exists to be *tested*.
