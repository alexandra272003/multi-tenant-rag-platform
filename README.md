# capstone-platform (Week 7 Capstone I)
Multi-tenant document + RAG platform: JWT auth, RBAC, Postgres row-level security, pgvector retrieval,
audit log, optimistic locking, soft delete, **async ingestion** (Celery + Redis), **conversational RAG**,
**per-tenant quotas + rate limits**, Prometheus metrics, structured JSON logs.

## Run
```
docker compose up --build
```
Services: `postgres`, `redis`, `api` (:8000), `worker` (Celery + beat sweeper). Set `LLM_API_KEY` in `.env`.

## Try it
1. `POST /auth/register` → token (`Authorization: Bearer …`)
2. `POST /documents` (file) → 202, `status: uploaded`; poll `GET /documents/{id}` until `ready`
   (`failed` → `POST /documents/{id}/retry`)
3. `POST /conversations`, then `POST /conversations/{id}/messages {"query": "..."}` — ask a follow-up
4. `GET /usage` (admin/owner), `GET /metrics`

## Verify
* `pytest` — 468 tests, no Postgres/Redis/network needed.
* `docker compose exec api python scripts/verify_postgres.py` — RLS, grants, concurrency on real Postgres.

## Docs
`docs/WEEK7_SUMMARY.md` (start here) · `WEEK7_TENANCY.md` · `WEEK7_INGESTION.md` · `WEEK7_OPERATIONS.md` ·
`WEEK7_DISTRIBUTED_DB_MEMO.md` · Week 6 README: `docs/README_WEEK6.md`
# multi-tenant-rag-platform
