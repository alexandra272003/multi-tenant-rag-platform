# Operating the platform: quotas, rate limits, metrics, logs (Week 7, Saturday)

Goal: one tenant must not be able to degrade the platform for the others, and when something
goes wrong an operator must be able to say **which tenant, which request, which route**.

## 1. Two different limits, on purpose

| | Rate limit | Daily quota |
|---|---|---|
| Question it answers | "Is this tenant *bursting*?" | "Has this tenant spent its *budget*?" |
| Window | 60 s, fixed | UTC day |
| State | memory or Redis (disposable) | `usage_counters` table (durable, shared) |
| Rejection | `429 rate_limited`, `Retry-After` ≈ seconds | `429 quota_exceeded`, `Retry-After` = until UTC midnight |
| Counts rejected requests? | n/a | **No** — rejected requests consume nothing |

Clients need to tell them apart: back off for seconds vs. stop until tomorrow. Hence two error codes
on one status.

**Guarded endpoints** (`app/auth/limits.py`): `/ask`, `/ask/langchain`, `/ask/structured`, `/agent`,
`/agent/stream`, `POST /conversations/{id}/messages` (quota *queries*); `POST /documents`
(quota *uploads*). `/retrieve` and all reads are free: no LLM cost.

**Order of checks** — RBAC (403) → rate limit → quota. Cheapest first: a flood is turned away before it
costs a DB write, a forbidden request never touches counters, a throttled request never burns budget.

### Atomic quota
```sql
UPDATE usage_counters SET value = value + 1
 WHERE tenant_id=:t AND day=:d AND metric=:m AND value < :limit
RETURNING value
```
One statement decides *admit or reject*; the database serialises concurrent updates on the row, so 12
simultaneous requests with 5 left admit exactly 5 (tested on SQLite and checked on Postgres in
`scripts/verify_postgres.py`). "Read, compare, then write" would over-admit under concurrency.

Decisions: counted at **admission**, no refund on failure (the provider call may already have cost
money; refunds invite retry-loop abuse). Per-tenant overrides: `tenants.daily_query_limit` /
`daily_upload_limit` (NULL = platform default; `0` blocks the tenant).

### Rate limiter backends
`RATE_LIMIT_BACKEND=memory` — per process; with N API processes the real limit is N×.
`redis` — shared; **fails open** if Redis is down (counted in `rate_limiter_errors_total`): a cache
outage must not become a platform outage. `off` — disabled. Fixed windows allow up to 2× the limit
across a window boundary; fine for noisy-neighbour protection, not for a strict contract.

`GET /usage` (admin/owner) shows the tenant its own used/limit/remaining and reset time.

## 2. Metrics — `GET /metrics` (Prometheus)

| Metric | Labels | Use |
|---|---|---|
| `http_requests_total` | method, route, status | traffic, error rate (5xx / total) |
| `http_request_duration_seconds` | method, route | latency histogram → p50/p95/p99 |
| `rate_limited_total` | scope | throttling pressure |
| `quota_rejected_total` | metric | tenants hitting budgets (sales signal or abuse) |
| `rate_limiter_errors_total` | — | Redis trouble (limiter failing open) |
| `ingest_jobs_total` | result | ready / failed / superseded / busy / already_ready… |

**Cardinality rule:** labels use the route *template* (`/documents/{document_id}`), never the raw path,
and **never tenant or user ids** — a series per tenant grows without bound. Per-tenant questions are
answered by `/usage` and by the logs, which are built for high cardinality.

Caveats: metrics are per process. The Celery worker has its own registry (`ingest_jobs_total` is
incremented *there* and is not visible at the API's `/metrics`); scraping the worker needs its own
endpoint or a pushgateway. With several API processes, use the Prometheus multiprocess mode or scrape each.
`METRICS_TOKEN` puts a bearer token on `/metrics`.

Suggested alerts: 5xx ratio > 2% for 5 min; p95 latency of `/ask` > 10 s; `rate_limiter_errors_total`
increasing; documents stuck in `uploaded`/`processing` (query `documents` by status/age — the sweeper
also logs `sweeper re-enqueued N`); `quota_rejected_total` spike for one metric.

## 3. Logs
`LOG_FORMAT=json` → one JSON object per line on stdout (API **and** worker), e.g.
```json
{"ts":"2026-10-24T09:12:01.532+00:00","level":"INFO","logger":"access","message":"POST /ask -> 200",
 "request_id":"9f1c…","tenant_id":7,"user_id":31,"method":"POST","route":"/ask","status":200,"duration_ms":842.3}
```
* `X-Request-ID` is accepted if it matches `[A-Za-z0-9._-]{1,64}` (otherwise replaced), put on every log
  line, and echoed in the response — quote it in a bug report and the whole request is one filter away.
* `tenant_id` / `user_id` are bound by the auth dependency into a mutable per-request context (a plain
  `ContextVar.set` inside a dependency would be invisible to the middleware's access line).
* Never logged: request/response bodies, tokens, passwords (tested). Document text and queries stay out of logs.
* Uvicorn's own access log is disabled in JSON mode — ours replaces it.

## 4. Known gaps
* Login has no brute-force throttling (per-IP / per-email limiter) — the obvious next guard.
* Rate limits are not tenant-configurable (quotas are).
* No per-tenant token accounting; `queries` counts requests, not tokens. Token-based budgets need the usage
  numbers from every LLM path, which only the Week 6 structured/agent paths report today.
* Upload quota counts attempts that later fail validation after admission (e.g. duplicate → 409).
