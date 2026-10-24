# Design memo: taking this platform from one Postgres to a distributed database (Week 7, Sunday)

**Status: a proposal.** Nothing here is implemented; the current system is one Postgres primary with
pgvector, one Redis, N stateless API processes and a Celery worker pool. This memo says what would
break first as load and customer requirements grow, what I would change in what order, and what I would
deliberately refuse to do.

## 1. What the data looks like (it decides everything)
* **Tenant-partitioned by construction.** Every row carries `tenant_id`; no query crosses tenants
  except the sweeper and metrics. This is the best property a system can have before sharding.
* **Mixed access patterns:** small transactional rows (users, documents, conversations, quotas); a large,
  append-mostly, read-heavy **vector table** (chunks); a hot **counter row** per tenant/metric/day; a
  durable **job state** (documents.status) that the queue merely points at.
* **Correctness requirements differ by table.** Quotas and optimistic-lock versions need linearizable
  single-row updates. Retrieval can tolerate a few seconds of staleness. Audit must never lose a row.

## 2. What breaks first, in order
| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Vector search slows, then hurts OLTP | ivfflat/HNSW on one primary shares CPU and cache with transactional work | **Read replicas** for retrieval |
| 2 | Primary CPU/IO saturated | embeddings + chunk inserts + counters + audit on one node | **Shard by tenant** |
| 3 | One tenant dominates a shard | big tenants (skew) | **Move tenant** to its own shard |
| 4 | Customers demand EU-only data | residency | **Region-pinned shards** |
| 5 | Region outage | single region | **Cross-region replica / failover** |

I would do 1 long before 2: replicas are cheap, reversible and need no application re-architecture.

## 3. Step 1 — read replicas for retrieval and listing
Route `/retrieve`, `/ask*` context fetch, `GET /documents` to replicas; keep writes, auth lookup, quotas,
`PATCH`/`DELETE` and **read-your-writes-sensitive** reads on the primary.

The hazards that are specific to *this* system:
* **Stale authorization is a security bug, not a UX bug.** A document made private or deleted on the primary
  must stop being retrievable. With asynchronous replication a replica can serve it for the lag window.
  Options: (a) accept a bounded lag and *document it* ("revocation takes effect within X s"); (b) re-check
  visibility of the returned document ids against the primary (cheap PK lookup, after the vector search);
  (c) synchronous replication for the `documents` table only. I would take **(b)**: it keeps the fast path
  on replicas and makes revocation immediate, at the price of one small primary query.
* **Read-your-writes:** upload → poll status must read the primary (or use a replication-position/LSN
  token). `documents.status` is a state machine the client polls; a stale replica would show `uploaded` after
  `ready`.
* **Lag monitoring and fallback:** if lag > threshold, send reads to the primary rather than serve very old data.
* RLS is per-database state: replicas replicate policies, but the `SET LOCAL ROLE` + tenant hook must run on
  replica connections too.

## 4. Step 2 — sharding: the key is `tenant_id`
**Why tenant, not document or hash-of-row:** every query is already tenant-scoped, so a router can send a
request to exactly one shard with no scatter-gather; RLS, foreign keys, unique constraints
(`uq_doc_active_content`), joins and *transactions* stay within a shard. Sharding by document id would turn
"list my documents", the agent's SQL tool and tenant-scoped uniqueness into cross-shard operations.

**Mechanism:** a small, highly available **directory** (tenant → shard) — itself a replicated table with a cached
copy in every API process — and a router that resolves it from the JWT's `tid`. Not consistent hashing of
`tenant_id`: directory lookup lets me *move one tenant* (section 5) without rehashing everyone.

**What has to change in the code**
* Global uniqueness breaks: `users.email` is globally unique today (login needs no tenant slug). On shards that
  cannot be enforced by a constraint. Move **identity** (email → tenant, credentials) into a small global
  directory database, or require the tenant in login. This is the single hardest application change.
* Cross-tenant jobs: the sweeper iterates shards; Celery tasks carry `(tenant_id, document_id)` already, so
  workers resolve the shard from the tenant.
* IDs: per-shard sequences collide across shards. Use a global id (UUIDv7 / snowflake) *before* sharding;
  retrofitting ids is painful — this is the migration I would schedule first even while still on one node.
* Migrations run per shard; the expand/contract discipline already in use (additive first) matters even more
  when shards are at different versions mid-rollout.
* Metrics/analytics across tenants: a warehouse fed by CDC, not queries over shards.

**What I would not do:** a distributed SQL engine (Spanner-/Cockroach-style) just for sharding. It would buy
transparent cross-shard transactions that this workload never needs, while pgvector support, RLS semantics and
operating cost are all worse. Revisit only if a genuine cross-tenant transactional need appears.

## 4b. Hot spots specific to this schema
* **Quota counter row** — one row per tenant/metric/day, updated on every LLM request. Fine at hundreds of req/s
  per tenant; beyond that row-lock contention serialises requests. Fix: **sharded counters** (N sub-rows chosen by
  hash, admit if `sum < limit`) with the cost that the limit becomes approximate near the boundary — or move the
  hot path to Redis `INCR` with periodic flush to Postgres and accept losing at most one flush interval on a crash.
  Billing-grade accuracy would need the durable path.
* **ivfflat/HNSW growth** — per-tenant vector counts are small, global is large. With tenant sharding each shard's index
  is smaller and *recall for a tenant stops depending on other tenants' data*. Today the planner may scan the global
  index and filter by tenant afterwards (post-filtering hurts recall); partitioning chunks by tenant, or per-tenant
  indexes for large tenants, fixes that and is worth doing before sharding.
* **Audit log** — append-only and unbounded: time-partition it, ship to cold storage, keep 90 days hot.

## 5. Moving a tenant (the operation that proves the design)
1. Mark tenant `migrating` in the directory (writes → 503 with Retry-After, reads continue) — or use dual-write if
   the downtime is unacceptable.
2. Copy rows (logical replication filtered by `tenant_id`, or `COPY` per table in FK order).
3. Verify counts + checksums per table.
4. Flip the directory entry; unpause. 5. Delete from the source after a retention window.
In-flight Celery jobs carry only ids, so they follow the directory; the fencing token (`documents.attempts`) stops
a worker that read the old shard from publishing to the new one. RLS and `FORCE` policies must exist on the target
before the flip.

## 6. Consistency model per operation
| Operation | Needs | Mechanism |
|---|---|---|
| Edit a document | linearizable per document | conditional UPDATE on `version` (primary) |
| Quota admit | linearizable per counter | conditional UPDATE (primary), see 4b |
| Upload + enqueue | **atomic** DB write + message | today: dual write with a sweeper; **better: transactional outbox** |
| Retrieval | bounded staleness, **but revocation must be immediate** | replica + primary visibility re-check |
| Audit write | durable, in the same transaction as the change | same shard, same transaction |
| Conversation turn | per-conversation ordering | `messages.id` order; one writer per conversation turn is enough |

**Outbox:** today a crash between "commit document" and "enqueue" leaves the document `uploaded`; the sweeper heals
it after ~2 minutes. That is correct but slow. With an outbox table written in the *same* transaction as the document
and a relay publishing to the queue, the message cannot be lost, only delayed or duplicated — and duplicates are already
harmless because `process_document` is idempotent and fenced. I would add it when sweeper latency becomes a product complaint,
not before: it adds a moving part.

## 7. Failure scenarios and what each design gives you
* **Primary dies:** automatic failover to a synchronous standby (no data loss) or async standby (loses the last
  moments of commits). Quotas can under-count after an async failover (more admitted than the budget — tolerable);
  documents committed but lost mean a client holds an id that 404s — hence synchronous replication for the primary of
  record. Workers retry connection errors as transient.
* **Replica lag spike:** reads fall back to the primary (section 3); stale-visibility re-check prevents leaks.
* **Network partition between API and one shard:** only that shard's tenants are affected (blast radius = shard);
  the directory and other shards keep serving. This is the strongest argument for tenant sharding.
* **Redis down:** rate limits fail open; queue unavailable → uploads still accepted (`uploaded`), sweeper recovers
  when Redis returns; quotas are unaffected (they live in Postgres).
* **Directory unavailable:** API processes keep serving from their cached copy; tenant moves and signups pause.
  The directory is the one component that must be the most boring and the most replicated.
* **Split brain on failover:** fencing at the storage layer (only one writable primary per shard — enforced by the
  failover tool, not by application code).

CAP/PACELC in one sentence: per shard, I choose **consistency over availability under partition** for writes (one primary per
tenant), and **latency over consistency in normal operation for retrieval reads only**, with the revocation check as the
guard that keeps that trade from becoming a security hole.

## 8. Order of work
1. Global ids (UUIDv7) and make every code path take `tenant_id` explicitly (mostly done). 2. Per-tenant partitioning/indexing of
chunks. 3. Read replicas for retrieval with the visibility re-check. 4. Identity directory (email → tenant) split out.
5. Tenant directory + router + move-tenant tooling. 6. Second shard, large tenants first. 7. Region-pinned shards if residency demands.
Each step is independently shippable and reversible until step 5.

## 9. Interview-ready answers
* *Why shard by tenant?* Every query is already tenant-scoped; transactions, uniqueness and RLS stay inside one shard; blast radius
  equals one shard.
* *What's hard about it?* Anything globally unique (user emails, ids), cross-tenant analytics, and moving a tenant live.
* *Read replicas and security?* Replica lag turns revocation into an eventual operation; re-check visibility on the primary.
* *Exactly-once?* Not available end to end; at-least-once delivery + idempotent handler + fenced publish.
* *Where's the hot row?* The per-tenant quota counter; shard it or front it with Redis depending on accuracy needs.
