# Multi-tenant platform: design, decisions, verification (Week 7, Mon–Wed)

Status: this document covers Mon–Wed (tenant schema + context, RBAC, migrations/audit/soft-delete/
versioning). Thursday–Sunday are in `WEEK7_INGESTION.md`, `WEEK7_OPERATIONS.md`,
`WEEK7_DISTRIBUTED_DB_MEMO.md`; the whole week is summarised in `WEEK7_SUMMARY.md`.

## 1. Tenancy model

```
Tenant 1──* User            (a user belongs to exactly one tenant, with one role)
Tenant 1──* Document 1──* Chunk      (chunks carry tenant_id too, denormalised on purpose)
Tenant 1──* AuditLog
```
`users.email` is **globally unique** (login needs no tenant slug). The cost: one person
cannot belong to two tenants without a memberships table — a deliberate simplification.

## 2. Tenant isolation, three independent layers

| Layer | Mechanism | Where |
|---|---|---|
| **API** | `get_principal` builds the caller from the DB; every query goes through `AccessFilter.document_clause()`; foreign ids return **404, never 403** | `app/auth/` |
| **SQL / retrieval** | `retrieve()` takes a **required** `access` argument — a caller that forgets to scope cannot compile; the filter runs in the WHERE clause *before* ranking, so other tenants' chunks never enter the candidate set | `retrieval_service.py` |
| **Database** | Postgres **row-level security**: `FORCE ROW LEVEL SECURITY`, policy `tenant_id = NULLIF(current_setting('app.tenant_id', true),'')::int`, fail-closed (no tenant set → zero rows) | migration `d4b8f2a1e6c3`, `core/tenancy.py` |

Why three: a bug in one (a forgotten WHERE, a raw SQL path, an agent tool) is caught by the
next. The agent's SQL tool, for instance, is already confined by its own scoped views *and*
by RLS underneath them.

**How RLS is wired.** The app connects as the schema owner (owners/superusers bypass RLS), so
at the start of **every transaction** a SQLAlchemy `after_begin` hook runs
`SET LOCAL ROLE rag_app` and sets `app.tenant_id`. `rag_app` is a NOLOGIN role with least
privilege: no `DELETE` on documents/chunks (deletion is soft), and `audit_log` is
**append-only** (`SELECT, INSERT` only). It is per *transaction*, not per request, because
every commit starts a new transaction and a request-level setting would silently vanish
after the first commit; `LOCAL` guarantees nothing leaks onto a pooled connection.

## 3. Authentication and RBAC

**Auth.** `POST /auth/register` (new tenant + first user = OWNER), `POST /auth/login`, JWT
(HS256, pinned algorithm, required `exp/sub/tid`), passwords hashed with scrypt. Login gives
the same error for wrong password / unknown email / inactive user and burns equal hashing
time for unknown emails (no user enumeration).
**Revocation is immediate:** each request re-reads the user (active? role? tenant?) from the
database, so deactivation and role changes apply to tokens that have not expired. Cost: one
primary-key lookup per request.

**Permissions** (`app/auth/rbac.py`):

| | viewer | editor | admin | owner |
|---|:-:|:-:|:-:|:-:|
| read documents / ask / search | ✓ | ✓ | ✓ | ✓ |
| create documents | | ✓ | ✓ | ✓ |
| edit/delete **own** documents | | ✓ | ✓ | ✓ |
| edit/delete **any** document | | | ✓ | ✓ |
| list users / read audit log | | | ✓ | ✓ |
| manage users | | | editors & viewers only | everyone |

Two kinds of check, kept separate: a **role** permission (`require(...)`) and an **object**
check (`can_write_document` — *this* editor, *this* document). A role check alone lets every
editor edit every document. Rules: nobody changes their own role or status; an admin cannot
touch owners/admins or mint admins; a tenant always keeps one active owner.
`visibility='private'` documents are readable only by their owner, admins and owners —
enforced in the same single `AccessFilter` used by retrieval and the agent.

## 4. Document lifecycle

- **Optimistic locking.** `PATCH` must send `expected_version`; the update is one atomic
  `UPDATE … WHERE id=? AND version=?`. Zero rows → `409 version_conflict` with the current
  version. No lock is held while a user thinks; a lost update is impossible. Verified with
  6 simultaneous edits on Postgres: exactly one 200, five 409.
- **Soft delete.** `deleted_at/deleted_by`; rows and chunks stay for audit/restore but every read
  path filters them. A **partial unique index** (`… WHERE deleted_at IS NULL`) lets a deleted
  file be re-uploaded. (Embeddings of deleted documents still occupy index space — a purge job
  is future work.)
- **Per-tenant uniqueness.** The duplicate rule is `(tenant_id, sha256, chunk params)`; a global
  rule would tell tenant B that tenant A already uploaded the file.
- **Audit.** `audit_log` rows are written in the *same transaction* as the change, so a rejected
  change leaves no phantom record. Covers tenant/user/document events and logins.

## 5. Migrations (Alembic — and why not Liquibase)

The sprint's stack table allows "Alembic **or** Liquibase", the project already runs Alembic, and
switching migration tools mid-sprint is exactly the "changing stack midway" red flag. The skills
transfer: Liquibase changesets ≈ Alembic revisions; both give reproducible, ordered, reversible schema change.

- `c3a7e1d92f10` — tenancy: **expand → backfill → contract** (add columns nullable, backfill
  tenant 1 + timestamps, then `NOT NULL`; adding NOT NULL to a populated table directly fails),
  reset the tenant id sequence, swap the global unique constraint for the partial per-tenant index.
- `d4b8f2a1e6c3` — RLS + least-privilege role.
- Downgrades exist but are lossy by design (tenants/users/audit are dropped) and refuse to recreate
  the old global constraint if two tenants hold identical files — better than silently deleting data.

## 6. Failure modes

| Situation | Behavior |
|---|---|
| No/invalid/expired token, `alg:none`, wrong tenant claim | 401, one generic message |
| Valid token, user since deactivated or demoted | next request reflects it |
| Foreign / deleted / private-not-yours document | 404 (same as nonexistent) |
| Role lacks permission | 403 naming the permission |
| Editor edits someone else's document | 403 (after the 404 check) |
| Stale `expected_version` | 409 `version_conflict` + current version |
| Duplicate email (incl. simultaneous signups) | exactly one 201, rest 409 |
| Code forgets a tenant filter | RLS returns only the current tenant's rows; no tenant set → none |
| `JWT_SECRET` left at the default outside development | app refuses to start |

## 7. How this was verified

- **382 offline tests** (SQLite): auth, RBAC matrix, isolation across API/retrieval/Q&A/agent/SQL tool/private docs,
  versioning (incl. a concurrent-update race), soft delete, audit. Every new control was **mutation-checked**
  (break it → a test fails): 17 of 17.
- **`scripts/verify_postgres.py`** (44 checks, real Postgres 16 + pgvector): migration on legacy data, **no drift between
  models and migrations**, downgrade → upgrade round trip, RLS behaviour (fail-closed, WITH CHECK, no
  context leak on pooled connections, append-only audit), full API under RLS, 6-way edit race, signup race.
  The owner-role **control** check proves enforcement comes from the role switch, not from luck.

**Bugs the verification found** (worth telling as a story): (1) a signup race returned 500 — the unique
violation fires at `flush()`, outside my handler; reproduced on SQLite, fixed by classifying
`IntegrityError` (email → 409, slug → retry); (2) SQLite and Postgres disagree on a CTE named like its own
source table — the agent's SQL scoping now rewrites table names instead; (3) the tests' own raw inserts
exposed that timestamps are ORM-side defaults, not database defaults (kept, noted below).

## 8. Known limits / next

- Timestamps are filled by the ORM, not DB defaults, so raw SQL writers must supply them.
- Single-tenant-per-user; `document_ids`/scope in the agent now derive from the caller, but quotas/rate
  limits per tenant (Sat) are not built, so one tenant can still consume all model capacity.
- JWT is HS256 with a shared secret and no refresh/rotation; production wants key rotation (`kid`) and short TTLs.
- RLS is exercised via `SET ROLE`; a production deployment should connect as a non-owner login role instead.
- Embeddings of soft-deleted documents are not purged.

## 9. Interview questions this answers

1. **How do you prevent cross-tenant leakage in RAG?** Three layers: the API builds every query from one `AccessFilter`;
   retrieval *requires* it and applies it before ranking (so other tenants' chunks never enter the candidate set); and
   Postgres RLS enforces the tenant regardless. Tests cover each layer, including the agent and its SQL tool.
2. **JWT proves identity — how do you enforce ownership?** Role permission *and* an object-level check; role and tenant are
   re-read from the DB each request so revocation is immediate.
3. **Why 404 instead of 403 for another tenant's document?** 403 confirms the id exists.
4. **Optimistic locking vs `SELECT … FOR UPDATE`?** Optimistic for user edits (no lock held while a human thinks; conflict
   is rare and reported); pessimistic for short critical sections like inventory decrement.
5. **Why does RLS fail closed, and what is `NULLIF` for?** An unset setting makes the predicate NULL → no rows. After a
   `set_config(..., true)` transaction a pooled connection reports `''`, and `''::int` would raise.
6. **What breaks first at 100× scale?** The per-request user lookup and `IN (:ids)` agent scopes; fix with caching a
   short-lived principal and a tenant-keyed subquery instead of id lists.
