"""
Verifies, against a REAL Postgres + pgvector, everything the SQLite unit tests
cannot: the migrations (on legacy data, both directions), model/schema drift,
row-level security, and the full API end to end.

    python scripts/verify_postgres.py --admin-url postgresql+asyncpg://USER:PASS@HOST:5432/postgres

It creates and drops a throwaway database `rag_verify`; the admin role needs
CREATEDB (and, for the vector extension, superuser). Not part of `pytest`
because it needs a live server -- run it before trusting a schema change.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--admin-url", default="postgresql+asyncpg://rag_user:rag_pass@localhost:5432/postgres")
ARGS = parser.parse_args()
DB = "rag_verify"
ADMIN_URL = ARGS.admin_url
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + f"/{DB}"
os.environ["DATABASE_URL"] = TEST_URL  # BEFORE any app import: app.core.db reads it at import time
os.environ["PASSWORD_HASH_N"] = "16"

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

PASSED = []


def ok(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra and not cond else ""))
    if not cond:
        sys.exit(1)
    PASSED.append(name)


def alembic(*args: str) -> None:
    r = subprocess.run(["alembic", *args], cwd=ROOT, env=os.environ, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(f"alembic {' '.join(args)} failed")


async def scalar(conn, sql, **params):
    return (await conn.execute(text(sql), params)).scalar()


VEC = "[" + ",".join(["1"] + ["0"] * 383) + "]"


async def main() -> None:
    admin = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    async with admin.connect() as c:
        await c.execute(text(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)"))
        await c.execute(text(f"CREATE DATABASE {DB}"))
    owner = create_async_engine(TEST_URL)

    # ---------------------------------------------------------------- 1. legacy data -> upgrade
    print("\n== migration on a populated (pre-tenancy) database ==")
    alembic("upgrade", "8a1c2f9d4b3e")
    async with owner.begin() as c:
        for i, name in enumerate(["legacy_a.txt", "legacy_b.txt"], start=1):
            await c.execute(text(
                "INSERT INTO documents (filename, source_type, content_sha256, page_count, chunk_size, "
                "chunk_overlap, chunk_count, created_at) VALUES (:f,'txt',:h,1,500,0,2, now() - interval '1 day')"), {"f": name, "h": f"h{i}"})
        await c.execute(text(
            "INSERT INTO chunks (document_id, chunk_index, content, char_start, char_end, token_count, embedding, created_at) "
            "SELECT d.id, g, 'legacy chunk ' || d.id || '-' || g, 0, 10, 3, CAST(:v AS vector), now() "
            "FROM documents d, generate_series(0,1) g"), {"v": VEC})
    alembic("upgrade", "head")
    async with owner.connect() as c:
        ok("legacy documents adopted by tenant 1", await scalar(c, "SELECT count(*) FROM documents WHERE tenant_id = 1") == 2)
        ok("legacy chunks backfilled with their document's tenant",
           await scalar(c, "SELECT count(*) FROM chunks WHERE tenant_id = 1") == 4)
        ok("version=1, visibility='tenant', updated_at=created_at backfilled",
           await scalar(c, "SELECT count(*) FROM documents WHERE version=1 AND visibility='tenant' AND updated_at = created_at") == 2)
        ok("default tenant exists", await scalar(c, "SELECT slug FROM tenants WHERE id = 1") == "default")
        ok("tenant id sequence advanced past the default tenant",
           await scalar(c, "INSERT INTO tenants (name, slug) VALUES ('T2','t2') RETURNING id") == 2)
        await c.commit()
        ok("old global unique constraint replaced by partial per-tenant index",
           await scalar(c, "SELECT count(*) FROM pg_indexes WHERE indexname='uq_doc_active_content' AND indexdef LIKE '%deleted_at IS NULL%'") == 1
           and await scalar(c, "SELECT count(*) FROM pg_constraint WHERE conname='uq_doc_content_params'") == 0)
        ok("new tenant-scoped indexes exist",
           await scalar(c, "SELECT count(*) FROM pg_indexes WHERE indexname IN ('ix_documents_tenant_id_id','ix_chunks_tenant_document','ix_audit_tenant_created','ix_users_tenant_id')") == 4)
        ok("ivfflat index survived", await scalar(c, "SELECT count(*) FROM pg_indexes WHERE indexname='ix_chunks_embedding_cosine'") == 1)
        ok("RLS enabled + forced on every tenant table (incl. sources, conversations, messages, usage)",
           await scalar(c, "SELECT count(*) FROM pg_class WHERE relname IN ('documents','chunks','audit_log','document_sources','conversations','messages','usage_counters') AND relrowsecurity AND relforcerowsecurity") == 7)

    # ---------------------------------------------------------------- 2. model/schema drift
    print("\n== models vs migrated schema ==")
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from app.models import Base

    def include_object(obj, name, type_, reflected, compare_to):
        return not (type_ == "index" and name == "ix_chunks_embedding_cosine")  # ivfflat is migration-only

    def diff(sync_conn):
        ctx = MigrationContext.configure(sync_conn, opts={"compare_type": True, "include_object": include_object})
        return compare_metadata(ctx, Base.metadata)

    async with owner.connect() as c:
        drift = await c.run_sync(diff)
    ok("no drift between SQLAlchemy models and migrations", drift == [], str(drift)[:600])

    # ---------------------------------------------------------------- 3. downgrade / upgrade round trip
    print("\n== reproducibility: downgrade and upgrade again ==")
    alembic("downgrade", "c3a7e1d92f10")  # drop RLS only
    async with owner.connect() as c:
        ok("RLS migration reverses cleanly", await scalar(c, "SELECT count(*) FROM pg_roles WHERE rolname='rag_app'") == 0
           and await scalar(c, "SELECT count(*) FROM pg_policies") == 0)
    alembic("downgrade", "8a1c2f9d4b3e")
    async with owner.connect() as c:
        ok("tenancy migration reverses; legacy rows intact",
           await scalar(c, "SELECT count(*) FROM documents") == 2 and await scalar(c, "SELECT count(*) FROM chunks") == 4
           and await scalar(c, "SELECT count(*) FROM information_schema.tables WHERE table_name='tenants'") == 0)
        ok("old unique constraint is back", await scalar(c, "SELECT count(*) FROM pg_constraint WHERE conname='uq_doc_content_params'") == 1)
    alembic("upgrade", "head")
    async with owner.connect() as c:
        ok("re-upgrade lands in the same schema (still no drift)", await c.run_sync(diff) == [])
        ok("legacy data still adopted after the round trip", await scalar(c, "SELECT count(*) FROM documents WHERE tenant_id = 1") == 2)

    # ---------------------------------------------------------------- 4. row-level security
    print("\n== row-level security (through the app's session hook) ==")
    from app.core import tenancy  # noqa: F401  (registers the after_begin hook)
    app_engine = create_async_engine(TEST_URL)
    Sess = async_sessionmaker(app_engine, expire_on_commit=False)

    # Stand-in for the Celery queue: run the real worker function inline, on the
    # same RLS-restricted engine the API uses.
    from app.jobs import dispatch
    from app.services import ingestion_service

    async def _inline_enqueue(tenant_id, document_id):
        await ingestion_service.process_document(Sess, tenant_id, document_id)

    dispatch.enqueue_ingestion = _inline_enqueue

    async def as_tenant(tid):
        s = Sess()
        if tid is not None:
            s.info["tenant_id"] = tid
        return s

    async with owner.begin() as c:
        await c.execute(text("INSERT INTO tenants (name, slug) VALUES ('T3','t3')"))  # id 3 (id 2 made above was rolled back by downgrade)
    async with owner.connect() as c:
        t_other = await scalar(c, "SELECT max(id) FROM tenants")

    s = await as_tenant(t_other)
    async with s:
        await s.execute(text(
            "INSERT INTO documents (tenant_id, filename, source_type, content_sha256, page_count, chunk_size, chunk_overlap, chunk_count, created_at, updated_at) "
            "VALUES (:t, 'other.txt', 'txt', 'hx', 1, 500, 0, 1, now(), now())"), {"t": t_other})
        await s.commit()

    s1, s2, s0 = await as_tenant(1), await as_tenant(t_other), await as_tenant(None)
    async with s1, s2, s0:
        ok("tenant 1 sees only its own documents", await scalar(s1, "SELECT count(*) FROM documents") == 2)
        ok("other tenant sees only its own document", await scalar(s2, "SELECT count(*) FROM documents") == 1)
        ok("NO tenant context -> zero rows (fail closed)", await scalar(s0, "SELECT count(*) FROM documents") == 0)
        ok("'SELECT * FROM documents' with no WHERE is still confined (injection-style)",
           {r[0] for r in (await s2.execute(text("SELECT filename FROM documents"))).all()} == {"other.txt"})
        ok("chunks are confined too", await scalar(s2, "SELECT count(*) FROM chunks") == 0 and await scalar(s1, "SELECT count(*) FROM chunks") == 4)
        ok("current role inside the app transaction is rag_app (not the owner)", await scalar(s1, "SELECT current_user") == "rag_app")

    s2b = await as_tenant(t_other)
    async with s2b:
        res = await s2b.execute(text("UPDATE documents SET title = 'pwned'"))
        ok("UPDATE with no WHERE only touches own tenant's rows", res.rowcount == 1)
        await s2b.commit()
    async with owner.connect() as c:
        ok("tenant 1's rows were not modified", await scalar(c, "SELECT count(*) FROM documents WHERE title = 'pwned'") == 1)

    s2c = await as_tenant(t_other)
    async with s2c:
        try:
            await s2c.execute(text(
                "INSERT INTO documents (tenant_id, filename, source_type, content_sha256, page_count, chunk_size, chunk_overlap, chunk_count, created_at, updated_at) "
                "VALUES (1, 'smuggled.txt', 'txt', 'hy', 1, 500, 0, 1, now(), now())"))
            blocked = False
        except Exception as e:
            blocked = "row-level security" in str(e).lower()
        ok("cannot INSERT a row into another tenant (WITH CHECK)", blocked)
        await s2c.rollback()

    s2d = await as_tenant(t_other)
    async with s2d:
        try:
            await s2d.execute(text("DELETE FROM documents")); denied = False
        except Exception as e:
            denied = "permission denied" in str(e).lower()
        ok("hard DELETE on documents is not granted (deletion is soft)", denied)
        await s2d.rollback()
    s2e = await as_tenant(t_other)
    async with s2e:
        await s2e.execute(text("INSERT INTO audit_log (tenant_id, action, resource_type) VALUES (:t, 'x', 'y')"), {"t": t_other})
        await s2e.commit()
    for stmt in ("UPDATE audit_log SET action = 'tampered'", "DELETE FROM audit_log"):
        s2f = await as_tenant(t_other)
        async with s2f:
            try:
                await s2f.execute(text(stmt)); denied = False
            except Exception as e:
                denied = "permission denied" in str(e).lower()
            ok(f"audit_log is append-only ({stmt.split()[0]} refused)", denied)
            await s2f.rollback()

    s1b = await as_tenant(1)
    async with s1b:
        await s1b.execute(text("UPDATE documents SET title = 'first' WHERE id = (SELECT min(id) FROM documents)"))
        await s1b.commit()  # a commit ends the transaction...
        ok("tenant context survives a commit (hook re-applies per transaction)", await scalar(s1b, "SELECT count(*) FROM documents") == 2)
    s_clean = await as_tenant(None)
    async with s_clean:
        ok("no context leaks onto a pooled connection reused after another tenant", await scalar(s_clean, "SELECT count(*) FROM documents") == 0)
    async with owner.connect() as c:
        ok("CONTROL: the owner role (no SET ROLE) sees every tenant's rows -- the role switch is what enforces isolation",
           await scalar(c, "SELECT count(*) FROM documents") == 3)

    # ---------------------------------------------------------------- 5. the real API on Postgres
    print("\n== full API on Postgres (RLS + pgvector + real concurrency) ==")
    import httpx
    from httpx import ASGITransport, AsyncClient
    from app.core.config import settings
    from app.main import app
    from app.agent import loop
    from tests.agent_helpers import ScriptedModel, text_turn, tool_turn
    import app.services.ingestion_service as ing, app.services.retrieval_service as ret

    async def fake_embed(texts):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]
    ing.embed_texts = fake_embed
    ret.embed_texts = fake_embed

    from app.cli.bootstrap_owner import bootstrap_owner
    await bootstrap_owner("legacy-owner@x.test", "password-123")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as api:
        async def login(email, pw="password-123"):
            r = await api.post("/auth/login", json={"email": email, "password": pw})
            return {"Authorization": "Bearer " + r.json()["access_token"]}

        legacy = await login("legacy-owner@x.test")
        docs = (await api.get("/documents", headers=legacy)).json()
        ok("bootstrap_owner makes pre-tenancy documents reachable", {d["filename"] for d in docs} == {"legacy_a.txt", "legacy_b.txt"})
        r = await api.post("/retrieve", headers=legacy, json={"query": "legacy chunk", "top_k": 10})
        ok("pgvector retrieval works under RLS for the legacy tenant", len(r.json()["results"]) == 4)

        ra = await api.post("/auth/register", json={"tenant_name": "Acme", "email": "a@acme.test", "password": "password-123"})
        rb = await api.post("/auth/register", json={"tenant_name": "Beta", "email": "b@beta.test", "password": "password-123"})
        ok("self-service signup works (sequence + RLS audit insert)", ra.status_code == 201 and rb.status_code == 201, ra.text + rb.text)
        A = {"Authorization": "Bearer " + ra.json()["access_token"]}
        B = {"Authorization": "Bearer " + rb.json()["access_token"]}

        up = await api.post("/documents", headers=A, files={"file": ("payroll.txt", b"SECRET-TOKEN-XYZ payroll: alice 900k")}, params={"chunk_size": 500, "chunk_overlap": 0})
        ok("upload under RLS (document + chunks + audit in one transaction)", up.status_code == 202, up.text)
        a_doc = up.json()["id"]
        await api.post("/documents", headers=B, files={"file": ("menu.txt", b"Beta lunch menu: soup and bread")}, params={"chunk_size": 500, "chunk_overlap": 0})

        ok("B cannot GET A's document (404)", (await api.get(f"/documents/{a_doc}", headers=B)).status_code == 404)
        rr = await api.post("/retrieve", headers=B, json={"query": "payroll", "top_k": 10, "document_id": a_doc})
        ok("B's pgvector retrieval never returns A's chunks", rr.json()["results"] == [])
        rr = await api.post("/retrieve", headers=B, json={"query": "payroll", "top_k": 10})
        ok("B only retrieves its own corpus", {x["filename"] for x in rr.json()["results"]} == {"menu.txt"})
        dup = await api.post("/documents", headers=B, files={"file": ("payroll.txt", b"SECRET-TOKEN-XYZ payroll: alice 900k")}, params={"chunk_size": 500, "chunk_overlap": 0})
        ok("same bytes in B succeed (per-tenant uniqueness, no leak)", dup.status_code == 202, dup.text)

        model = ScriptedModel([tool_turn("query_documents", {"sql": "select filename from documents"}, "c1"), text_turn("done")])
        loop._get_model = lambda names: model
        await api.post("/agent", headers=B, json={"query": "q"})
        out = model.calls[1][-1].content
        ok("agent SQL tool on Postgres: B sees only B's files", "menu.txt" in out and "legacy" not in out)

        # real concurrency: row-level locking, not SQLite's single writer
        d = (await api.get("/documents", headers=A)).json()[0]
        res = await asyncio.gather(*[
            api.patch(f"/documents/{d['id']}", headers=A, json={"title": f"t{i}", "expected_version": 1}) for i in range(6)])
        codes = sorted(r.status_code for r in res)
        ok("6 simultaneous edits on Postgres -> exactly one 200, five 409", codes == [200] + [409] * 5, str(codes))
        res = await asyncio.gather(*[
            api.post("/auth/register", json={"tenant_name": f"Race{i}", "email": "race@x.test", "password": "password-123"}) for i in range(4)])
        codes = sorted(r.status_code for r in res)
        ok("4 simultaneous signups with one email -> exactly one 201", codes == [201, 409, 409, 409], str(codes))

        await api.delete(f"/documents/{a_doc}", headers=A)
        rr = await api.post("/retrieve", headers=A, json={"query": "payroll"})
        ok("soft-deleted document leaves pgvector retrieval", rr.json()["results"] == [])
        up2 = await api.post("/documents", headers=A, files={"file": ("payroll.txt", b"SECRET-TOKEN-XYZ payroll: alice 900k")}, params={"chunk_size": 500, "chunk_overlap": 0})
        ok("deleted file can be re-uploaded (partial unique index on Postgres)", up2.status_code == 202, up2.text)
        audit = (await api.get("/audit", headers=A)).json()
        ok("audit trail recorded the lifecycle", {"document.uploaded", "document.ingested", "document.deleted"} <= {e["action"] for e in audit})

        # ---- async ingestion on real Postgres ------------------------------------
        from datetime import datetime, timedelta, timezone
        dispatch.enqueue_ingestion = lambda t, d: asyncio.sleep(0)  # broker "down": nothing runs
        up3 = await api.post("/documents", headers=B, files={"file": ("late.txt", b"Beta late document text for sweeper")}, params={"chunk_size": 500, "chunk_overlap": 0})
        d3 = up3.json()
        ok("upload with no worker stays 'uploaded' (202)", up3.status_code == 202 and d3["status"] == "uploaded")
        tid = (await api.get("/me", headers=B)).json()["tenant_id"]
        ok("job is idempotent under concurrency: 4 workers, one wins",
           sorted(await asyncio.gather(*[ingestion_service.process_document(Sess, tid, d3["id"]) for _ in range(4)]))
           .count("ready") == 1)
        again = await ingestion_service.process_document(Sess, tid, d3["id"])
        ok("redelivery after ready is a no-op", again == "already_ready", again)
        chunks = (await api.get(f"/documents/{d3['id']}/chunks", headers=B)).json()
        ok("exactly one set of chunks was written", len(chunks) == (await api.get(f"/documents/{d3['id']}", headers=B)).json()["chunk_count"])
        dispatch.enqueue_ingestion = _inline_enqueue
        up4 = await api.post("/documents", headers=B, files={"file": ("swept.txt", b"Beta swept document text")}, params={"chunk_size": 500, "chunk_overlap": 0})
        ok("once a worker is back, new uploads reach ready", up4.json()["status"] == "ready")

        # ---- conversations + quotas on real Postgres -------------------------------
        from app.core.config import settings as _settings
        from app.services import generation_service as _gen

        async def _fake_llm(messages):
            return "Alice earns 900k [1]." if not messages[0]["content"].startswith("Rewrite") else "What does alice earn?"
        _gen.get_chat_completion = _fake_llm
        _settings.rate_limit_backend = "off"

        cv = await api.post("/conversations", headers=A, json={})
        ok("conversation created under RLS (sequence grant works)", cv.status_code == 201, cv.text)
        cid = cv.json()["id"]
        t1 = await api.post(f"/conversations/{cid}/messages", headers=A, json={"query": "payroll alice"})
        t2 = await api.post(f"/conversations/{cid}/messages", headers=A, json={"query": "and bob?"})
        ok("two conversational turns persisted (follow-up condensed)", t1.status_code == 200 and t2.status_code == 200 and t2.json()["standalone_query"] == "What does alice earn?", t2.text)
        ok("B cannot read or post to A's conversation (RLS + user scope)",
           (await api.get(f"/conversations/{cid}", headers=B)).status_code == 404
           and (await api.post(f"/conversations/{cid}/messages", headers=B, json={"query": "x"})).status_code == 404)
        ok("A's conversation has 4 messages", len((await api.get(f"/conversations/{cid}", headers=A)).json()["messages"]) == 4)
        ok("deleting a conversation removes its messages", (await api.delete(f"/conversations/{cid}", headers=A)).status_code == 204)

        used_before = (await api.get("/usage", headers=A)).json()["queries"]["used"]
        _settings.daily_query_limit = used_before + 5
        res = await asyncio.gather(*[api.post("/ask", headers=A, json={"query": "payroll"}) for _ in range(15)])
        codes = sorted(r.status_code for r in res)
        ok("15 simultaneous queries at 5 remaining -> exactly 5 admitted (atomic quota on Postgres)",
           codes.count(200) == 5 and codes.count(429) == 10, str(codes))
        ok("B's quota is independent", (await api.post("/ask", headers=B, json={"query": "menu"})).status_code == 200)
        ok("usage endpoint agrees", (await api.get("/usage", headers=A)).json()["queries"]["remaining"] == 0)
        _settings.daily_query_limit = 500

    await app_engine.dispose(); await owner.dispose()
    async with admin.connect() as c:
        await c.execute(text(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)"))
    await admin.dispose()
    print(f"\nALL {len(PASSED)} POSTGRES CHECKS PASSED")


asyncio.run(main())
