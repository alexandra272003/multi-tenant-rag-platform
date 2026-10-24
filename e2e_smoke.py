"""
End-to-end smoke test against a RUNNING stack (real Postgres, Redis, Celery worker, LLM).

    python scripts/e2e_smoke.py [--base http://localhost:8000]

Exercises the whole product through its public API: signup, async upload + polling, duplicate
detection, grounded Q&A with citations, conversational follow-up, tenant isolation, RBAC, a failing
job + retry, quotas/usage and metrics. Prints PASS/FAIL per check; exit code 1 on any FAIL.
A WARN means something model-dependent (wording of an LLM answer) did not match -- not a bug.
"""
import argparse
import sys
import time
import uuid

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://localhost:8000")
parser.add_argument("--timeout", type=int, default=180, help="seconds to wait for a document to finish ingesting")
ARGS = parser.parse_args()

FAILS = 0
TEXT = ("The launch of Project Atlas was approved by Dana Whitfield in March. "
        "The budget for Atlas is 2 million dollars. The Atlas team has 12 engineers.")


def check(name: str, cond: bool, extra: str = "") -> bool:
    global FAILS
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra and not cond else ""))
    if not cond:
        FAILS += 1
    return cond


def warn(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS  " if cond else "WARN  ") + name + (f"   [{extra}]" if extra and not cond else ""))


def wait_for(c: httpx.Client, doc_id: int, want: set[str]) -> dict:
    deadline = time.time() + ARGS.timeout
    doc = {}
    while time.time() < deadline:
        doc = c.get(f"/documents/{doc_id}").json()
        if doc.get("status") in want:
            return doc
        time.sleep(1.5)
    return doc


def signup(base: str, label: str) -> httpx.Client:
    c = httpx.Client(base_url=base, timeout=60)
    email = f"{label}-{uuid.uuid4().hex[:8]}@e2e.test"
    r = c.post("/auth/register", json={"tenant_name": f"E2E {label} {uuid.uuid4().hex[:4]}", "email": email, "password": "password-123"})
    assert r.status_code == 201, r.text
    c.headers["Authorization"] = "Bearer " + r.json()["access_token"]
    c.email = email
    return c


def main() -> None:
    print(f"== end-to-end smoke test against {ARGS.base} ==")
    a = signup(ARGS.base, "alpha")
    check("API is up (ping)", httpx.get(ARGS.base + "/ping", timeout=10).json()["status"] == "ok")
    check("tenant A signed up and is authenticated", a.get("/me").status_code == 200)

    print("\n-- async ingestion (real Redis + Celery) --")
    r = a.post("/documents", params={"chunk_size": 200, "chunk_overlap": 20}, files={"file": ("atlas.txt", TEXT.encode())})
    check("upload returns 202 uploaded with Location", r.status_code == 202 and r.json()["status"] in ("uploaded", "processing") and r.headers.get("location", "").startswith("/documents/"), r.text[:200])
    doc_id = r.json()["id"]
    t0 = time.time()
    doc = wait_for(a, doc_id, {"ready", "failed"})
    check(f"worker finished the job: ready with chunks ({time.time() - t0:.0f}s)", doc.get("status") == "ready" and doc.get("chunk_count", 0) > 0, str(doc))
    check("attempts == 1 (no duplicate processing)", doc.get("attempts") == 1, str(doc.get("attempts")))
    check("same bytes again -> 409 duplicate", a.post("/documents", params={"chunk_size": 200, "chunk_overlap": 20}, files={"file": ("atlas.txt", TEXT.encode())}).status_code == 409)

    print("\n-- grounded Q&A with the real LLM --")
    r = a.post("/ask", json={"query": "Who approved Project Atlas?"})
    check("/ask returns 200", r.status_code == 200, r.text[:300])
    body = r.json() if r.status_code == 200 else {}
    check("retrieved the document", body.get("retrieved_count", 0) >= 1)
    check("answer has verified citations to atlas.txt", bool(body.get("citations")) and body["citations"][0]["filename"] == "atlas.txt" and body.get("all_citations_valid") is True, str(body)[:300])
    warn("answer mentions Dana Whitfield", "dana" in body.get("answer", "").lower(), body.get("answer", "")[:200])

    print("\n-- conversational RAG --")
    conv = a.post("/conversations", json={})
    check("conversation created", conv.status_code == 201)
    cid = conv.json()["id"]
    t1 = a.post(f"/conversations/{cid}/messages", json={"query": "Who approved Project Atlas?"})
    t2 = a.post(f"/conversations/{cid}/messages", json={"query": "and what is its budget?"})
    check("two turns answered", t1.status_code == 200 and t2.status_code == 200, (t1.text + t2.text)[:300])
    if t2.status_code == 200:
        warn("follow-up was rewritten to mention Atlas", "atlas" in t2.json()["standalone_query"].lower(), t2.json()["standalone_query"])
        warn("answer states the 2 million budget", "2" in t2.json()["answer"]["content"], t2.json()["answer"]["content"][:200])
    check("history persisted (4 messages)", len(a.get(f"/conversations/{cid}").json()["messages"]) == 4)

    print("\n-- tenant isolation --")
    b = signup(ARGS.base, "beta")
    check("B cannot read A's document (404)", b.get(f"/documents/{doc_id}").status_code == 404)
    rb = b.post("/ask", json={"query": "Who approved Project Atlas?"})
    check("B's question retrieves nothing from A", rb.status_code == 200 and rb.json()["retrieved_count"] == 0 and "Dana" not in rb.text, rb.text[:200])
    check("B cannot read A's conversation (404)", b.get(f"/conversations/{cid}").status_code == 404)

    print("\n-- RBAC --")
    rv = a.post("/users", json={"email": f"viewer-{uuid.uuid4().hex[:6]}@e2e.test", "password": "password-123", "role": "viewer"})
    check("owner created a viewer", rv.status_code == 201, rv.text[:200])
    v = httpx.Client(base_url=ARGS.base, timeout=60)
    v.headers["Authorization"] = "Bearer " + v.post("/auth/login", json={"email": rv.json()["email"], "password": "password-123"}).json()["access_token"]
    check("viewer may read documents", v.get("/documents").status_code == 200)
    check("viewer may NOT upload (403)", v.post("/documents", files={"file": ("x.txt", b"hello viewer")}).status_code == 403)
    check("viewer may NOT read the audit log (403)", v.get("/audit").status_code == 403)

    print("\n-- failing job and retry --")
    r = a.post("/documents", files={"file": ("broken.pdf", b"this is not really a pdf")})
    check("bad PDF accepted for processing (202)", r.status_code == 202)
    bad = wait_for(a, r.json()["id"], {"failed", "ready"})
    check("bad PDF ends failed with a message", bad.get("status") == "failed" and bool(bad.get("error_message")), str(bad))
    rr = a.post(f"/documents/{bad['id']}/retry")
    check("retry accepted (202)", rr.status_code == 202, rr.text[:200])
    again = wait_for(a, bad["id"], {"failed", "ready"})
    check("retry runs and fails again cleanly", again.get("status") == "failed" and again.get("attempts", 0) >= 2, str(again))

    print("\n-- quotas, audit, metrics --")
    u = a.get("/usage").json()
    check("usage counts queries and uploads", u["queries"]["used"] >= 3 and u["uploads"]["used"] >= 3, str(u))
    actions = {e["action"] for e in a.get("/audit", params={"limit": 100}).json()}
    check("audit trail has upload, ingest, failure and retry events", {"document.uploaded", "document.ingested", "document.ingest_failed", "document.retry_requested"} <= actions, str(sorted(actions)))
    m = httpx.get(ARGS.base + "/metrics", timeout=10)
    check("/metrics serves Prometheus text with route templates", m.status_code == 200 and "http_requests_total" in m.text and 'route="/documents/{document_id}"' in m.text)

    print(f"\n{'ALL CHECKS PASSED' if FAILS == 0 else str(FAILS) + ' CHECK(S) FAILED'}")
    sys.exit(1 if FAILS else 0)


main()
