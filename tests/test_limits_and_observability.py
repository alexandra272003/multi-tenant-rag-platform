"""Week 7, Saturday: quotas, rate limits, metrics, structured logs."""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from app.core import ratelimit
from app.core.logging_setup import JsonFormatter, request_context
from app.models import Tenant
from app.services import quota_service
from tests.tenancy_helpers import add_user, upload, upload_ok

DOC = "Quota test document about Project Atlas."


@pytest.fixture
def set_limits(monkeypatch):
    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(f"app.core.config.settings.{k}", v)
    return _set


async def _ask(c, q="atlas?"):
    return await c.post("/ask", json={"query": q})


# ---- daily quota --------------------------------------------------------------------------


async def test_query_quota_admits_up_to_the_limit_then_429s_with_details(client, set_limits):
    set_limits(daily_query_limit=3, rate_limit_per_minute=1000)
    await upload_ok(client, "a.txt", DOC)
    assert [(await _ask(client)).status_code for _ in range(3)] == [200, 200, 200]
    r = await _ask(client)
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "quota_exceeded"
    assert err["details"]["metric"] == "queries" and err["details"]["limit"] == 3 and err["details"]["used"] == 3
    assert int(r.headers["retry-after"]) >= 1


async def test_rejected_requests_do_not_consume_quota(client, set_limits):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    await _ask(client)
    for _ in range(3):
        assert (await _ask(client)).status_code == 429
    usage = (await client.get("/usage")).json()
    assert usage["queries"] == {"used": 1, "limit": 1, "remaining": 0}


async def test_every_llm_endpoint_draws_on_the_same_query_quota(client, set_limits):
    set_limits(daily_query_limit=2, rate_limit_per_minute=1000)
    await upload_ok(client, "a.txt", DOC)
    conv = (await client.post("/conversations", json={})).json()["id"]
    assert (await _ask(client)).status_code == 200
    assert (await client.post(f"/conversations/{conv}/messages", json={"query": "atlas?"})).status_code == 200
    assert (await client.post("/agent", json={"query": "q"})).status_code == 429
    assert (await client.post("/ask/langchain", json={"query": "q"})).status_code == 429


async def test_non_llm_endpoints_are_not_counted(client, set_limits):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    await upload_ok(client, "a.txt", DOC)
    for _ in range(5):
        assert (await client.post("/retrieve", json={"query": "atlas"})).status_code == 200
        assert (await client.get("/documents")).status_code == 200
    assert (await client.get("/usage")).json()["queries"]["used"] == 0


async def test_upload_quota(client, set_limits):
    set_limits(daily_upload_limit=2, upload_rate_limit_per_minute=1000)
    assert (await upload(client, "a.txt", "first document text")).status_code == 202
    assert (await upload(client, "b.txt", "second document text")).status_code == 202
    r = await upload(client, "c.txt", "third document text")
    assert r.status_code == 429 and r.json()["error"]["details"]["metric"] == "uploads"
    assert len((await client.get("/documents")).json()) == 2  # nothing was created


async def test_quotas_are_per_tenant(client, make_client, set_limits):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    b = await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})
    assert (await _ask(client)).status_code == 200
    assert (await _ask(client)).status_code == 429
    assert (await _ask(b)).status_code == 200  # B is unaffected by A's spend


async def test_tenant_override_beats_the_platform_default(client, session, set_limits):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    me = (await client.get("/me")).json()
    await session.execute(update(Tenant).where(Tenant.id == me["tenant_id"]).values(daily_query_limit=3))
    await session.commit()
    assert [(await _ask(client)).status_code for _ in range(4)] == [200, 200, 200, 429]
    assert (await client.get("/usage")).json()["queries"]["limit"] == 3


async def test_a_limit_of_zero_blocks_the_tenant(client, set_limits):
    set_limits(daily_query_limit=0, rate_limit_per_minute=1000)
    assert (await _ask(client)).status_code == 429


async def test_quota_resets_on_the_next_utc_day(client, set_limits, monkeypatch):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    assert (await _ask(client)).status_code == 200
    assert (await _ask(client)).status_code == 429
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(quota_service, "_utcnow", lambda: tomorrow)
    assert (await _ask(client)).status_code == 200


async def test_concurrent_requests_never_exceed_the_quota(client, set_limits):
    """The point of the conditional UPDATE: 12 simultaneous requests, limit 5 -> exactly 5 admitted."""
    set_limits(daily_query_limit=5, rate_limit_per_minute=1000)
    await upload_ok(client, "a.txt", DOC)
    codes = [r.status_code for r in await asyncio.gather(*[_ask(client) for _ in range(12)])]
    assert codes.count(200) == 5 and codes.count(429) == 7
    assert (await client.get("/usage")).json()["queries"]["used"] == 5


def test_seconds_until_reset():
    now = datetime(2026, 10, 24, 23, 0, 0, tzinfo=timezone.utc)
    assert quota_service.seconds_until_reset(now) == 3600


# ---- /usage ---------------------------------------------------------------------------------------


async def test_usage_report_shape_and_permissions(client, make_client):
    usage = (await client.get("/usage")).json()
    assert set(usage) >= {"day", "resets_at", "queries", "uploads", "rate_limit_per_minute"}
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    assert (await viewer.get("/usage")).status_code == 403
    assert (await (await make_client()).get("/usage")).status_code == 401


# ---- rate limiting ----------------------------------------------------------------------------------


async def test_rate_limit_429s_with_retry_after_and_is_distinct_from_quota(client, set_limits):
    set_limits(rate_limit_per_minute=2, daily_query_limit=1000)
    codes = [(await _ask(client)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = await _ask(client)
    assert r.json()["error"]["code"] == "rate_limited" and 1 <= int(r.headers["retry-after"]) <= 61


async def test_rate_limited_requests_do_not_consume_daily_quota(client, set_limits):
    set_limits(rate_limit_per_minute=1, daily_query_limit=1000)
    for _ in range(4):
        await _ask(client)
    assert (await client.get("/usage")).json()["queries"]["used"] == 1


async def test_rate_limit_is_per_tenant(client, make_client, set_limits):
    set_limits(rate_limit_per_minute=1, daily_query_limit=1000)
    b = await make_client(register={"tenant_name": "Tenant B", "email": "owner@b.test"})
    await _ask(client)
    assert (await _ask(client)).status_code == 429
    assert (await _ask(b)).status_code == 200


async def test_rate_limiter_can_be_switched_off(client, set_limits):
    set_limits(rate_limit_per_minute=1, rate_limit_backend="off", daily_query_limit=1000)
    assert [(await _ask(client)).status_code for _ in range(3)] == [200, 200, 200]


async def test_forbidden_requests_are_rejected_before_any_counting(client, make_client, set_limits):
    set_limits(rate_limit_per_minute=1, upload_rate_limit_per_minute=1, daily_upload_limit=1)
    viewer = await add_user(client, make_client, "v@t.test", "viewer")
    for _ in range(3):
        assert (await upload(viewer, "x.txt", "text")).status_code == 403
    assert (await upload(client, "a.txt", "owner document text")).status_code == 202  # budget untouched


async def test_memory_limiter_windows():
    clock = {"t": 1000.0}
    lim = ratelimit.MemoryLimiter(clock=lambda: clock["t"])
    assert [(await lim.hit("k", 2, 60))[0] for _ in range(3)] == [True, True, False]
    assert (await lim.hit("other", 2, 60))[0] is True  # keys are independent
    clock["t"] += 60
    assert (await lim.hit("k", 2, 60))[0] is True  # new window


async def test_redis_limiter_fails_open(monkeypatch):
    from app.core import metrics
    lim = ratelimit.RedisLimiter("redis://127.0.0.1:1/0")  # nothing listens here
    before = metrics.RATE_LIMITER_ERRORS._value.get()
    allowed, _ = await lim.hit("k", 1, 60)
    assert allowed is True and metrics.RATE_LIMITER_ERRORS._value.get() == before + 1


# ---- metrics ------------------------------------------------------------------------------------------


def _sample(text: str, name: str, **labels) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def test_metrics_use_route_templates_not_raw_paths(client):
    doc = await upload_ok(client, "a.txt", DOC)
    for _ in range(2):
        await client.get(f"/documents/{doc}")
    await client.get("/no/such/route/12345")
    text = (await client.get("/metrics")).text
    assert _sample(text, "http_requests_total", route="/documents/{document_id}", method="GET", status="200") >= 2
    assert f'/documents/{doc}"' not in text  # bounded cardinality
    assert 'route="unmatched"' in text and "/no/such/route" not in text
    assert "http_request_duration_seconds_bucket" in text


async def test_metrics_count_rejections_and_have_no_tenant_labels(client, set_limits):
    set_limits(daily_query_limit=1, rate_limit_per_minute=1000)
    before = _sample((await client.get("/metrics")).text, "quota_rejected_total", metric="queries")
    await _ask(client); await _ask(client)
    text = (await client.get("/metrics")).text
    assert _sample(text, "quota_rejected_total", metric="queries") == before + 1
    assert "tenant" not in text.split("# HELP http_requests_total")[1].split("# HELP rate_limited_total")[0]


async def test_ingestion_outcomes_are_counted(client):
    before = _sample((await client.get("/metrics")).text, "ingest_jobs_total", result="ready")
    await upload_ok(client, "a.txt", DOC)
    assert _sample((await client.get("/metrics")).text, "ingest_jobs_total", result="ready") == before + 1


async def test_metrics_token_protects_the_endpoint(make_client, set_limits):
    set_limits(metrics_token="s3cret")
    anon = await make_client()
    assert (await anon.get("/metrics")).status_code == 401
    assert (await anon.get("/metrics", headers={"Authorization": "Bearer wrong"})).status_code == 401
    assert (await anon.get("/metrics", headers={"Authorization": "Bearer s3cret"})).status_code == 200


# ---- structured logs / request ids --------------------------------------------------------------------------


async def test_request_id_is_generated_and_echoed(client):
    r = await client.get("/ping")
    assert len(r.headers["x-request-id"]) == 32


async def test_valid_incoming_request_id_is_kept_and_junk_is_replaced(client):
    assert (await client.get("/ping", headers={"X-Request-ID": "trace-abc.123"})).headers["x-request-id"] == "trace-abc.123"
    junk = (await client.get("/ping", headers={"X-Request-ID": "bad id\twith spaces & <script>"})).headers["x-request-id"]
    assert "script" not in junk and len(junk) == 32


async def test_access_log_carries_request_tenant_and_user(client, caplog):
    me = (await client.get("/me")).json()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="access"):
        r = await client.get("/documents")
    rec = next(x for x in caplog.records if x.name == "access" and x.route == "/documents")
    assert rec.status == 200 and rec.duration_ms >= 0 and rec.method == "GET"
    # tenant/user are bound into the request context by the auth dependency; the formatter reads them
    fmt = JsonFormatter()
    request_context.set({"request_id": r.headers["x-request-id"], "tenant_id": me["tenant_id"], "user_id": me["id"]})
    line = json.loads(fmt.format(rec))
    assert line["tenant_id"] == me["tenant_id"] and line["user_id"] == me["id"]
    assert line["route"] == "/documents" and line["status"] == 200 and line["level"] == "INFO"


async def test_auth_binds_tenant_and_user_into_the_live_request_context(client, monkeypatch):
    seen = {}
    from app.routers import documents as docs_router
    real = docs_router.repo.list_documents

    async def spy(*a, **kw):
        seen.update(request_context.get() or {})
        return await real(*a, **kw)

    monkeypatch.setattr(docs_router.repo, "list_documents", spy)
    me = (await client.get("/me")).json()
    await client.get("/documents")
    assert seen["tenant_id"] == me["tenant_id"] and seen["user_id"] == me["id"] and seen["request_id"]


def test_json_formatter_emits_one_parseable_line_with_extras_and_exceptions():
    rec = logging.LogRecord("x", logging.ERROR, "f.py", 1, "boom %s", ("now",), None)
    rec.custom_field = 42
    try:
        raise ValueError("bad")
    except ValueError:
        import sys
        rec.exc_info = sys.exc_info()
    request_context.set({"request_id": "r1"})
    out = JsonFormatter().format(rec)
    assert "\n" not in out.split('"exc"')[0]
    data = json.loads(out)
    assert data["message"] == "boom now" and data["custom_field"] == 42 and data["request_id"] == "r1"
    assert "ValueError" in data["exc"]


async def test_secrets_and_bodies_never_reach_the_access_log(client, caplog):
    with caplog.at_level(logging.INFO, logger="access"):
        await client.post("/auth/login", json={"email": "owner@default.test", "password": "password-123"})
    text = " ".join(str(r.__dict__) for r in caplog.records if r.name == "access")
    assert "password-123" not in text and "Bearer" not in text
