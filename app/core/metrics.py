"""
Prometheus metrics. Rules followed:
  * labels have BOUNDED cardinality: the route TEMPLATE ("/documents/{document_id}"), never the raw
    path, and never tenant_id or user_id (one time series per tenant would grow without limit).
    Per-tenant numbers come from /usage and from the logs, which are built for high cardinality.
  * metrics are per PROCESS. The API and the Celery worker are different processes with different
    registries; this module exposes the API's. Worker metrics need their own endpoint or a
    pushgateway -- noted in docs/WEEK7_OPERATIONS.md.
"""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by the rate limiter", ["scope"])
RATE_LIMITER_ERRORS = Counter("rate_limiter_errors_total", "Rate limiter backend failures (failed open)")
QUOTA_REJECTED = Counter("quota_rejected_total", "Requests rejected because a daily quota was spent", ["metric"])
INGEST_JOBS = Counter("ingest_jobs_total", "Ingestion job outcomes (this process)", ["result"])


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
