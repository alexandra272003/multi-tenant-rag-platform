"""Pure-ASGI middleware: request id, access log, HTTP metrics. (Pure ASGI rather than
BaseHTTPMiddleware: it keeps contextvars and streaming responses behaving.)"""
import logging
import re
import time
import uuid

from app.core import metrics
from app.core.logging_setup import request_context

access_log = logging.getLogger("access")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ObservabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex  # never trust/echo arbitrary input
        ctx = {"request_id": request_id}
        token = request_context.set(ctx)
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            method, status = scope["method"], status_holder["status"]
            if route != "/metrics":
                metrics.HTTP_REQUESTS.labels(method, route, str(status)).inc()
                metrics.HTTP_LATENCY.labels(method, route).observe(elapsed)
            access_log.info(
                "%s %s -> %s", method, route, status,
                extra={"method": method, "route": route, "path": scope["path"], "status": status,
                       "duration_ms": round(elapsed * 1000, 1)},
            )
            request_context.reset(token)
