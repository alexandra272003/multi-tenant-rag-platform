"""
Structured logging: one JSON object per line, every line carrying the request id and, once the
caller is authenticated, tenant_id and user_id. That is what makes "show me everything tenant 7
did between 14:02 and 14:03" a one-line query in any log system.

The request context lives in a ContextVar holding a MUTABLE dict. The middleware creates the dict
and the auth dependency fills in tenant/user later: a plain ContextVar.set() inside a dependency
would not be visible to the middleware's own log line (they run in different tasks).
"""
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from app.core.config import settings

request_context: ContextVar[dict | None] = ContextVar("request_context", default=None)

_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


def bind(**fields) -> None:
    ctx = request_context.get()
    if ctx is not None:
        ctx.update(fields)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(request_context.get() or {})
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    if settings.log_format != "json":
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers[:] = [handler]
    for noisy in ("uvicorn", "uvicorn.error"):
        logging.getLogger(noisy).handlers[:] = []
        logging.getLogger(noisy).propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our own access line
