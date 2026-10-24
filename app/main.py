from fastapi import FastAPI

from app.core import tenancy as _tenancy  # noqa: F401  (importing registers the per-transaction tenant-context hook)
from app.core.errors import AppError, app_error_handler
from app.core.logging_setup import configure_logging
from app.core.middleware import ObservabilityMiddleware
from app.routers import (
    agent, ask, ask_langchain, ask_structured, audit, auth, conversations, documents, lab, ops, retrieval, users,
)

configure_logging()

app = FastAPI(title="RAG Pipeline")
app.add_middleware(ObservabilityMiddleware)

app.add_exception_handler(AppError, app_error_handler)

app.include_router(lab.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(audit.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(ask.router)
app.include_router(ask_langchain.router)
app.include_router(ask_structured.router)
app.include_router(agent.router)
app.include_router(conversations.router)
app.include_router(ops.router)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
