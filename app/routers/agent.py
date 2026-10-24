import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import run_agent
from app.agent.scope import AgentScope
from app.agent.tools import TOOLS
from app.auth.access import AccessFilter
from app.auth.limits import guard_query
from app.auth.rbac import Permission, Principal
from app.core.config import settings
from app.core.db import get_session
from app.core.errors import AppError
from app.models import Document
from app.schemas import AgentRequest, AgentResponse, Citation

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)


async def _scope_from(
    payload: AgentRequest, session: AsyncSession, access: AccessFilter
) -> AgentScope:
    """
    The agent's document scope is what the AUTHENTICATED CALLER may read,
    computed server-side, narrowed (never widened) by the request's optional
    `document_ids`. This closes the gap noted in the Week 6 threat model:
    scope used to come straight from the request body, which was not a
    security boundary. Now an empty/forged list can only shrink the scope,
    and the scope is never "None = everything" -- it is always an explicit
    set of ids this user is allowed to see (fail closed).
    """
    readable = frozenset(
        (await session.execute(select(Document.id).where(access.document_clause()))).scalars().all()
    )
    if payload.document_ids is not None:
        readable = readable & frozenset(payload.document_ids)
    return AgentScope(
        document_ids=readable,
        allowed_tools=frozenset(payload.allowed_tools if payload.allowed_tools is not None else TOOLS),
        max_steps=payload.max_steps,
        max_tool_calls=settings.agent_max_tool_calls,
        max_calls_per_step=settings.agent_max_calls_per_step,
        max_tool_output_chars=settings.agent_max_tool_output_chars,
    )


@router.post("/agent", response_model=AgentResponse)
async def agent(
    payload: AgentRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """
    Tool-calling agent, non-streaming: runs the whole loop and returns the
    answer plus the trace of every tool call. Same loop as /agent/stream.
    """
    access = AccessFilter.from_principal(principal)
    scope = await _scope_from(payload, session, access)
    events = [e async for e in run_agent(session, payload.query, scope, access)]
    final = events[-1]
    return AgentResponse(
        query=payload.query,
        answer=final["answer"],
        citations=[Citation(**c) for c in final["citations"]],
        all_citations_valid=final["all_citations_valid"],
        steps=final["steps"],
        tool_calls_used=final["tool_calls_used"],
        budget_exhausted=final["budget_exhausted"],
        usage=final["usage"],
        trace=[e for e in events if e["event"] not in ("token", "final")],
    )


def _sse(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event, default=str)}\n\n"


@router.post("/agent/stream")
async def agent_stream(
    payload: AgentRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """
    Same agent, as Server-Sent Events. Event order for a normal run:
    (tool_call, tool_result|tool_denied)* then token* then final then done.
    On failure an `error` event is sent instead of `done` -- by then the
    HTTP status is already 200, so errors can only travel in-band.
    A client that disconnects cancels the generator; the finally block
    always releases the DB session.
    """
    access = AccessFilter.from_principal(principal)
    scope = await _scope_from(payload, session, access)

    async def events():
        try:
            async for event in run_agent(session, payload.query, scope, access):
                yield _sse(event)
            yield _sse({"event": "done"})
        except AppError as exc:
            yield _sse({"event": "error", "code": exc.code, "message": exc.message})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("agent stream crashed")
            yield _sse({"event": "error", "code": "internal_error", "message": "internal error"})
        finally:
            await session.close()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
