from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.limits import guard_query
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.langchain_port.structured import ask_structured
from app.schemas import AskRequest, Citation, StructuredAskResponse, Usage

router = APIRouter(tags=["generation-structured"])


@router.post("/ask/structured", response_model=StructuredAskResponse)
async def ask_structured_endpoint(
    payload: AskRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """
    Day 37 (Week 6): same retrieval and grounding rules as /ask/langchain,
    but the model must return a validated JSON object (`GroundedAnswer`)
    instead of free text we regex for [N]. Invalid output is fed back to
    the model and retried; if it never complies the response is a 502
    `invalid_model_output`, never a silently-guessed answer.
    """
    result = await ask_structured(
        session,
        payload.query,
        access=AccessFilter.from_principal(principal),
        top_k=payload.top_k,
        document_id=payload.document_id,
        source_type=payload.source_type,
    )
    return StructuredAskResponse(
        query=payload.query,
        answerable=result["answerable"],
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        retrieved_count=result["retrieved_count"],
        attempts=result["attempts"],
        validation_errors=result["validation_errors"],
        usage=Usage(**result["usage"]),
    )
