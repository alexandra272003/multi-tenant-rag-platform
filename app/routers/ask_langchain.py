from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.limits import guard_query
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.langchain_port.chain import ask_with_langchain
from app.schemas import AskRequest, AskResponse, Citation, Usage

router = APIRouter(tags=["generation-langchain"])


@router.post("/ask/langchain", response_model=AskResponse)
async def ask_langchain(payload: AskRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """
    Day 35 (Week 6): the identical /ask request and response contract,
    answered by the LangChain-based chain (app/langchain_port/) instead
    of the hand-built generation_service.ask() (Day 31). Send the same
    question to both endpoints to compare retrieval, prompt, and
    citation behavior directly -- the orchestration differs, the
    external contract does not.
    """
    result = await ask_with_langchain(
        session,
        payload.query,
        access=AccessFilter.from_principal(principal),
        top_k=payload.top_k,
        document_id=payload.document_id,
        source_type=payload.source_type,
    )
    return AskResponse(
        query=payload.query,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        all_citations_valid=result["all_citations_valid"],
        retrieved_count=result["retrieved_count"],
        usage=Usage(**result["usage"]),
    )
