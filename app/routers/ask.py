from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.limits import guard_query
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.schemas import AskRequest, AskResponse, Citation
from app.services import generation_service

router = APIRouter(tags=["generation"])


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """
    Day 31: the full pipeline end to end -- retrieve top-k chunks, build a
    numbered prompt with citation instructions, generate an answer, and
    verify the model's citations actually refer to chunks that were really
    retrieved. See generation_service.ask for the citation-verification
    logic and the deterministic refusal path when nothing is retrieved.
    """
    result = await generation_service.ask(
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
    )
