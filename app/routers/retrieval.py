from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.deps import require
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult
from app.services import retrieval_service

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    payload: RetrievalRequest,
    principal: Principal = Depends(require(Permission.QUERY)),
    session: AsyncSession = Depends(get_session),
):
    """
    Day 30: top-k retrieval + metadata filters. Embeds `payload.query` and
    returns the top_k chunks with the smallest cosine distance, optionally
    narrowed to one document_id and/or source_type.

    Deliberately no LLM call here -- that's Day 31/32 (grounded answer
    with citations). Keeping retrieval as its own endpoint means its
    quality (does the right chunk come back at all?) can be evaluated
    independently of answer quality (did the LLM use it well?).
    """
    results = await retrieval_service.retrieve(
        session,
        payload.query,
        access=AccessFilter.from_principal(principal),
        top_k=payload.top_k,
        document_id=payload.document_id,
        source_type=payload.source_type,
    )
    return RetrievalResponse(
        query=payload.query,
        results=[
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                distance=distance,
            )
            for chunk, filename, distance in results
        ],
    )
