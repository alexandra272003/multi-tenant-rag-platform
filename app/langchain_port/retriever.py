"""
Day 35 (Week 6): port the Day 30 retrieval pipeline behind LangChain's
BaseRetriever interface.

Nothing about retrieval itself is reimplemented here -- this is purely an
adapter. The same pgvector cosine-distance query, the same
`ivfflat.probes` fix, the same metadata filters applied before ranking
(see app/services/retrieval_service.py) all run exactly as before. What
changes is the shape callers see: a LangChain chain can now call
`.ainvoke(query)` on this object the same way it would call any other
LangChain retriever, including a hosted vector-store-backed one --
which is the actual point of this port. The interview question this
answers directly: "what did LangChain's retriever abstraction replace?"
Here, nothing structural -- it's a thin adapter over code that already
existed, which is itself worth saying plainly.
"""
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.services import retrieval_service


class PgVectorRetriever(BaseRetriever):
    """One instance per request -- `session` is request-scoped (FastAPI's
    get_session dependency), so this is built fresh per call, not shared
    as a module-level singleton the way `llm` and `prompt` are in chain.py."""

    model_config = {"arbitrary_types_allowed": True}

    session: AsyncSession
    # Week 7: required. Pydantic field with no default -> constructing a
    # retriever without an AccessFilter raises, so tenant scoping can't be skipped.
    access: AccessFilter
    top_k: int = 5
    document_id: int | None = None
    source_type: str | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError(
            "This project's retrieval is async-only (async SQLAlchemy session, "
            "async embedding call) -- use ainvoke()/_aget_relevant_documents, "
            "never the sync path."
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = await retrieval_service.retrieve(
            self.session,
            query,
            access=self.access,
            top_k=self.top_k,
            document_id=self.document_id,
            source_type=self.source_type,
        )
        return [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "distance": distance,
                },
            )
            for chunk, filename, distance in results
        ]
