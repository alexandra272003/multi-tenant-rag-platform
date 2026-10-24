"""
Day 35 (Week 6): the hand-built generation_service.ask() (Day 31),
rebuilt on LangChain's prompt + LLM + LCEL-chain abstractions instead of
a plain async function calling a hand-rolled retry wrapper.

Reused as-is from generation_service.py, not duplicated: SYSTEM_PROMPT
(the exact same grounding/refusal/citation instructions), CITATION_RE,
and NO_DOCUMENTS_ANSWER. The comparison this sets up is deliberate: same
prompt text, same citation-verification rule, same /ask response shape
-- only the orchestration layer differs. `_format_docs` below IS a small,
honest duplication of generation_service._build_context_block: the input
type differs (LangChain Document vs. a raw (Chunk, filename, distance)
tuple), and four lines didn't seem worth a shared abstraction across
both. Named here rather than hidden.

What LangChain replaces in THIS file, concretely:
  - ChatOpenAI replaces core/llm_client.py's _with_retries() wrapper --
    LangChain's client has its own retry/timeout handling built in
    (max_retries, timeout kwargs below), so the hand-rolled exponential
    backoff loop isn't needed here. Trade-off: that retry logic is now
    inside a dependency instead of in this repo, readable and testable
    only through LangChain's own code, not ours.
  - ChatPromptTemplate replaces the plain f-string message-building in
    generation_service.ask() -- for a two-message prompt like this one,
    it's a wash in lines of code; it starts earning its keep with
    multi-turn conversation history or few-shot examples, neither of
    which this project needs yet.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter

from app.core.config import settings
from app.core.errors import ProviderError
from app.langchain_port.retriever import PgVectorRetriever
from app.langchain_port.usage import add_message_usage, new_usage
from app.services.generation_service import CITATION_RE, NO_DOCUMENTS_ANSWER, SYSTEM_PROMPT

llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    timeout=settings.llm_timeout_seconds,
    max_retries=settings.llm_max_retries,
)

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}")]
)

# Module-level so tests can monkeypatch `_chain.ainvoke` directly, the
# same pattern used throughout this project for mocking a provider call
# (see tests/conftest.py's mock_llm, mock_embeddings).
_chain = prompt | llm


def _format_docs(docs: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        page_number = doc.metadata.get("page_number")
        page = f", page {page_number}" if page_number is not None else ""
        lines.append(f"[{i}] ({doc.metadata['filename']}{page})\n{doc.page_content}")
    return "\n\n".join(lines)


async def ask_with_langchain(
    session: AsyncSession,
    query: str,
    *,
    access: AccessFilter,
    top_k: int = 5,
    document_id: int | None = None,
    source_type: str | None = None,
) -> dict:
    """
    Same contract as generation_service.ask(): retrieve, build a
    numbered+cited prompt, generate, then verify every citation against
    what was actually retrieved -- citations are CHECKED here too, not
    trusted just because this path goes through a framework.
    """
    retriever = PgVectorRetriever(
        session=session, access=access, top_k=top_k, document_id=document_id, source_type=source_type
    )
    docs = await retriever.ainvoke(query)

    if not docs:
        return {
            "answer": NO_DOCUMENTS_ANSWER,
            "citations": [],
            "all_citations_valid": True,
            "retrieved_count": 0,
            "usage": new_usage(),  # the LLM was never called
        }

    try:
        response = await _chain.ainvoke({"context": _format_docs(docs), "question": query})
    except APIError as exc:
        raise ProviderError(
            "The language model provider failed to respond", details={"reason": str(exc)}
        ) from exc

    # ChatOpenAI returns an AIMessage; plain function calls into chain.py
    # (e.g. from tests, via a monkeypatched fake ainvoke) may return a
    # plain string instead -- handle both rather than assuming one shape.
    answer = response.content if hasattr(response, "content") else response
    usage = new_usage()
    add_message_usage(usage, response)

    cited_numbers = {int(n) for n in CITATION_RE.findall(answer)}
    valid_numbers = set(range(1, len(docs) + 1))
    all_valid = cited_numbers.issubset(valid_numbers)

    citations = [
        {
            "number": i,
            "chunk_id": doc.metadata["chunk_id"],
            "document_id": doc.metadata["document_id"],
            "filename": doc.metadata["filename"],
            "page_number": doc.metadata["page_number"],
            "distance": doc.metadata["distance"],
        }
        for i, doc in enumerate(docs, start=1)
        if i in cited_numbers
    ]

    return {
        "answer": answer,
        "citations": citations,
        "all_citations_valid": all_valid,
        "retrieved_count": len(docs),
        "usage": usage,
    }
