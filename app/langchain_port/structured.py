"""
Day 37 (Week 6): structured output + retry on invalid model output.

Problem: chain.py asks for free text and regexes `[N]` out of it. That
works until it doesn't -- an LLM is a probabilistic text generator, not a
function with a type signature. This module makes the contract explicit:
the model must return a `GroundedAnswer` (schemas.py), and nothing
downstream touches the output until it has been validated.

Three layers of defence, in order of how much they can be trusted:

  1. Constrained decoding (`response_format=json_schema` bound onto the
     model): the provider steers generation toward the schema. Cuts invalid
     output sharply but is provider/model dependent and does NOT
     guarantee semantic correctness.
  2. Schema validation (Pydantic): types, ranges, and internal
     consistency (answerable=true must cite something).
  3. Semantic validation (`_semantic_problems` below): checks that need
     request context -- every cited number must exist in the retrieved
     set. A schema cannot know how many chunks were retrieved.

If 2 or 3 fails, the bad output AND the exact error are sent back to the
model as extra messages and it is asked again, up to
`settings.llm_validation_retries` times. Then we fail loudly with a 502
(`invalid_model_output`) rather than returning a guess.

Design decisions worth defending out loud:
  - Retry feeds back the error. Resending the identical prompt at
    temperature > 0 is a re-roll; telling the model what was wrong is a
    correction. Only the LAST failed attempt is fed back, so the prompt
    cannot grow without bound.
  - Two retry layers on purpose. ChatOpenAI(max_retries) handles
    transport (timeout/429/5xx). This loop handles "HTTP 200 but wrong
    content". Auth errors and bad requests are never retried here: an
    APIError from the provider goes straight to ProviderError.
  - Refusals are normalized to NO_DOCUMENTS_ANSWER so /ask, /ask/langchain
    and /ask/structured refuse with the same text.
  - We bind `response_format` and parse the reply OURSELVES instead of
    using `llm.with_structured_output(...)`. Found by running the real
    LangChain/OpenAI-SDK path against a mocked HTTP server: with
    method="json_schema" the SDK parses inside the call and RAISES on bad
    output, even with include_raw=True -- so the bad text never reaches
    the retry loop and a malformed reply becomes a 500. Owning the parse
    step means every failure mode lands in one place with the raw text
    intact. (Trade-off: ~10 lines of our own parsing instead of a
    framework one-liner -- and we can explain every line.)
  - json_schema support varies by Groq model. The method is a setting
    (`llm_structured_method`: json_schema | json_mode | none); the
    validation loop keeps the endpoint correct even if the provider
    ignores the schema -- it just costs more retries. NOT verified
    against live Groq from this sandbox (no network): do that first.
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import APIError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.core.config import settings
from app.core.errors import InvalidModelOutputError, ProviderError
from app.langchain_port.chain import _format_docs, llm
from app.langchain_port.retriever import PgVectorRetriever
from app.langchain_port.usage import add_message_usage, new_usage
from app.schemas import GroundedAnswer
from app.services.generation_service import NO_DOCUMENTS_ANSWER, SYSTEM_PROMPT

# No curly braces in here: this string is part of a prompt TEMPLATE, and
# `{`/`}` would be parsed as variables.
STRUCTURED_SUFFIX = (
    " Respond with a single JSON object with exactly these fields: "
    "answerable (boolean: true only if the context contains the answer), "
    "answer (string), and cited_sources (list of integer source numbers "
    "that support the answer; empty when answerable is false). "
    "cited_sources is authoritative. Output JSON only, no prose around it."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + STRUCTURED_SUFFIX),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
        # Empty on the first attempt; on a retry holds [bad output, error].
        MessagesPlaceholder("feedback", optional=True),
    ]
)

def _bind_kwargs() -> dict:
    method = settings.llm_structured_method
    if method == "json_schema":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "GroundedAnswer",
                    "schema": GroundedAnswer.model_json_schema(),
                    "strict": False,
                },
            }
        }
    if method == "json_mode":
        return {"response_format": {"type": "json_object"}}
    if method == "none":
        return {}  # prompt-only: rely entirely on validate-and-retry
    raise ValueError(f"unknown llm_structured_method: {method!r}")


# Returns a plain AIMessage (no framework parsing). Module-level so tests
# can replace it wholesale (same pattern as chain._chain -- see
# FakeLangChainChain in tests/conftest.py).
_structured_chain = _prompt | llm.bind(**_bind_kwargs())

_MAX_ECHO_CHARS = 2000  # cap how much bad output is echoed back (cost + injection surface)


def _describe_error(exc: ValidationError) -> str:
    """Short, model-readable description of why validation failed."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "output"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _semantic_problems(parsed: GroundedAnswer, n_docs: int) -> list[str]:
    problems = []
    out_of_range = [n for n in parsed.cited_sources if n > n_docs]
    if out_of_range:
        problems.append(
            f"cited_sources {out_of_range} do not exist; valid source numbers are 1..{n_docs}"
        )
    return problems


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _parse(message) -> tuple[GroundedAnswer | None, str, str | None]:
    """
    -> (parsed or None, raw text, problem description or None).
    Accepts an AIMessage or a plain string. Tolerates a ```json fence --
    a formatting habit, not a contract violation worth a retry.
    """
    text = message if isinstance(message, str) else (getattr(message, "content", None) or "")
    if not isinstance(text, str):  # content-block lists: stringify rather than crash
        text = json.dumps(text, default=str)
    raw_text = text[:_MAX_ECHO_CHARS]
    candidate = text.strip()
    fenced = _FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    if not candidate:
        return None, raw_text, "model returned an empty reply"
    try:
        return GroundedAnswer.model_validate_json(candidate), raw_text, None
    except ValidationError as exc:
        return None, raw_text, _describe_error(exc)


async def _generate_validated(
    inputs: dict, n_docs: int, usage: dict | None = None
) -> tuple[GroundedAnswer, int, list[str]]:
    """
    Returns (validated answer, attempts used, validation errors of the
    failed attempts). Raises InvalidModelOutputError when retries run out.

    `usage`, if given, is summed IN PLACE across every attempt -- retries
    are real spend, so a request that needed 3 attempts reports 3 calls
    and ~3x the tokens. (If every attempt fails the endpoint returns 502 and no
    usage; that spend is real but is not reported -- see the memo's caveats.)
    """
    feedback: list = []
    failures: list[str] = []
    max_attempts = 1 + settings.llm_validation_retries

    for attempt in range(1, max_attempts + 1):
        message = await _structured_chain.ainvoke({**inputs, "feedback": feedback})
        if usage is not None:
            add_message_usage(usage, message)
        parsed, raw_text, problem = _parse(message)

        if parsed is not None:
            semantic = _semantic_problems(parsed, n_docs)
            if semantic:
                problem = "; ".join(semantic)

        if problem is None:
            return parsed, attempt, failures

        failures.append(problem)
        # Replace (not append to) feedback: bounded prompt growth.
        feedback = [
            AIMessage(content=raw_text or "(empty)"),
            HumanMessage(
                content=(
                    f"Your previous reply was invalid: {problem}. "
                    "Reply again with a corrected JSON object only."
                )
            ),
        ]

    raise InvalidModelOutputError(
        "The model did not return a valid structured answer",
        details={"attempts": max_attempts, "errors": failures},
    )


async def ask_structured(
    session: AsyncSession,
    query: str,
    *,
    access: AccessFilter,
    top_k: int = 5,
    document_id: int | None = None,
    source_type: str | None = None,
) -> dict:
    retriever = PgVectorRetriever(
        session=session, access=access, top_k=top_k, document_id=document_id, source_type=source_type
    )
    docs = await retriever.ainvoke(query)

    if not docs:
        # Same deterministic refusal as /ask: no evidence, no LLM call.
        return {
            "answerable": False,
            "answer": NO_DOCUMENTS_ANSWER,
            "citations": [],
            "retrieved_count": 0,
            "attempts": 0,
            "validation_errors": [],
            "usage": new_usage(),  # the LLM was never called
        }

    usage = new_usage()
    try:
        parsed, attempts, failures = await _generate_validated(
            {"context": _format_docs(docs), "question": query}, n_docs=len(docs), usage=usage
        )
    except APIError as exc:
        raise ProviderError(
            "The language model provider failed to respond", details={"reason": str(exc)}
        ) from exc

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
        if i in parsed.cited_sources
    ]

    return {
        "answerable": parsed.answerable,
        "answer": parsed.answer if parsed.answerable else NO_DOCUMENTS_ANSWER,
        "citations": citations,
        "retrieved_count": len(docs),
        "attempts": attempts,
        "validation_errors": failures,
        "usage": usage,
    }
