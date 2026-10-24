"""
Week 7, Friday: conversational RAG.

A follow-up like "and who approved it?" is meaningless to a vector index. The turn is
therefore handled in three steps:
  1. CONDENSE  - rewrite the follow-up into a standalone question using recent history
  2. RETRIEVE  - search with the standalone question, under the CALLER's access filter
                 as of NOW (not as of when earlier turns happened)
  3. GENERATE  - answer from the fresh context, with recent history as conversation

Security rules that shape this file:
  * Conversations are private to their author (404 for anyone else, admins included).
  * Retrieval never reuses earlier turns' chunks: a document made private or deleted since
    then must stop being quotable immediately. Only the *text* of earlier answers is replayed,
    and the caller wrote/received that text themselves.
  * Citation markers are stripped from replayed answers: "[2]" in turn 1 pointed at a different
    chunk than "[2]" in turn 5, so replaying them invites wrong citations.
"""
import logging

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.auth.rbac import Principal
from app.core.config import settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.models import Conversation, Message
from app.models.models import utcnow
from app.services import generation_service as gen
from app.services import retrieval_service

logger = logging.getLogger("chat")

CONDENSE_PROMPT = (
    "Rewrite the user's last message as ONE standalone question that can be understood without "
    "the conversation. Resolve pronouns and references using the history. Do not answer it. "
    "If it is already standalone, repeat it unchanged. Output only the question."
)
_MAX_STANDALONE = 500

CHAT_SYSTEM_PROMPT = (
    gen.SYSTEM_PROMPT
    + " Source numbers refer ONLY to the context in the final message; earlier turns are "
    "conversation history, not sources."
)


# ---- conversations (CRUD) ---------------------------------------------------------------


async def create_conversation(session: AsyncSession, principal: Principal, title: str | None) -> Conversation:
    conv = Conversation(tenant_id=principal.tenant_id, user_id=principal.user_id, title=title)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def get_conversation_or_404(session: AsyncSession, principal: Principal, conversation_id: int) -> Conversation:
    conv = (await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == principal.tenant_id,
            Conversation.user_id == principal.user_id,
        )
    )).scalar_one_or_none()
    if conv is None:
        raise NotFoundError("Conversation not found")
    return conv


async def list_conversations(session: AsyncSession, principal: Principal) -> list[Conversation]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.tenant_id == principal.tenant_id, Conversation.user_id == principal.user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    return list(result.scalars().all())


async def list_messages(session: AsyncSession, principal: Principal, conversation_id: int) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.tenant_id == principal.tenant_id)
        .order_by(Message.id)
    )
    return list(result.scalars().all())


async def delete_conversation(session: AsyncSession, principal: Principal, conversation_id: int) -> None:
    conv = await get_conversation_or_404(session, principal, conversation_id)
    await session.execute(delete(Message).where(Message.conversation_id == conv.id))
    await session.delete(conv)
    await session.commit()


# ---- history ------------------------------------------------------------------------------


def window_history(messages: list[Message]) -> list[dict]:
    """The recent tail, newest-last, within both a message and a character budget. The newest
    messages win: when the budget runs out, the OLDEST are dropped."""
    tail = messages[-settings.chat_history_messages:]
    kept: list[dict] = []
    used = 0
    for m in reversed(tail):
        text = gen.CITATION_RE.sub("", m.content).strip()
        if m.role == "assistant":
            text = text.replace("  ", " ")
        if used + len(text) > settings.chat_history_chars and kept:
            break
        text = text[: settings.chat_history_chars]
        used += len(text)
        kept.append({"role": m.role, "content": text})
    kept.reverse()
    # a window must start with a user turn, or some providers reject the conversation
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


async def condense(history: list[dict], query: str) -> str:
    """Standalone version of `query`. Best effort: on any problem, search with what the user
    typed -- a worse retrieval query is better than a failed turn."""
    if not history:
        return query
    transcript = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history)
    messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": f"HISTORY:\n{transcript}\n\nLAST MESSAGE: {query}"},
    ]
    try:
        rewritten = (await gen.complete(messages)).strip().strip('"')
    except AppError:
        logger.warning("condense failed; using the raw query")
        return query
    if not rewritten or "\n" in rewritten or len(rewritten) > _MAX_STANDALONE:
        return query
    return rewritten


# ---- one turn --------------------------------------------------------------------------------


async def chat(
    session: AsyncSession, principal: Principal, conversation_id: int, query: str, *,
    top_k: int = 5, document_id: int | None = None,
) -> dict:
    conv = await get_conversation_or_404(session, principal, conversation_id)
    messages = await list_messages(session, principal, conv.id)
    if len(messages) >= settings.chat_max_messages:
        raise ConflictError("This conversation is full; start a new one", details={"max_messages": settings.chat_max_messages})

    history = window_history(messages)
    standalone = await condense(history, query)

    access = AccessFilter.from_principal(principal)
    results = await retrieval_service.retrieve(session, standalone, access=access, top_k=top_k, document_id=document_id)

    if not results:
        answer, citations, all_valid = gen.NO_DOCUMENTS_ANSWER, [], True
    else:
        prompt = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *history, {
            "role": "user",
            "content": f"CONTEXT:\n{gen._build_context_block(results)}\n\nQUESTION: {query}",
        }]
        answer = await gen.complete(prompt)  # raises before anything is saved
        citations, all_valid = gen.verify_citations(answer, results)

    # Both messages are saved together and only after the answer exists: a provider failure
    # leaves no dangling question in the history.
    user_msg = Message(tenant_id=conv.tenant_id, conversation_id=conv.id, role="user", content=query)
    session.add(user_msg)
    await session.flush()
    assistant_msg = Message(
        tenant_id=conv.tenant_id, conversation_id=conv.id, role="assistant", content=answer,
        citations=citations, standalone_query=standalone if standalone != query else None,
    )
    session.add(assistant_msg)
    if conv.title is None:
        conv.title = query[:60]
    conv.updated_at = utcnow()
    await session.commit()
    await session.refresh(assistant_msg)
    return {
        "conversation_id": conv.id, "user_message_id": user_msg.id, "answer": assistant_msg,
        "standalone_query": standalone, "all_citations_valid": all_valid, "retrieved_count": len(results),
    }
