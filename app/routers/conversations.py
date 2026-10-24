from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import require
from app.auth.limits import guard_query
from app.auth.rbac import Permission, Principal
from app.core.db import get_session
from app.schemas import (
    ChatRequest, ChatResponse, ConversationCreate, ConversationDetail, ConversationRead, MessageRead,
)
from app.services import chat_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    principal: Principal = Depends(require(Permission.QUERY)),
    session: AsyncSession = Depends(get_session),
):
    return await chat_service.create_conversation(session, principal, payload.title)


@router.get("", response_model=list[ConversationRead])
async def list_conversations(
    principal: Principal = Depends(require(Permission.QUERY)),
    session: AsyncSession = Depends(get_session),
):
    """Your conversations only -- never other users', whatever your role."""
    return await chat_service.list_conversations(session, principal)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    principal: Principal = Depends(require(Permission.QUERY)),
    session: AsyncSession = Depends(get_session),
):
    conv = await chat_service.get_conversation_or_404(session, principal, conversation_id)
    messages = await chat_service.list_messages(session, principal, conv.id)
    return ConversationDetail(
        **ConversationRead.model_validate(conv).model_dump(),
        messages=[MessageRead.model_validate(m) for m in messages],
    )


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: int,
    payload: ChatRequest,
    principal: Principal = Depends(guard_query),
    session: AsyncSession = Depends(get_session),
):
    """One conversational turn: condense follow-up -> retrieve (under YOUR current access) -> answer."""
    result = await chat_service.chat(
        session, principal, conversation_id, payload.query, top_k=payload.top_k, document_id=payload.document_id
    )
    return ChatResponse(
        conversation_id=result["conversation_id"], user_message_id=result["user_message_id"],
        answer=MessageRead.model_validate(result["answer"]), standalone_query=result["standalone_query"],
        all_citations_valid=result["all_citations_valid"], retrieved_count=result["retrieved_count"],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    principal: Principal = Depends(require(Permission.QUERY)),
    session: AsyncSession = Depends(get_session),
):
    await chat_service.delete_conversation(session, principal, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
