from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None = None
    filename: str
    source_type: str
    page_count: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    visibility: str
    version: int
    owner_id: int | None = None
    # Ingestion lifecycle: uploaded | processing | ready | failed. Until `ready`
    # page_count / chunk_count are 0 and the document has no searchable chunks.
    status: str = "ready"
    error_message: str | None = None
    attempts: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentUpdate(BaseModel):
    """PATCH body. `expected_version` is the optimistic-lock token: the
    `version` the client last read."""

    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    visibility: Literal["tenant", "private"] | None = None

    @model_validator(mode="after")
    def at_least_one_change(self):
        if self.title is None and self.visibility is None:
            raise ValueError("provide at least one of: title, visibility")
        return self


# ---- Week 7: auth, users, audit ------------------------------------------------

RoleName = Literal["owner", "admin", "editor", "viewer"]


class RegisterRequest(BaseModel):
    tenant_name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: int
    tenant_slug: str | None = None
    role: RoleName


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    tenant_id: int
    email: str
    role: RoleName
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    role: RoleName = "viewer"


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: RoleName | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def at_least_one_change(self):
        if self.role is None and self.is_active is None:
            raise ValueError("provide at least one of: role, is_active")
        return self


class AuditRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    actor_user_id: int | None
    action: str
    resource_type: str
    resource_id: int | None
    details: dict | None
    created_at: datetime


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chunk_index: int
    page_number: int | None
    content: str
    char_start: int
    char_end: int
    token_count: int


class ChunkPreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    chunk_size: int = Field(default=500, ge=5, le=4000)
    chunk_overlap: int = Field(default=50, ge=0)
    clean: bool = True


class PreviewChunk(BaseModel):
    chunk_index: int
    page_number: int | None = None
    content: str
    char_start: int
    char_end: int
    token_count: int


class ChunkPreviewResponse(BaseModel):
    cleaned_char_count: int
    total_tokens: int
    chunk_count: int
    chunks: list[PreviewChunk]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    # Metadata filters (Day 30): narrow the search before ranking, rather
    # than ranking the whole corpus and discarding results afterward.
    document_id: int | None = None
    source_type: str | None = None  # "pdf" | "txt" | "md"


class RetrievalResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    page_number: int | None
    content: str
    # Cosine distance: 0 = identical direction, 2 = opposite. Returned raw
    # (not converted to a 0-1 "similarity score") so it's clear this is a
    # distance -- smaller is more relevant -- rather than implying a
    # normalized confidence the metric doesn't actually provide.
    distance: float


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: int | None = None
    source_type: str | None = None


class Citation(BaseModel):
    number: int  # the [N] the model cited, matching position in the retrieved list
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    distance: float


class Usage(BaseModel):
    """Provider-reported spend for one request (Week 6, Sat). 0 tokens with
    model_calls > 0 means the provider did not report usage, not that it was free."""
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    # False if the model cited a source number that was never actually
    # retrieved -- a fabricated citation, checked rather than trusted.
    all_citations_valid: bool
    retrieved_count: int
    # None on the hand-built /ask (llm_client returns text only); filled on
    # /ask/langchain so the modes can be compared on cost.
    usage: Usage | None = None


class GroundedAnswer(BaseModel):
    """
    Day 37: the contract the LLM must satisfy -- NOT an HTTP response
    schema. The model is told this shape (via json_schema / tool schema)
    and its output is validated against it before anything downstream
    trusts it. Validation here is structural + internal consistency;
    checks that need request context (are the cited numbers within the
    retrieved set?) live in structured.py, because a schema alone cannot
    know how many chunks were retrieved.
    """

    answerable: bool = Field(
        description="True only if the numbered context contains the answer."
    )
    answer: str = Field(
        min_length=1,
        description="The answer, using only the context. May use inline [N] markers.",
    )
    cited_sources: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=list,
        description="Source numbers [N] that support the answer. Empty if not answerable.",
    )

    @field_validator("cited_sources")
    @classmethod
    def unique_sorted(cls, v: list[int]) -> list[int]:
        # Normalize rather than reject: [2, 1, 2] is harmless noise, not a
        # contract violation worth burning a retry (and tokens) on.
        return sorted(set(v))

    @model_validator(mode="after")
    def citations_match_answerability(self):
        if self.answerable and not self.cited_sources:
            raise ValueError("answerable=true requires at least one entry in cited_sources")
        if not self.answerable and self.cited_sources:
            raise ValueError("answerable=false must have an empty cited_sources")
        return self


class StructuredAskResponse(BaseModel):
    query: str
    answerable: bool
    answer: str
    citations: list[Citation]
    retrieved_count: int
    # Observability for the retry layer: how many LLM calls it took, and
    # why each failed attempt was rejected. attempts == 0 means the LLM
    # was never called (nothing retrieved).
    attempts: int
    validation_errors: list[str]
    # Summed over ALL attempts, including the failed ones.
    usage: Usage | None = None


# ---- Week 6 agent (Days 36 / 38 / 39) -------------------------------------

ToolName = Literal["calculator", "document_search", "query_documents"]


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # Restricts which documents the agent's tools can see. None = all.
    # Single-tenant demo only: in a multi-tenant system this must come from
    # the authenticated context, not the request body (see agent/scope.py).
    document_ids: list[int] | None = Field(default=None, max_length=200)
    # Restricts which tools the model may call. None = all three.
    allowed_tools: list[ToolName] | None = None
    max_steps: int = Field(default=4, ge=1, le=6)


class AgentResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    all_citations_valid: bool
    steps: int
    tool_calls_used: int
    budget_exhausted: bool
    # Token accounting summed over every model call in the run (zeros if
    # the provider doesn't report usage while streaming).
    usage: dict
    # Ordered record of what the agent did: tool_call / tool_result /
    # tool_denied / budget_exhausted events (token events are omitted).
    trace: list[dict]


# ---- Week 7 Fri: conversational RAG -----------------------------------------------------------


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=120)


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    citations: list[dict] | None = None
    standalone_query: str | None = None
    created_at: datetime


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: int | None = None


class ChatResponse(BaseModel):
    conversation_id: int
    user_message_id: int
    answer: MessageRead
    standalone_query: str
    all_citations_valid: bool
    retrieved_count: int
