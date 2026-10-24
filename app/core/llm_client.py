"""
Day 31: the ONLY place in the app that calls the generation (chat) LLM
directly. Every other layer works with plain message dicts and strings,
never touching the openai client -- same reasoning as core/embeddings.py:
one place to swap the provider, one place to mock in tests.

This is a SEPARATE provider from embeddings (core/embeddings.py, which
runs locally via fastembed). Groq's chat endpoint is real -- unlike
embeddings, which Groq does not offer despite some third-party client
libraries implying otherwise.
"""
import asyncio
from typing import Awaitable, Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings

# SDK-level retries disabled so retry behavior lives in one place
# (_with_retries) where it's easy to reason about and test.
_client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
)

# Only transient failures are worth retrying. Auth errors and bad requests
# fail identically on every attempt, so retrying them just wastes time.
RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

T = TypeVar("T")


async def _with_retries(call: Callable[[], Awaitable[T]]) -> T:
    attempt = 0
    while True:
        try:
            return await call()
        except RETRYABLE:
            if attempt >= settings.llm_max_retries:
                raise
            await asyncio.sleep(settings.llm_retry_backoff_seconds * (2**attempt))
            attempt += 1


async def get_chat_completion(messages: list[dict]) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns the assistant's reply text.
    """
    response = await _with_retries(
        lambda: _client.chat.completions.create(model=settings.llm_model, messages=messages)
    )
    return response.choices[0].message.content or ""
