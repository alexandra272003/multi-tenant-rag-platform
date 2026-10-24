"""
Day 29: the ONLY place in the app that generates embeddings. Every other
layer works with plain lists of floats, never touching fastembed directly
-- same reasoning as any other provider client in this sprint: one place
to swap the model, one place to mock in tests.

Runs entirely locally via fastembed (ONNX Runtime) -- no API key, no
network call at request time, no provider outage to retry against. The
model weights are downloaded once (from Hugging Face) on first use and
cached on disk afterward.
"""
import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import settings
from app.core.errors import EmbeddingError

_MODEL_CACHE_DIR = "/root/.cache/fastembed"


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    # Loaded once per process and cached for the process lifetime -- the
    # first call pays for the model download + load, every call after
    # that reuses the same in-memory model.
    return TextEmbedding(model_name=settings.embedding_model, cache_dir=_MODEL_CACHE_DIR)


def _embed_sync(texts: list[str]) -> list[list[float]]:
    """The actual (blocking, CPU-bound) work -- run off the event loop, see below."""
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts, batch_size=settings.embedding_batch_size)]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns one embedding vector per input text, in the same order.

    fastembed is synchronous and CPU-bound (ONNX Runtime, no I/O to await),
    so it's run in a worker thread via asyncio.to_thread rather than called
    directly -- calling it inline would block the whole event loop (and
    every other in-flight request) for the duration of the batch.
    """
    if not texts:
        return []
    try:
        return await asyncio.to_thread(_embed_sync, texts)
    except Exception as exc:
        raise EmbeddingError(
            "The local embedding model failed to embed this batch", details={"reason": str(exc)}
        ) from exc
