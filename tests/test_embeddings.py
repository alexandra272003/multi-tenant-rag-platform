from sqlalchemy import select

from app.core import embeddings
from app.core.config import settings
from app.core.errors import EmbeddingError
from app.models import Chunk
from tests.conftest import LONG_TEXT


async def test_embed_texts_returns_one_vector_per_input(monkeypatch):
    monkeypatch.setattr(
        embeddings, "_embed_sync", lambda texts: [[0.1] * settings.embedding_dimensions for _ in texts]
    )
    vectors = await embeddings.embed_texts(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == settings.embedding_dimensions for v in vectors)


async def test_embed_texts_empty_input_never_touches_the_model(monkeypatch):
    """
    Proves the empty-list short circuit actually skips work, rather than
    calling the (possibly slow, first-call-downloads-a-model) embedder
    with zero inputs.
    """
    called = False

    def fake_embed_sync(texts):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(embeddings, "_embed_sync", fake_embed_sync)
    assert await embeddings.embed_texts([]) == []
    assert called is False


async def test_embed_texts_wraps_model_failure_as_embedding_error(monkeypatch):
    def failing(texts):
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(embeddings, "_embed_sync", failing)
    try:
        await embeddings.embed_texts(["hello"])
        assert False, "expected EmbeddingError"
    except EmbeddingError:
        pass


async def test_uploaded_chunks_are_stored_with_embeddings(client, session):
    resp = await client.post(
        "/documents",
        files={"file": ("notes.txt", LONG_TEXT.encode())},
        params={"chunk_size": 40, "chunk_overlap": 10},
    )
    assert resp.status_code == 202
    doc_id = resp.json()["id"]

    result = await session.execute(select(Chunk).where(Chunk.document_id == doc_id))
    chunks = list(result.scalars().all())
    assert len(chunks) > 1
    assert all(c.embedding is not None for c in chunks)
    assert all(len(c.embedding) == settings.embedding_dimensions for c in chunks)
