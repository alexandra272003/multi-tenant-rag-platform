from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import AccessFilter
from app.core.embeddings import embed_texts
from app.models import Chunk, Document


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Plain-Python cosine distance -- 0 = identical direction, 2 = opposite.
    Used only on the SQLite fallback path (see retrieve() below); Postgres
    computes this same quantity inside the database via pgvector instead.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # a zero vector has no direction; treat as maximally unrelated
    return 1.0 - dot / (norm_a * norm_b)


async def retrieve(
    session: AsyncSession,
    query: str,
    *,
    access: AccessFilter,
    top_k: int = 5,
    document_id: int | None = None,
    source_type: str | None = None,
    document_ids: list[int] | None = None,
) -> list[tuple[Chunk, str, float]]:
    """
    Day 30: top-k retrieval with metadata filters.

    The query is embedded with the SAME model used at ingestion time --
    this has to match, since a query embedded by a different model has no
    meaningful distance to chunks embedded by this one. Filters
    (document_id, source_type) narrow the candidate set BEFORE ranking,
    not after, so top_k results come from the filtered set rather than
    being padded out by irrelevant documents that happened to rank low.

    document_ids (plural) is a Day 32 addition for the evaluation script
    only -- it scopes retrieval to a specific set of documents (the eval
    corpus) so measured results aren't diluted by whatever else happens to
    be sitting in the same database from manual testing. Not exposed on
    the public /retrieve or /ask request schemas; internal use only.

    Week 7: `access` is REQUIRED (keyword-only, no default). Every
    retrieval is confined to the caller's tenant, to live (not soft-deleted)
    documents, and to documents the caller may read -- applied in the WHERE
    clause BEFORE ranking, so other tenants' chunks never enter the
    candidate set (and cannot be inferred from ranking or counts). A caller
    that "forgets" to scope a query cannot compile, which is the point.

    Returns (Chunk, filename, distance) tuples, sorted by distance
    ascending (closest first).

    Postgres: the distance is computed IN the database via pgvector's
    cosine_distance operator, using the ivfflat index from the Day 29
    migration -- an approximate nearest-neighbor search, not a full scan,
    which is what makes this viable at real corpus sizes.

    SQLite (test suite only): pgvector's comparator has no SQLite
    equivalent, so this path fetches the (small, test-scale) candidate set
    and ranks it in Python. This would not scale to a real corpus -- which
    is exactly why the Postgres path exists and uses an index instead of
    this fallback.
    """
    [query_vector] = await embed_texts([query])

    base_stmt = (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(access.document_clause(), Chunk.tenant_id == access.tenant_id)
    )
    if document_id is not None:
        base_stmt = base_stmt.where(Chunk.document_id == document_id)
    if document_ids is not None:
        base_stmt = base_stmt.where(Chunk.document_id.in_(document_ids))
    if source_type is not None:
        base_stmt = base_stmt.where(Document.source_type == source_type)

    if session.bind.dialect.name == "postgresql":
        # ivfflat splits the table into `lists` clusters (100, set in the
        # Day 29 migration) and by default probes only 1 of them per
        # search. On a small corpus -- a handful of documents, nowhere
        # near enough rows to fill 100 clusters meaningfully -- the one
        # probed cluster can easily miss every real match, silently
        # returning zero results with no error. Raising probes trades a
        # little search speed for actually finding the nearby vectors;
        # SET LOCAL scopes it to this transaction only, so it never leaks
        # onto a pooled connection reused by an unrelated request.
        await session.execute(text("SET LOCAL ivfflat.probes = 100"))
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = base_stmt.add_columns(distance.label("distance")).order_by(distance).limit(top_k)
        rows = await session.execute(stmt)
        return [(chunk, filename, float(dist)) for chunk, filename, dist in rows.all()]

    rows = await session.execute(base_stmt)
    scored = [
        (chunk, filename, _cosine_distance(chunk.embedding, query_vector))
        for chunk, filename in rows.all()
        if chunk.embedding is not None
    ]
    scored.sort(key=lambda row: row[2])
    return scored[:top_k]
