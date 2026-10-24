"""add embedding column to chunks + pgvector cosine index

Revision ID: 8a1c2f9d4b3e
Revises: 544f35434473
Create Date: 2026-09-29

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "8a1c2f9d4b3e"
down_revision: Union[str, Sequence[str], None] = "544f35434473"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match settings.embedding_dimensions in app/core/config.py.
# 384 = BAAI/bge-small-en-v1.5, the local fastembed model used for Day 29.
# If you swap in a different embedding model later, change this AND
# config.py, then re-run the migration on a fresh database -- the column
# dimension can't be altered in place once it holds data.
EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunks", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True))
    # ivfflat over cosine distance: the standard choice for normalized text
    # embeddings. `lists = 100` is a reasonable default for a dev-scale
    # corpus -- the usual guidance is roughly sqrt(row_count) once real
    # volume is known, and the index can be dropped/rebuilt later with a
    # different value without touching the data.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_cosine ON chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_cosine")
    op.drop_column("chunks", "embedding")
