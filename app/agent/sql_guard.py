"""
Day 36/39: the "safe SQL" tool. Letting an LLM write SQL against your
database is the riskiest of the three tools, so it gets layered defences
instead of one clever filter:

  Layer 1 -- validate_sql(): a strict ALLOWLIST. One SELECT statement,
            limited character set, every identifier must be a known
            keyword / function / table / column (or an alias declared
            with AS). Anything unknown is rejected, rather than trying to
            enumerate everything dangerous (a denylist always misses
            something: pg_sleep, pg_read_file, UNION tricks, ...).
  Layer 2 -- scoped views: the query never sees the real tables. Every
            `documents` / `chunks` in the (validated) query is rewritten
            to `scoped_documents` / `scoped_chunks`, CTEs that (a) expose
            only safe columns -- no chunk text, no embeddings -- and (b)
            are filtered to the request's allowed document ids. After the
            rewrite no token in the query can refer to a real table.
            (First version shadowed the table names with same-named CTEs;
            that works on Postgres but SQLite treats the inner reference
            as circular -- a dialect difference the test suite caught.
            Rewriting is portable, and stricter.)
  Layer 3 -- execution limits: wrapped in an outer LIMIT, statement
            timeout, read-only transaction (Postgres).
  Layer 4 (production, NOT implemented here) -- connect with a dedicated
            database role that has SELECT on only these views. The first
            three layers are application code; a DB role is a layer the
            application cannot get wrong.

Why chunk text is excluded: reading content must go through
document_search, which numbers sources for citation and wraps text as
untrusted. A SQL path to raw content would bypass both.
"""
import re
from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

DOCUMENT_COLUMNS = (
    "id", "filename", "source_type", "page_count",
    "chunk_size", "chunk_overlap", "chunk_count", "created_at",
)
CHUNK_COLUMNS = (
    "id", "document_id", "chunk_index", "page_number",
    "token_count", "char_start", "char_end",
)
TABLES = {"documents", "chunks"}

_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "like", "ilike", "is",
    "null", "order", "by", "group", "having", "limit", "offset", "as",
    "distinct", "asc", "desc", "between", "true", "false", "join", "inner",
    "left", "on",
}
_FUNCTIONS = {"count", "sum", "avg", "min", "max", "lower", "upper", "length"}
# Bare words that behave like values/functions without parentheses.
_SPECIAL_VALUES = {
    "current_user", "session_user", "user", "current_role", "current_catalog",
    "current_schema", "current_database", "current_date", "current_time",
    "current_timestamp", "localtime", "localtimestamp", "system_user",
}
_ALLOWED_WORDS = _KEYWORDS | _FUNCTIONS | TABLES | set(DOCUMENT_COLUMNS) | set(CHUNK_COLUMNS)

MAX_SQL_CHARS = 500
_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_CHARSET_RE = re.compile(r"^[a-z0-9_\s,.()*=<>!+\-/%|']*$")
_IDENT_RE = re.compile(r"[a-z_][a-z0-9_]*")
_ALIAS_RE = re.compile(r"\bas\s+([a-z_][a-z0-9_]*)")
# "FROM documents d" / "JOIN chunks c": an implicit table alias (no AS).
_TABLE_ALIAS_RE = re.compile(r"\b(?:from|join)\s+(?:documents|chunks)\s+([a-z_][a-z0-9_]*)")
_CALL_RE = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")


class SqlRejected(Exception):
    """The query failed validation. The message is safe to show the model."""


def validate_sql(sql: str) -> str:
    """Return the (trailing-semicolon-trimmed) query if it passes, else
    raise SqlRejected. Never returns a modified query beyond that trim."""
    query = sql.strip()
    if query.endswith(";"):
        query = query[:-1].rstrip()
    if not query:
        raise SqlRejected("empty query")
    if len(query) > MAX_SQL_CHARS:
        raise SqlRejected(f"query too long (max {MAX_SQL_CHARS} characters)")

    # Blank out string literals first so their CONTENT can't trip (or
    # smuggle past) the checks below.
    stripped = _LITERAL_RE.sub("''", query)
    low = stripped.lower()
    if "'" in low.replace("''", ""):
        raise SqlRejected("unterminated string literal")
    if ";" in low:
        raise SqlRejected("only a single statement is allowed")
    if "--" in low or "/*" in low or "*/" in low:
        raise SqlRejected("comments are not allowed")
    if not _CHARSET_RE.match(low):
        raise SqlRejected("query contains characters that are not allowed")
    if not re.match(r"\s*select\b", low):
        raise SqlRejected("only SELECT queries are allowed")

    aliases = set(_ALIAS_RE.findall(low))
    aliases |= {a for a in _TABLE_ALIAS_RE.findall(low) if a not in _KEYWORDS}
    bad_alias = aliases & _SPECIAL_VALUES
    if bad_alias:
        raise SqlRejected(f"alias {sorted(bad_alias)[0]!r} is not allowed")
    allowed = _ALLOWED_WORDS | aliases

    for ident in _IDENT_RE.findall(low):
        if ident not in allowed:
            raise SqlRejected(
                f"{ident!r} is not allowed. Tables: documents({', '.join(DOCUMENT_COLUMNS)}), "
                f"chunks({', '.join(CHUNK_COLUMNS)}). Functions: {', '.join(sorted(_FUNCTIONS))}."
            )
    for name in _CALL_RE.findall(low):
        if name not in _FUNCTIONS and name not in _KEYWORDS:
            raise SqlRejected(f"function {name!r} is not allowed")
    return query


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool


_TABLE_WORD_RE = re.compile(r"\b(documents|chunks)\b", re.IGNORECASE)
_SPLIT_LITERALS_RE = re.compile(r"('(?:[^']|'')*')")


def _rewrite_tables(query: str) -> str:
    """documents -> scoped_documents, chunks -> scoped_chunks, but only
    OUTSIDE string literals (a filter like ILIKE '%documents%' must keep
    its text)."""
    parts = _SPLIT_LITERALS_RE.split(query)
    for i in range(0, len(parts), 2):  # even indexes are non-literal
        parts[i] = _TABLE_WORD_RE.sub(lambda m: "scoped_" + m.group(1).lower(), parts[i])
    return "".join(parts)


def _scoped_ctes(ids: list[int] | None) -> str:
    doc_cols = ", ".join(DOCUMENT_COLUMNS)
    chunk_cols = ", ".join(CHUNK_COLUMNS)
    doc_where = "" if ids is None else " WHERE id IN :ids"
    chunk_where = "" if ids is None else " WHERE document_id IN :ids"
    return (
        f"WITH scoped_documents AS (SELECT {doc_cols} FROM documents{doc_where}), "
        f"scoped_chunks AS (SELECT {chunk_cols} FROM chunks{chunk_where}) "
    )


def _jsonable(v):
    return v if v is None or isinstance(v, (int, float, str, bool)) else str(v)


async def run_scoped_query(
    session: AsyncSession,
    sql: str,
    allowed_document_ids: frozenset[int] | None,
    row_limit: int | None = None,
    timeout_ms: int | None = None,
) -> QueryResult:
    """Validate, scope, limit and run. Raises SqlRejected (validation) or
    SqlRejected (execution failure, message trimmed) -- both safe to show
    the model."""
    row_limit = row_limit or settings.agent_sql_row_limit
    timeout_ms = timeout_ms or settings.agent_sql_timeout_ms
    query = validate_sql(sql)

    if allowed_document_ids is not None and not allowed_document_ids:
        return QueryResult(columns=[], rows=[], truncated=False)  # empty scope: nothing visible

    ids = None if allowed_document_ids is None else sorted(allowed_document_ids)
    # +1 row so we can tell "exactly row_limit rows" from "cut off".
    stmt = text(f"{_scoped_ctes(ids)} SELECT * FROM ({_rewrite_tables(query)}) AS q LIMIT :lim")
    if ids is not None:
        stmt = stmt.bindparams(bindparam("ids", expanding=True))
    params: dict = {"lim": row_limit + 1}
    if ids is not None:
        params["ids"] = ids

    try:
        if session.bind.dialect.name == "postgresql":
            # LOCAL: scoped to this transaction, so settings never leak onto
            # a pooled connection reused by another request.
            await session.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            await session.execute(text("SET LOCAL transaction_read_only = on"))
        result = await session.execute(stmt, params)
        rows = result.fetchall()
        columns = list(result.keys())
        if session.bind.dialect.name == "postgresql":
            # The tight timeout was for THIS query; don't let it linger and
            # cancel a later (legitimately slower) retrieval step in the
            # same request. The read-only flag stays on: the agent only reads.
            await session.execute(text("SET LOCAL statement_timeout = DEFAULT"))
    except SQLAlchemyError as exc:
        await session.rollback()  # a failed statement poisons a Postgres transaction
        first_line = (getattr(exc, "orig", None) or exc)
        raise SqlRejected(f"query failed: {str(first_line).splitlines()[0][:200]}") from exc

    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    return QueryResult(
        columns=columns,
        rows=[[_jsonable(v) for v in row] for row in rows],
        truncated=truncated,
    )
