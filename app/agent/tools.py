"""
Days 36/39 (Week 6): the three agent tools -- and nothing else.

    calculator       pure computation, no I/O
    document_search  retrieval, scoped to the request's allowed documents
    query_documents  read-only metadata SQL behind scoped views

Design rules every tool follows (each is also a test in
tests/test_agent_security.py):

  1. ARGUMENTS ARE UNTRUSTED. The model produces them, so each tool has a
     Pydantic schema with extra="forbid" -- a model that invents a
     `document_ids` or `tenant_id` argument gets a validation error, not a
     widened scope. Scope comes from AgentScope, never from arguments.
  2. DENIAL IS A RESULT, NOT A CRASH. A refused call returns an error
     string to the model (so it can recover or give up) and is logged as a
     security event; it does not raise out of the agent loop.
  3. OUTPUT IS UNTRUSTED TOO. Text that originated in a file is wrapped in
     <retrieved_document>/<tool_data> delimiters and scanned by the
     injection heuristic (flag only).
  4. NO WRITE, NO NETWORK, NO SHELL. The tool set has no side effects, so
     the worst a hijacked model can do is read what the scope allows.
"""
import ast
import json
import math
import operator
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.scope import AgentScope
from app.auth.access import AccessFilter
from app.agent.security import flag_injection, wrap_untrusted
from app.agent.sql_guard import SqlRejected, run_scoped_query
from app.services import retrieval_service


class ToolError(Exception):
    """Expected failure (bad input, rejected query). Message goes to the model."""


class ToolDenied(Exception):
    """Authorization failure. Logged as a security event."""


@dataclass
class SourceRegistry:
    """Stable [N] numbers for every chunk surfaced during ONE agent run,
    across all document_search calls -- so citations in the final answer
    can be checked against what was actually retrieved."""

    sources: list[dict] = field(default_factory=list)

    def register(self, chunk, filename: str, distance: float) -> int:
        for s in self.sources:
            if s["chunk_id"] == chunk.id:
                return s["number"]
        number = len(self.sources) + 1
        self.sources.append(
            {
                "number": number,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": filename,
                "page_number": chunk.page_number,
                "distance": distance,
            }
        )
        return number


@dataclass
class ToolContext:
    session: AsyncSession
    scope: AgentScope
    sources: SourceRegistry
    # The caller's AccessFilter. Retrieval REQUIRES one; None only in unit
    # tests of tools that never touch documents (e.g. the calculator).
    access: AccessFilter | None = None


@dataclass
class ToolOutput:
    text: str
    injection_flags: list[str] = field(default_factory=list)


# ---- calculator ------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_MAX_POW_EXPONENT = 100
_MAX_POW_BASE = 1_000_000


def _calc(node):
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):  # excludes bool/str
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _calc(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > _MAX_POW_EXPONENT or abs(left) > _MAX_POW_BASE
        ):
            raise ToolError("exponent or base too large")  # blocks 9**9**9 style DoS
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("result too large") from None
    raise ToolError("unsupported syntax: only numbers, + - * / // % ** and parentheses")


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(
        min_length=1, max_length=200,
        description="Arithmetic expression, e.g. '(12.5 * 4) / 3'. Numbers and + - * / // % ** only.",
    )


async def _run_calculator(args: CalculatorArgs, ctx: ToolContext) -> ToolOutput:
    try:
        tree = ast.parse(args.expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        raise ToolError("could not parse expression") from None
    result = _calc(tree)
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ToolError("result is not a finite number")
        return ToolOutput(format(result, ".10g"))
    return ToolOutput(str(result))


# ---- document_search ---------------------------------------------------------


class DocumentSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, description="What to look for, as a natural-language query.")
    top_k: int = Field(default=3, ge=1, le=5, description="How many passages to return (1-5).")
    document_id: int | None = Field(default=None, description="Optionally restrict to one document id.")


async def _run_document_search(args: DocumentSearchArgs, ctx: ToolContext) -> ToolOutput:
    scope = ctx.scope
    if args.document_id is not None and not scope.permits_document(args.document_id):
        # Same wording whether the id is out of scope or doesn't exist: the
        # model (or an attacker steering it) must not be able to probe
        # which document ids exist.
        raise ToolDenied(f"document_id {args.document_id} is not available to this request")

    if ctx.access is None:
        raise ToolDenied("document search is unavailable without an authenticated caller")
    ids = None if scope.document_ids is None else sorted(scope.document_ids)
    results = await retrieval_service.retrieve(
        ctx.session, args.query, access=ctx.access, top_k=args.top_k,
        document_id=args.document_id, document_ids=ids,
    )

    blocks, flags = [], []
    for chunk, filename, distance in results:
        # Defence in depth: authorization is enforced in the query above AND
        # re-checked on the way out, so a future retrieval bug can't leak.
        if not scope.permits_document(chunk.document_id):
            continue
        number = ctx.sources.register(chunk, filename, distance)
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        blocks.append(wrap_untrusted(f"[{number}] {filename}{page}", chunk.content))
        flags += flag_injection(chunk.content)
    if not blocks:
        return ToolOutput("No matching passages found.")
    return ToolOutput("\n\n".join(blocks), sorted(set(flags)))


# ---- query_documents -----------------------------------------------------------


class QueryDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(
        min_length=1, max_length=500,
        description=(
            "One read-only SELECT over documents(id, filename, source_type, page_count, "
            "chunk_size, chunk_overlap, chunk_count, created_at) and chunks(id, document_id, "
            "chunk_index, page_number, token_count, char_start, char_end). Metadata only: "
            "it cannot read document text."
        ),
    )


async def _run_query_documents(args: QueryDocumentsArgs, ctx: ToolContext) -> ToolOutput:
    try:
        result = await run_scoped_query(ctx.session, args.sql, ctx.scope.document_ids)
    except SqlRejected as exc:
        raise ToolError(str(exc)) from None
    payload = json.dumps(
        {"columns": result.columns, "rows": result.rows, "truncated": result.truncated}
    )
    # Filenames are attacker-controlled (whoever uploads picks them), so SQL
    # output is untrusted text too.
    return ToolOutput(wrap_untrusted("query_documents result", payload, tag="tool_data"),
                      flag_injection(payload))


# ---- registry ------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], Awaitable[ToolOutput]]

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("calculator", "Evaluate an arithmetic expression exactly. Use for any calculation.",
             CalculatorArgs, _run_calculator),
        Tool("document_search",
             "Search the user's documents for passages relevant to a query. Returns numbered "
             "passages; cite them as [N].",
             DocumentSearchArgs, _run_document_search),
        Tool("query_documents",
             "Run a read-only SQL SELECT over document METADATA (counts, filenames, page counts). "
             "Cannot read document text -- use document_search for that.",
             QueryDocumentsArgs, _run_query_documents),
    )
}
