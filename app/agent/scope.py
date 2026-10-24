"""
What a single agent request is ALLOWED to do. Built server-side, once per
request, and enforced by the tool executor -- never by the model.

The model can ask for anything; `AgentScope` decides what happens. That
separation is the whole security design: a prompt-injected model is an
attacker with the model's capabilities, so authorization cannot live in
the prompt ("please only read document 3") -- it has to live in code the
model cannot talk its way past.

NOTE on where scope comes from: in this single-tenant project it is taken
from the request body (`document_ids`), which is fine for a demo and NOT a
security boundary -- a caller can simply omit it. In the multi-tenant
capstone it must be derived from the authenticated tenant/user context
(e.g. "documents this user's role can read"), never from client input.
"""
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass(frozen=True)
class AgentScope:
    # None = no restriction (all documents). An EMPTY set is NOT "no
    # restriction": it means "nothing is visible" -- None and empty are
    # deliberately different, since confusing them is a classic authz bug.
    document_ids: frozenset[int] | None = None
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    max_steps: int = settings.agent_max_steps
    max_tool_calls: int = settings.agent_max_tool_calls
    max_calls_per_step: int = settings.agent_max_calls_per_step
    max_tool_output_chars: int = settings.agent_max_tool_output_chars

    def permits_document(self, document_id: int) -> bool:
        return self.document_ids is None or document_id in self.document_ids

    def permits_tool(self, name: str) -> bool:
        return name in self.allowed_tools
