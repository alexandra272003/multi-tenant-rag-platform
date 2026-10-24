"""
THE answer to "which documents may this caller see?" -- defined once.

Every read path (list, get, chunks, retrieval, ask, structured, agent)
builds its SQL from AccessFilter.document_clause(), so there is exactly one
place where tenant isolation, soft delete and visibility are decided. If a
new endpoint forgets it, retrieval cannot be called at all (the filter is a
required argument), which turns "remember to scope the query" from a habit
into a type error.
"""
from dataclasses import dataclass

from sqlalchemy import and_, or_

from app.auth.rbac import Principal, Role
from app.models import Document


@dataclass(frozen=True)
class AccessFilter:
    tenant_id: int
    user_id: int | None
    role: Role

    @classmethod
    def from_principal(cls, principal: Principal) -> "AccessFilter":
        return cls(tenant_id=principal.tenant_id, user_id=principal.user_id, role=principal.role)

    @property
    def sees_private(self) -> bool:
        return self.role in (Role.OWNER, Role.ADMIN)

    def document_clause(self):
        """SQLAlchemy predicate over Document: my tenant, not deleted, and
        (tenant-visible OR mine OR I am an admin/owner)."""
        clauses = [Document.tenant_id == self.tenant_id, Document.deleted_at.is_(None)]
        if not self.sees_private:
            visible = [Document.visibility == "tenant"]
            if self.user_id is not None:
                visible.append(Document.owner_id == self.user_id)
            clauses.append(or_(*visible))
        return and_(*clauses)
