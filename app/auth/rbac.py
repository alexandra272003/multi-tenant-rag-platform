"""
Role-based access control: WHO may do WHAT. One table, easy to audit.

                         viewer  editor  admin  owner
    document:read          Y       Y       Y      Y     (visible docs only, see access.py)
    query (ask/search)     Y       Y       Y      Y
    document:create        -       Y       Y      Y
    document:write_own     -       Y       Y      Y     (edit/delete documents you created)
    document:write_any     -       -       Y      Y
    user:read              -       -       Y      Y
    user:manage            -       -       Y*     Y     (*admin: editors/viewers only)
    audit:read             -       -       Y      Y

Two kinds of check, deliberately separate:
  1. ROLE permission  ("may an editor create documents?")  -> require(...)
  2. OBJECT check     ("may THIS editor edit THIS document?") -> can_write_document
A role check alone is the classic authz bug: it lets every editor edit
every document. JWT proves identity; this module decides what that
identity may touch.
"""
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class Permission(str, Enum):
    DOCUMENT_READ = "document:read"
    QUERY = "query"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_WRITE_OWN = "document:write_own"
    DOCUMENT_WRITE_ANY = "document:write_any"
    USER_READ = "user:read"
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"


_VIEWER = {Permission.DOCUMENT_READ, Permission.QUERY}
_EDITOR = _VIEWER | {Permission.DOCUMENT_CREATE, Permission.DOCUMENT_WRITE_OWN}
_ADMIN = _EDITOR | {
    Permission.DOCUMENT_WRITE_ANY, Permission.USER_READ, Permission.USER_MANAGE, Permission.AUDIT_READ,
}
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset(_VIEWER),
    Role.EDITOR: frozenset(_EDITOR),
    Role.ADMIN: frozenset(_ADMIN),
    Role.OWNER: frozenset(_ADMIN),  # owners differ from admins in WHO they may manage, not in what they may do
}


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: built from the database row (never from
    token claims), so role changes and deactivation apply immediately."""

    user_id: int | None  # None only for system/offline principals
    tenant_id: int
    role: Role
    email: str = ""

    def can(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]

    @classmethod
    def system(cls, tenant_id: int) -> "Principal":
        """Offline scripts (eval, bootstrap): full rights inside ONE tenant."""
        return cls(user_id=None, tenant_id=tenant_id, role=Role.OWNER, email="system")


def can_write_document(principal: Principal, document) -> bool:
    if principal.can(Permission.DOCUMENT_WRITE_ANY):
        return True
    return (
        principal.can(Permission.DOCUMENT_WRITE_OWN)
        and principal.user_id is not None
        and document.owner_id == principal.user_id
    )


def can_assign_role(actor: Role, target_current: Role | None, new_role: Role) -> bool:
    """May `actor` give `new_role` to a user who currently has `target_current`
    (None = creating a brand-new user)?  Owners: anything. Admins: may only
    move people between editor and viewer, and may not touch owners/admins
    (an admin who could promote themselves or demote an owner would make the
    hierarchy meaningless). Everyone else: nothing."""
    if actor is Role.OWNER:
        return True
    if actor is Role.ADMIN:
        return new_role in (Role.EDITOR, Role.VIEWER) and target_current in (
            None, Role.EDITOR, Role.VIEWER,
        )
    return False
