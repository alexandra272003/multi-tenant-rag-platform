import re
import secrets

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import Principal, Role, can_assign_role
from app.core.errors import (
    ConflictError, ForbiddenError, InvalidInputError, NotFoundError, UnauthorizedError,
)
from app.core.passwords import hash_password, verify_password
from app.core.tenancy import set_tenant_context
from app.core.tokens import create_access_token
from app.models import Tenant, User
from app.services.audit_service import add_audit

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_dummy_hash: str | None = None


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email) or len(email) > 254:
        raise InvalidInputError("Invalid email address", details={"field": "email"})
    return email


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]
    return slug or "tenant"


async def register_tenant(session: AsyncSession, tenant_name: str, email: str, password: str):
    """Self-service signup: creates a tenant and its first user, the OWNER.

    Two things can fail under concurrency, and the pre-checks below cannot
    prevent either (check-then-insert is a race): the email unique key and
    the slug unique key. So the database constraints are the source of truth
    and this function CLASSIFIES the IntegrityError: email taken -> 409;
    slug taken -> retry with a random suffix. (Found by running the signup
    race on real Postgres: the violation fires at flush(), so it must be
    handled around every statement that can raise it, not just the commit.)
    """
    email = normalize_email(email)
    password_hash = hash_password(password)
    base_slug = _slugify(tenant_name)
    taken = ConflictError("An account with this email already exists", details={"field": "email"})

    if (await session.execute(select(User.id).where(User.email == email))).first():
        raise taken

    for attempt in range(5):
        slug = base_slug if attempt == 0 else f"{base_slug}-{secrets.token_hex(2)}"
        try:
            tenant = Tenant(name=tenant_name.strip(), slug=slug)
            session.add(tenant)
            await session.flush()
            # Bind the session to the NEW tenant before touching tenant-scoped
            # tables (audit_log): under Postgres RLS a row for tenant N is only
            # writable once the transaction's tenant context says N.
            await set_tenant_context(session, tenant.id)
            user = User(tenant_id=tenant.id, email=email, password_hash=password_hash, role=Role.OWNER.value)
            session.add(user)
            await session.flush()
            add_audit(session, tenant_id=tenant.id, actor_user_id=user.id, action="tenant.created",
                      resource_type="tenant", resource_id=tenant.id)
            add_audit(session, tenant_id=tenant.id, actor_user_id=user.id, action="user.created",
                      resource_type="user", resource_id=user.id, details={"role": user.role})
            await session.commit()
            return tenant, user, create_access_token(user.id, tenant.id, user.role)
        except IntegrityError:
            await session.rollback()
            session.info.pop("tenant_id", None)
            if (await session.execute(select(User.id).where(User.email == email))).first():
                raise taken from None  # someone else won the email
            # otherwise it was the slug: loop and try a suffixed one
    raise ConflictError("Could not allocate a unique tenant slug; please retry")


async def authenticate(session: AsyncSession, email: str, password: str) -> tuple[User, str]:
    global _dummy_hash
    email = email.strip().lower()
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        # Spend the same hashing time as a real check so response time doesn't
        # reveal whether the email exists (user enumeration via timing).
        _dummy_hash = _dummy_hash or hash_password("not-a-real-password")
        verify_password(password, _dummy_hash)
        raise UnauthorizedError("Invalid email or password")
    ok = verify_password(password, user.password_hash)
    if not ok or not user.is_active:
        raise UnauthorizedError("Invalid email or password")  # same message for wrong pw / inactive / unknown

    await set_tenant_context(session, user.tenant_id)
    add_audit(session, tenant_id=user.tenant_id, actor_user_id=user.id, action="auth.login",
              resource_type="user", resource_id=user.id)
    await session.commit()
    return user, create_access_token(user.id, user.tenant_id, user.role)


async def create_user(session: AsyncSession, principal: Principal, email: str, password: str, role: Role) -> User:
    if not can_assign_role(principal.role, None, role):
        raise ForbiddenError(f"A {principal.role.value} cannot create a {role.value}")
    email = normalize_email(email)
    if (await session.execute(select(User.id).where(User.email == email))).first():
        raise ConflictError("An account with this email already exists", details={"field": "email"})
    user = User(tenant_id=principal.tenant_id, email=email, password_hash=hash_password(password), role=role.value)
    session.add(user)
    try:
        await session.flush()
        add_audit(session, tenant_id=principal.tenant_id, actor_user_id=principal.user_id,
                  action="user.created", resource_type="user", resource_id=user.id,
                  details={"role": role.value})
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise ConflictError("An account with this email already exists", details={"field": "email"}) from None
    await session.refresh(user)
    return user


async def list_users(session: AsyncSession, principal: Principal) -> list[User]:
    result = await session.execute(
        select(User).where(User.tenant_id == principal.tenant_id).order_by(User.id)
    )
    return list(result.scalars().all())


async def update_user(
    session: AsyncSession, principal: Principal, user_id: int,
    role: Role | None, is_active: bool | None,
) -> User:
    # tenant_id in the WHERE: a user id from another tenant is simply "not found"
    # (404, not 403 -- don't confirm that it exists).
    target = (await session.execute(
        select(User).where(User.id == user_id, User.tenant_id == principal.tenant_id)
    )).scalar_one_or_none()
    if target is None:
        raise NotFoundError(f"User {user_id} not found")
    if target.id == principal.user_id:
        # No self-escalation, and no locking yourself out by deactivating yourself.
        raise ForbiddenError("You cannot change your own role or status")

    target_role = Role(target.role)
    changes: dict = {}
    if role is not None and role != target_role:
        if not can_assign_role(principal.role, target_role, role):
            raise ForbiddenError(f"A {principal.role.value} cannot change a {target_role.value} to {role.value}")
        changes["role"] = role.value
    if is_active is not None and is_active != target.is_active:
        # Deactivating/reactivating needs the same authority as managing that person's role.
        if not can_assign_role(principal.role, target_role, target_role):
            raise ForbiddenError(f"A {principal.role.value} cannot change a {target_role.value}'s status")
        changes["is_active"] = is_active

    losing_owner = target_role is Role.OWNER and (
        changes.get("role") not in (None, Role.OWNER.value) or changes.get("is_active") is False
    )
    if losing_owner:
        other_owners = (await session.execute(
            select(func.count()).select_from(User).where(
                User.tenant_id == principal.tenant_id, User.role == Role.OWNER.value,
                User.is_active.is_(True), User.id != target.id,
            )
        )).scalar_one()
        if other_owners == 0:
            raise ConflictError("A tenant must keep at least one active owner")

    if changes:
        for key, value in changes.items():
            setattr(target, key, value)
        add_audit(session, tenant_id=principal.tenant_id, actor_user_id=principal.user_id,
                  action="user.updated", resource_type="user", resource_id=target.id, details=changes)
        await session.commit()
        await session.refresh(target)
    return target
