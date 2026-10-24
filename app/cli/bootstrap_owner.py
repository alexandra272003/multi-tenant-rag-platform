"""
Create (or attach) an OWNER for the default tenant, so documents that
existed before multi-tenancy (adopted by tenant 1 in the migration) become
reachable. Idempotent about the tenant; refuses to overwrite an existing user.

    docker compose exec api python -m app.cli.bootstrap_owner \
        --email you@example.com --password 'a-strong-password'
"""
import argparse
import asyncio

from sqlalchemy import select

from app.auth.rbac import Role
from app.core.db import SessionLocal
from app.core.passwords import hash_password
from app.core.tenancy import set_tenant_context
from app.models import Tenant, User
from app.services.audit_service import add_audit


async def bootstrap_owner(email: str, password: str, tenant_slug: str = "default") -> int:
    email = email.strip().lower()
    async with SessionLocal() as session:
        tenant = (await session.execute(select(Tenant).where(Tenant.slug == tenant_slug))).scalar_one_or_none()
        if tenant is None:
            tenant = Tenant(name=tenant_slug.title(), slug=tenant_slug)
            session.add(tenant)
            await session.flush()
        if (await session.execute(select(User.id).where(User.email == email))).first():
            raise SystemExit(f"A user with email {email} already exists")
        await set_tenant_context(session, tenant.id)
        user = User(tenant_id=tenant.id, email=email, password_hash=hash_password(password), role=Role.OWNER.value)
        session.add(user)
        await session.flush()
        add_audit(session, tenant_id=tenant.id, actor_user_id=user.id, action="user.created",
                  resource_type="user", resource_id=user.id, details={"role": "owner", "via": "bootstrap_owner"})
        await session.commit()
        return user.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--tenant-slug", default="default")
    args = parser.parse_args()
    if len(args.password) < 8:
        raise SystemExit("password must be at least 8 characters")
    print(f"created owner id={asyncio.run(bootstrap_owner(args.email, args.password, args.tenant_slug))}")
