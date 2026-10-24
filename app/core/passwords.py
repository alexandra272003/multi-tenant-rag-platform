"""Password hashing with scrypt from the standard library -- memory-hard,
no extra dependency. Format: scrypt$N$r$p$salt$hash (parameters travel with
the hash, so the cost can be raised later and old hashes still verify)."""
import base64
import hashlib
import hmac
import os

from app.core.config import settings

_R, _P, _DKLEN = 8, 1, 32


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def hash_password(password: str) -> str:
    n = settings.password_hash_n
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${n}${_R}${_P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)  # constant-time comparison
