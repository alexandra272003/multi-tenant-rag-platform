from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code = 400
    code = "app_error"

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class UnauthorizedError(AppError):
    """Not authenticated: missing, malformed, expired or revoked credentials."""
    status_code = 401
    code = "unauthorized"
    headers = {"WWW-Authenticate": "Bearer"}


class ForbiddenError(AppError):
    """Authenticated, but this role/principal may not do that."""
    status_code = 403
    code = "forbidden"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class VersionConflictError(AppError):
    """Optimistic-lock failure: the row changed since the client read it."""
    status_code = 409
    code = "version_conflict"


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class InvalidDocumentError(AppError):
    """The file was readable as a request, but yielded no usable text."""
    status_code = 422
    code = "invalid_document"


class InvalidChunkParamsError(AppError):
    status_code = 422
    code = "invalid_chunk_params"


class EmbeddingError(AppError):
    """The embedding provider failed, timed out, or returned something unusable."""
    status_code = 502
    code = "embedding_error"


class ProviderError(AppError):
    """The LLM (generation) provider failed, timed out, or returned something unusable."""
    status_code = 502
    code = "provider_error"


class InvalidModelOutputError(AppError):
    """
    The LLM answered (transport was fine) but never produced output that
    satisfied the response contract, even after validation retries.
    Distinct from ProviderError so callers/dashboards can tell "provider
    is down" from "model won't follow the schema" -- different fixes.
    """
    status_code = 502
    code = "invalid_model_output"


class RateLimitedError(AppError):
    """Too many requests in a short window. Retry-After says when to come back."""
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, retry_after: int, details: dict | None = None):
        super().__init__(message, {**(details or {}), "retry_after_seconds": retry_after})
        self.headers = {"Retry-After": str(retry_after)}


class QuotaExceededError(RateLimitedError):
    """The tenant's daily budget is spent. Same status as a rate limit, different
    code: a client should back off for seconds in one case and until tomorrow in the other."""
    code = "quota_exceeded"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        headers=getattr(exc, "headers", None),
    )


class TransientIngestionError(Exception):
    """A background-ingestion failure that retrying might fix (embedding provider
    unavailable, database blip). Not an HTTP error: it never reaches a client, it
    tells the Celery task to retry with backoff."""
