"""Turn an uploaded file into pages of raw text."""
import io
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.errors import InvalidDocumentError, UnsupportedFileTypeError

SUPPORTED = {".txt": "txt", ".md": "md", ".pdf": "pdf"}


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based for PDFs; None for plain text / markdown
    text: str


def source_type_for(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or filename}'",
            details={"supported": sorted(SUPPORTED)},
        )
    return SUPPORTED[ext]


def extract_pages(filename: str, data: bytes) -> list[Page]:
    kind = source_type_for(filename)
    if kind in ("txt", "md"):
        return [Page(None, data.decode("utf-8", errors="replace"))]

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidDocumentError("PDF is password-protected")
        return [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    except PyPdfError as exc:
        raise InvalidDocumentError(f"Could not read PDF: {exc}") from exc
