from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.schemas import ChunkPreviewRequest, ChunkPreviewResponse
from app.services import ingestion_service

router = APIRouter(tags=["chunk-lab"])

_PAGE = Path(__file__).resolve().parent.parent / "static" / "chunk-lab.html"


@router.post("/chunk-preview", response_model=ChunkPreviewResponse)
async def chunk_preview(payload: ChunkPreviewRequest):
    """Chunk text with the given settings and return the result. Saves nothing."""
    return ingestion_service.preview_chunks(
        payload.text, payload.chunk_size, payload.chunk_overlap, payload.clean
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chunk_lab_page():
    # Served by the API itself, so the page and the API share one origin --
    # no CORS setup needed, unlike opening an HTML file straight from disk.
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))
