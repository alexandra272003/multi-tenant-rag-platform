from app.core.config import settings
from tests.conftest import LONG_TEXT, make_pdf


async def _upload(client, name, data, **params):
    return await client.post("/documents", files={"file": (name, data)}, params=params)


async def _doc(client, response):
    """Uploads are asynchronous: the 202 body is the document as it was ACCEPTED
    (status 'uploaded', no chunks). The test `client` runs the job inline, so one GET
    later the document is in its final state."""
    return (await client.get(f"/documents/{response.json()['id']}")).json()


async def test_ping_and_lab_page(client):
    assert (await client.get("/ping")).json()["message"] == "pong"
    page = await client.get("/")
    assert page.status_code == 200
    assert "Chunk lab" in page.text


async def test_upload_text_file_creates_document_and_chunks(client):
    resp = await _upload(client, "notes.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)
    assert resp.status_code == 202
    doc = await _doc(client, resp)
    assert doc["status"] == "ready"
    assert doc["source_type"] == "txt"
    assert doc["chunk_count"] > 1
    assert doc["chunk_size"] == 40

    chunks = (await client.get(f"/documents/{doc['id']}/chunks", params={"limit": 500})).json()
    assert len(chunks) == doc["chunk_count"]
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["page_number"] is None for c in chunks)


async def test_markdown_file_keeps_list_structure(client):
    md = b"# Title\n\n- alpha item\n- beta item\n\nA closing sentence."
    resp = await _upload(client, "doc.md", md, chunk_size=200, chunk_overlap=0)
    assert resp.status_code == 202
    chunks = (await client.get(f"/documents/{resp.json()['id']}/chunks")).json()
    assert "- alpha item\n- beta item" in chunks[0]["content"]


async def test_pdf_chunks_carry_page_numbers(client):
    pdf = make_pdf(
        [
            ["The first page describes apples and orchards."],
            ["The second page describes rivers and bridges."],
            ["The third page describes deserts and dunes."],
        ]
    )
    resp = await _upload(client, "book.pdf", pdf, chunk_size=100, chunk_overlap=0)
    assert resp.status_code == 202
    doc = await _doc(client, resp)
    assert doc["source_type"] == "pdf"
    assert doc["page_count"] == 3

    chunks = (await client.get(f"/documents/{doc['id']}/chunks")).json()
    by_page = {c["page_number"]: c["content"] for c in chunks}
    assert "apples" in by_page[1]
    assert "rivers" in by_page[2]
    assert "deserts" in by_page[3]


async def test_stored_offsets_index_into_the_cleaned_page_text(client):
    from app.ingestion.cleaning import clean_text

    resp = await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)
    chunks = (await client.get(f"/documents/{resp.json()['id']}/chunks", params={"limit": 500})).json()
    cleaned = clean_text(LONG_TEXT)
    for c in chunks:
        assert cleaned[c["char_start"] : c["char_end"]] == c["content"]


async def test_chunks_endpoint_paginates(client):
    doc = (await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)).json()
    first = (await client.get(f"/documents/{doc['id']}/chunks", params={"limit": 2, "offset": 0})).json()
    second = (await client.get(f"/documents/{doc['id']}/chunks", params={"limit": 2, "offset": 2})).json()
    assert [c["chunk_index"] for c in first] == [0, 1]
    assert [c["chunk_index"] for c in second] == [2, 3]


async def test_unsupported_file_type_is_rejected(client):
    resp = await _upload(client, "image.png", b"\x89PNG....")
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_file_type"


async def test_empty_file_is_rejected(client):
    resp = await _upload(client, "empty.txt", b"")
    assert resp.status_code == 422


async def test_pdf_without_text_ends_up_failed_with_a_helpful_message(client):
    """Parsing happens in the worker, so a bad file is no longer a 422 at upload: the
    upload is accepted and the DOCUMENT fails, with the reason on it."""
    blank = make_pdf([[""], [""]])
    resp = await _upload(client, "scan.pdf", blank)
    assert resp.status_code == 202
    doc = await _doc(client, resp)
    assert doc["status"] == "failed"
    assert "OCR" in doc["error_message"]
    assert doc["chunk_count"] == 0


async def test_corrupt_pdf_ends_up_failed(client):
    resp = await _upload(client, "broken.pdf", b"this is not a pdf at all")
    assert resp.status_code == 202
    doc = await _doc(client, resp)
    assert doc["status"] == "failed" and doc["error_message"]


async def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 100)
    resp = await _upload(client, "big.txt", b"x" * 500)
    assert resp.status_code == 413


async def test_overlap_not_smaller_than_size_is_rejected(client):
    resp = await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=20, chunk_overlap=20)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_chunk_params"


async def test_same_file_and_settings_twice_is_a_conflict(client):
    first = await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)
    second = await _upload(client, "renamed.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)
    assert second.status_code == 409
    assert second.json()["error"]["details"]["existing_document_id"] == first.json()["id"]


async def test_same_file_with_different_settings_is_allowed(client):
    a = await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=40, chunk_overlap=10)
    b = await _upload(client, "a.txt", LONG_TEXT.encode(), chunk_size=80, chunk_overlap=10)
    assert a.status_code == b.status_code == 202
    a_doc, b_doc = await _doc(client, a), await _doc(client, b)
    assert b_doc["chunk_count"] < a_doc["chunk_count"]  # bigger chunks -> fewer of them


async def test_list_get_and_delete_document(client):
    doc = (await _upload(client, "a.txt", LONG_TEXT.encode())).json()
    assert [d["id"] for d in (await client.get("/documents")).json()] == [doc["id"]]
    assert (await client.get(f"/documents/{doc['id']}")).status_code == 200

    assert (await client.delete(f"/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/documents/{doc['id']}")).status_code == 404
    assert (await client.get(f"/documents/{doc['id']}/chunks")).status_code == 404


async def test_deleting_a_document_allows_re_ingesting_it(client):
    doc = (await _upload(client, "a.txt", LONG_TEXT.encode())).json()
    await client.delete(f"/documents/{doc['id']}")
    assert (await _upload(client, "a.txt", LONG_TEXT.encode())).status_code == 202


async def test_missing_document_is_404(client):
    assert (await client.get("/documents/999")).status_code == 404


async def test_chunk_preview_does_not_save_anything(client):
    resp = await client.post(
        "/chunk-preview", json={"text": LONG_TEXT, "chunk_size": 40, "chunk_overlap": 10}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_count"] == len(body["chunks"]) > 1
    assert (await client.get("/documents")).json() == []


async def test_chunk_preview_rejects_bad_overlap(client):
    resp = await client.post(
        "/chunk-preview", json={"text": "hello world.", "chunk_size": 10, "chunk_overlap": 10}
    )
    assert resp.status_code == 422
