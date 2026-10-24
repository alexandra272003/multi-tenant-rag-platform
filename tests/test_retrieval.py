from tests.conftest import LONG_TEXT, install_topic_embeddings


async def test_retrieve_ranks_by_relevance(client, monkeypatch):
    install_topic_embeddings(monkeypatch)

    cat_doc = "Cats are small feline mammals that enjoy napping in sunny spots for hours."
    finance_doc = "Quarterly revenue increased due to strong finance department performance this year."

    await client.post(
        "/documents", files={"file": ("cats.txt", cat_doc.encode())},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )
    await client.post(
        "/documents", files={"file": ("finance.txt", finance_doc.encode())},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    resp = await client.post("/retrieve", json={"query": "Tell me about kittens", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["filename"] == "cats.txt"
    assert results[0]["distance"] < results[1]["distance"]


async def test_retrieve_document_id_filter_excludes_other_documents(client, monkeypatch):
    install_topic_embeddings(monkeypatch)

    doc1 = (
        await client.post(
            "/documents",
            files={"file": ("cats1.txt", b"Cats are feline mammals that nap in the sun.")},
            params={"chunk_size": 500, "chunk_overlap": 0},
        )
    ).json()
    doc2 = (
        await client.post(
            "/documents",
            files={"file": ("cats2.txt", b"A second document, also about cats and kittens.")},
            params={"chunk_size": 500, "chunk_overlap": 0},
        )
    ).json()

    resp = await client.post(
        "/retrieve", json={"query": "kittens", "top_k": 5, "document_id": doc2["id"]}
    )
    results = resp.json()["results"]
    assert len(results) >= 1
    assert all(r["document_id"] == doc2["id"] for r in results)
    assert doc1["id"] not in {r["document_id"] for r in results}


async def test_retrieve_source_type_filter_excludes_non_matching_documents(client, monkeypatch):
    install_topic_embeddings(monkeypatch)

    await client.post(
        "/documents",
        files={"file": ("cats.txt", b"Cats napping is common feline behavior.")},
        params={"chunk_size": 500, "chunk_overlap": 0},
    )

    resp = await client.post(
        "/retrieve", json={"query": "cats", "top_k": 5, "source_type": "pdf"}
    )
    assert resp.json()["results"] == []  # only a .txt was uploaded; the pdf filter excludes it


async def test_retrieve_top_k_limits_result_count(client, monkeypatch):
    install_topic_embeddings(monkeypatch)

    await client.post(
        "/documents",
        files={"file": ("notes.txt", LONG_TEXT.encode())},
        params={"chunk_size": 40, "chunk_overlap": 10},
    )
    resp = await client.post("/retrieve", json={"query": "topic", "top_k": 2})
    assert len(resp.json()["results"]) <= 2


async def test_retrieve_rejects_empty_query(client):
    resp = await client.post("/retrieve", json={"query": ""})
    assert resp.status_code == 422


async def test_retrieve_against_empty_corpus_returns_no_results(client):
    resp = await client.post("/retrieve", json={"query": "anything at all"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []
