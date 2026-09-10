import importlib


def _fresh_stack(tmp_path, monkeypatch):
    """Reimports config + vector_store pointed at an isolated temp DATA_DIR.
    retrieval.py and reranker.py hold references to the *modules*
    `config`/`vector_store`, so reloading those in place is picked up by
    them automatically without needing their own reload."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import config
    import retrieval
    import vector_store

    importlib.reload(config)
    importlib.reload(vector_store)
    return retrieval, vector_store


def _chunk(text, start, end, embedding):
    return {"text": text, "start": start, "end": end, "embedding": embedding}


def test_single_video_scope_excludes_other_videos(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_video(
        "v1", "Video One", "http://x/1", "notes1",
        [_chunk("cats are wonderful pets", 0, 10, [1.0, 0.0, 0.0])],
    )
    vector_store.save_video(
        "v2", "Video Two", "http://x/2", "notes2",
        [_chunk("dogs are loyal companions", 0, 10, [1.0, 0.0, 0.0])],
    )

    results = retrieval.retrieve("tell me about pets", [1.0, 0.0, 0.0], video_id="v1", top_k=5)

    assert len(results) == 1
    assert all(c["video_id"] == "v1" for c in results)
    assert results[0]["text"] == "cats are wonderful pets"


def test_library_scope_spans_multiple_videos(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_video(
        "v1", "Video One", "http://x/1", "notes1",
        [_chunk("cats are wonderful pets", 0, 10, [1.0, 0.0, 0.0])],
    )
    vector_store.save_video(
        "v2", "Video Two", "http://x/2", "notes2",
        [_chunk("dogs are loyal companions", 0, 10, [0.9, 0.1, 0.0])],
    )

    results = retrieval.retrieve("tell me about pets", [1.0, 0.0, 0.0], video_id=None, top_k=5)

    video_ids = {c["video_id"] for c in results}
    assert video_ids == {"v1", "v2"}


def test_metadata_is_preserved_on_every_retrieved_chunk(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_video(
        "v1", "My Keynote", "http://x/1", "notes",
        [_chunk("the launch date is march third", 72, 100, [1.0, 0.0, 0.0])],
    )

    results = retrieval.retrieve("when is the launch?", [1.0, 0.0, 0.0], video_id="v1", top_k=1)

    assert len(results) == 1
    chunk = results[0]
    for key in ("video_id", "title", "start", "end", "source_type", "text"):
        assert key in chunk, f"missing '{key}' in retrieved chunk"
    assert chunk["video_id"] == "v1"
    assert chunk["title"] == "My Keynote"
    assert chunk["start"] == 72
    assert chunk["end"] == 100
    assert chunk["source_type"] == "youtube_video"


def test_hybrid_retrieval_recovers_exact_term_missed_by_vector_search(tmp_path, monkeypatch):
    """A chunk containing a rare exact term (a product code) is embedded far
    from the query in vector space (simulating an embedding model that
    doesn't capture that term well), while three decoy chunks are embedded
    identically to the query but share no keywords with it. Pure vector
    search should miss the exact-term chunk entirely; hybrid search should
    recover it via BM25."""
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    query = "what is the product code zylotex7742?"
    query_embedding = [1.0, 0.0, 0.0]

    chunks = [
        _chunk("general discussion about weather and travel plans", 0, 10, [1.0, 0.0, 0.0]),
        _chunk("an unrelated chat about cooking recipes tonight", 10, 20, [1.0, 0.0, 0.0]),
        _chunk("more small talk about weekend sports results", 20, 30, [1.0, 0.0, 0.0]),
        _chunk("the product code is zylotex7742 and it ships in march", 30, 40, [0.0, 1.0, 0.0]),
    ]
    vector_store.save_video("v1", "Support Call", "http://x/1", "notes", chunks)

    vector_only = retrieval.hybrid_search(
        query, query_embedding, video_id="v1", candidate_count=2, mode="vector"
    )
    assert not any("zylotex7742" in c["text"] for c in vector_only), (
        "sanity check: pure vector search should miss the exact-term chunk here"
    )

    hybrid = retrieval.hybrid_search(
        query, query_embedding, video_id="v1", candidate_count=2, mode="hybrid"
    )
    assert any("zylotex7742" in c["text"] for c in hybrid), (
        "hybrid search should recover the exact-term chunk via keyword matching"
    )

    final = retrieval.retrieve(
        query, query_embedding, video_id="v1", top_k=2, candidate_count=2
    )
    assert any("zylotex7742" in c["text"] for c in final), (
        "the exact-term chunk should survive reranking into the final results"
    )


def test_keyword_only_mode_ignores_vector_similarity(tmp_path, monkeypatch):
    # BM25 needs enough documents for a rare term to earn a positive IDF, so
    # this uses a few decoys alongside the one relevant chunk (matches the
    # shape of the hybrid-recovery test above).
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    chunks = [
        _chunk("completely unrelated small talk", 0, 10, [1.0, 0.0, 0.0]),
        _chunk("another unrelated chat about the weather", 10, 20, [1.0, 0.0, 0.0]),
        _chunk("more small talk about the weekend", 20, 30, [1.0, 0.0, 0.0]),
        _chunk("the invoice total is 4821 dollars", 30, 40, [0.0, 1.0, 0.0]),
    ]
    vector_store.save_video("v1", "Call", "http://x/1", "notes", chunks)

    results = retrieval.hybrid_search(
        "invoice total 4821", [1.0, 0.0, 0.0], video_id="v1", candidate_count=5, mode="keyword"
    )

    assert len(results) == 1
    assert "4821" in results[0]["text"]


def test_retrieve_returns_empty_list_when_scope_has_no_chunks(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    assert retrieval.retrieve("anything", [1.0, 0.0, 0.0], video_id="missing-video") == []
    assert retrieval.hybrid_search("anything", [1.0, 0.0, 0.0], video_id=None) == []
