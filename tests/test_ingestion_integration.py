"""Integration-style tests: do PDF/text/audio sources, once saved through
vector_store, behave correctly through the existing hybrid retrieval +
reranking pipeline, side by side with YouTube sources - and is duplicate
ingestion actually prevented end to end."""
import importlib


def _fresh_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import config
    import retrieval
    import vector_store

    importlib.reload(config)
    importlib.reload(vector_store)
    return retrieval, vector_store


def _pdf_chunk(text, page, embedding):
    return {"text": text, "start": None, "end": None, "page": page, "embedding": embedding}


def _video_chunk(text, start, end, embedding):
    return {"text": text, "start": start, "end": end, "embedding": embedding}


def test_duplicate_upload_is_recognized_by_content_hash_not_filename(tmp_path, monkeypatch):
    _, vector_store = _fresh_stack(tmp_path, monkeypatch)
    from ingestion.hashing import hash_bytes

    file_bytes = b"the exact same pdf bytes"
    source_id = hash_bytes(file_bytes)

    assert not vector_store.source_exists(source_id)

    vector_store.save_source(
        source_id, "Original Name", "original.pdf", "notes",
        [_pdf_chunk("some page content", 1, [1.0, 0.0, 0.0])],
        source_type="pdf_document",
    )

    # Re-"uploading" the identical bytes under a different filename hashes
    # to the same source_id, so it's recognized as the same source.
    reuploaded_id = hash_bytes(file_bytes)
    assert reuploaded_id == source_id
    assert vector_store.source_exists(reuploaded_id)
    assert len(vector_store.list_sources()) == 1


def test_pdf_source_metadata_preserved_through_retrieval(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_source(
        "pdf1", "User Manual", "manual.pdf", "notes",
        [_pdf_chunk("installation instructions are on this page", 3, [1.0, 0.0, 0.0])],
        source_type="pdf_document",
    )

    results = retrieval.retrieve("installation instructions", [1.0, 0.0, 0.0], video_id="pdf1", top_k=1)

    assert len(results) == 1
    chunk = results[0]
    assert chunk["source_type"] == "pdf_document"
    assert chunk["page"] == 3
    assert chunk["start"] is None
    assert chunk["title"] == "User Manual"


def test_single_source_scope_excludes_other_source_types(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_source(
        "pdf1", "PDF Doc", "doc.pdf", "notes",
        [_pdf_chunk("pdf content about rivers", 1, [1.0, 0.0, 0.0])],
        source_type="pdf_document",
    )
    vector_store.save_source(
        "yt1", "YouTube Video", "http://x/1", "notes",
        [_video_chunk("video content about rivers", 0, 10, [1.0, 0.0, 0.0])],
        source_type="youtube_video",
    )

    pdf_only = retrieval.retrieve("rivers", [1.0, 0.0, 0.0], video_id="pdf1", top_k=5)
    assert all(c["source_type"] == "pdf_document" for c in pdf_only)
    assert all(c["video_id"] == "pdf1" for c in pdf_only)


def test_library_wide_retrieval_spans_mixed_source_types(tmp_path, monkeypatch):
    retrieval, vector_store = _fresh_stack(tmp_path, monkeypatch)

    vector_store.save_source(
        "pdf1", "PDF Doc", "doc.pdf", "notes",
        [_pdf_chunk("quarterly earnings report content", 5, [1.0, 0.0, 0.0])],
        source_type="pdf_document",
    )
    vector_store.save_source(
        "yt1", "YouTube Video", "http://x/1", "notes",
        [_video_chunk("earnings call recording content", 0, 10, [0.9, 0.1, 0.0])],
        source_type="youtube_video",
    )
    vector_store.save_source(
        "txt1", "Text Notes", "notes.txt", "notes",
        [{"text": "plain text notes about earnings", "start": None, "end": None, "page": None, "embedding": [0.8, 0.2, 0.0]}],
        source_type="text_document",
    )

    results = retrieval.retrieve("earnings", [1.0, 0.0, 0.0], video_id=None, top_k=5)

    source_types = {c["source_type"] for c in results}
    assert source_types == {"pdf_document", "youtube_video", "text_document"}


def test_backward_compatible_with_chunks_saved_before_page_field_existed(tmp_path, monkeypatch):
    """Simulates rows written by the pre-ingestion-upgrade version of
    vector_store, which never wrote 'source_id' or 'page' into metadata -
    _chunk_from_metadata() must still handle them without KeyErrors."""
    _, vector_store = _fresh_stack(tmp_path, monkeypatch)

    old_style_metadata = {
        "video_id": "legacy1",
        "title": "Legacy Video",
        "start": 0,
        "end": 10,
        "source_type": "youtube_video",
        # no "source_id", no "page" - as an old row would look
    }
    chunk = vector_store._chunk_from_metadata("legacy1_0", "legacy chunk text", old_style_metadata)

    assert chunk["video_id"] == "legacy1"
    assert chunk["source_id"] == "legacy1"  # falls back to video_id
    assert chunk["start"] == 0
    assert chunk["end"] == 10
    assert chunk["page"] is None
