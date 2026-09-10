import importlib

import pytest


def _fresh_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "40")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "5")
    monkeypatch.setenv("UPLOAD_MAX_TEXT_MB", "1")
    import config

    importlib.reload(config)
    import ingestion.text_source as text_source

    importlib.reload(text_source)
    return text_source


def test_ingest_text_upload_happy_path(tmp_path, monkeypatch):
    text_source = _fresh_config(tmp_path, monkeypatch)

    content = ("This is a reasonably long piece of plain text. " * 5).encode("utf-8")
    source = text_source.ingest_text_upload(content, "notes.txt")

    assert source.source_type == "text_document"
    assert source.title == "notes"
    assert source.origin == "notes.txt"
    assert len(source.chunks) > 1
    assert all(c["start"] is None and c["page"] is None for c in source.chunks)
    assert source.full_text.startswith("This is a reasonably long piece")


def test_ingest_text_upload_source_id_is_content_hash(tmp_path, monkeypatch):
    text_source = _fresh_config(tmp_path, monkeypatch)
    from ingestion.hashing import hash_bytes

    content = b"identical bytes"
    a = text_source.ingest_text_upload(content, "a.txt")
    b = text_source.ingest_text_upload(content, "b.txt")

    assert a.source_id == b.source_id == hash_bytes(content)


def test_ingest_text_upload_rejects_empty_file(tmp_path, monkeypatch):
    text_source = _fresh_config(tmp_path, monkeypatch)

    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="empty"):
        text_source.ingest_text_upload(b"   \n\n  ", "empty.txt")


def test_ingest_text_upload_rejects_oversized_file(tmp_path, monkeypatch):
    text_source = _fresh_config(tmp_path, monkeypatch)
    from ingestion.errors import IngestionError

    big = b"x" * (2 * 1024 * 1024)  # 2 MB, over the 1 MB limit set above
    with pytest.raises(IngestionError, match="too large"):
        text_source.ingest_text_upload(big, "big.txt")


def test_ingest_text_upload_falls_back_on_bad_encoding(tmp_path, monkeypatch):
    text_source = _fresh_config(tmp_path, monkeypatch)

    # Bytes that aren't valid UTF-8 shouldn't crash ingestion.
    content = "café".encode("latin-1")
    source = text_source.ingest_text_upload(content, "latin1.txt")
    assert "caf" in source.full_text
