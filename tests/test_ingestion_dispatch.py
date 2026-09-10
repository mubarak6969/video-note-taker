import pytest

import ingestion
from ingestion.errors import IngestionError
from ingestion.types import IngestedSource


def _stub_source(source_type):
    return IngestedSource(
        source_id="stub",
        title="Stub",
        source_type=source_type,
        origin="stub",
        full_text="stub text",
        raw_segments=[{"text": "stub text", "start": None, "end": None, "page": None}],
        chunks=[{"text": "stub text", "start": None, "end": None, "page": None}],
    )


def test_dispatches_pdf_by_extension(monkeypatch):
    called = {}

    def fake_ingest_pdf(file_bytes, filename):
        called["args"] = (file_bytes, filename)
        return _stub_source("pdf_document")

    monkeypatch.setattr(ingestion, "ingest_pdf", fake_ingest_pdf)

    result = ingestion.ingest_uploaded_file("doc.PDF", b"bytes")
    assert result.source_type == "pdf_document"
    assert called["args"] == (b"bytes", "doc.PDF")


def test_dispatches_text_by_extension(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_text_upload", lambda b, f: _stub_source("text_document"))
    result = ingestion.ingest_uploaded_file("notes.txt", b"bytes")
    assert result.source_type == "text_document"


def test_dispatches_audio_by_extension_and_forwards_language(monkeypatch):
    called = {}

    def fake_ingest_audio(file_bytes, filename, language=None, model_size=None):
        called["language"] = language
        called["model_size"] = model_size
        return _stub_source("audio_upload")

    monkeypatch.setattr(ingestion, "ingest_audio_upload", fake_ingest_audio)

    result = ingestion.ingest_uploaded_file("clip.mp3", b"bytes", language="en", model_size="small")
    assert result.source_type == "audio_upload"
    assert called == {"language": "en", "model_size": "small"}


def test_unsupported_extension_raises_ingestion_error():
    with pytest.raises(IngestionError, match="Unsupported"):
        ingestion.ingest_uploaded_file("archive.zip", b"bytes")


def test_supported_upload_extensions_cover_pdf_txt_and_audio():
    extensions = set(ingestion.SUPPORTED_UPLOAD_EXTENSIONS)
    assert {"pdf", "txt", "mp3", "mp4", "wav"} <= extensions
    # st.file_uploader's `type=` param expects extensions without a dot.
    assert all(not e.startswith(".") for e in extensions)
