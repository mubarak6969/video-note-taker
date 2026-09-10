import importlib

import pytest


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    """Stands in for pypdf.PdfReader so these tests never parse a real
    PDF - they only exercise ingestion's own logic (page/chunk metadata,
    validation)."""

    pages_to_return = [FakePage("Chapter one covers apples in detail."), FakePage("Chapter two covers oranges.")]
    is_encrypted = False
    decrypt_should_fail = False

    def __init__(self, _stream):
        self.pages = FakeReader.pages_to_return
        self.is_encrypted = FakeReader.is_encrypted

    def decrypt(self, _password):
        if FakeReader.decrypt_should_fail:
            raise ValueError("bad password")


def _fresh_pdf_source(monkeypatch, pages=None, encrypted=False, decrypt_fails=False, max_pdf_mb=None):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "100")
    if max_pdf_mb is not None:
        monkeypatch.setenv("UPLOAD_MAX_PDF_MB", str(max_pdf_mb))
    import config

    importlib.reload(config)
    import ingestion.pdf_source as pdf_source

    importlib.reload(pdf_source)

    FakeReader.pages_to_return = pages if pages is not None else [
        FakePage("Chapter one covers apples in detail."),
        FakePage("Chapter two covers oranges."),
    ]
    FakeReader.is_encrypted = encrypted
    FakeReader.decrypt_should_fail = decrypt_fails
    monkeypatch.setattr(pdf_source, "PdfReader", FakeReader)
    return pdf_source


def test_ingest_pdf_extracts_pages_and_stamps_page_metadata(monkeypatch):
    pdf_source = _fresh_pdf_source(monkeypatch)

    source = pdf_source.ingest_pdf(b"%PDF-fake-bytes", "manual.pdf")

    assert source.source_type == "pdf_document"
    assert source.title == "manual"
    assert source.origin == "manual.pdf"
    assert len(source.chunks) >= 2
    pages_seen = {c["page"] for c in source.chunks}
    assert pages_seen == {1, 2}
    assert all(c["start"] is None for c in source.chunks)
    assert "apples" in source.full_text and "oranges" in source.full_text


def test_ingest_pdf_source_id_is_content_hash(monkeypatch):
    pdf_source = _fresh_pdf_source(monkeypatch)
    from ingestion.hashing import hash_bytes

    data = b"%PDF-fake-bytes-v2"
    source = pdf_source.ingest_pdf(data, "report.pdf")
    assert source.source_id == hash_bytes(data)


def test_ingest_pdf_rejects_when_no_extractable_text(monkeypatch):
    pdf_source = _fresh_pdf_source(monkeypatch, pages=[FakePage(""), FakePage("   ")])
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="No extractable text"):
        pdf_source.ingest_pdf(b"%PDF-scanned-image", "scanned.pdf")


def test_ingest_pdf_rejects_oversized_file(monkeypatch):
    pdf_source = _fresh_pdf_source(monkeypatch, max_pdf_mb=0)
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="too large"):
        pdf_source.ingest_pdf(b"some bytes", "big.pdf")


def test_ingest_pdf_rejects_undecryptable_encrypted_file(monkeypatch):
    pdf_source = _fresh_pdf_source(monkeypatch, encrypted=True, decrypt_fails=True)
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="password-protected"):
        pdf_source.ingest_pdf(b"%PDF-encrypted", "locked.pdf")


def test_ingest_pdf_continues_past_a_page_that_fails_to_extract(monkeypatch):
    class BrokenPage(FakePage):
        def extract_text(self):
            raise RuntimeError("corrupt page stream")

    pdf_source = _fresh_pdf_source(monkeypatch, pages=[BrokenPage(""), FakePage("Readable page text here.")])

    source = pdf_source.ingest_pdf(b"%PDF-partial", "partial.pdf")
    assert any(c["page"] == 2 for c in source.chunks)
