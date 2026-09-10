"""PDF ingestion: extracts text per page (page numbers preserved as chunk
metadata for citations) and chunks it with chunker.chunk_pages()."""
import io
import logging
import os

from pypdf import PdfReader

import config
from chunker import chunk_pages

from .errors import IngestionError
from .hashing import hash_bytes
from .types import IngestedSource

logger = logging.getLogger(__name__)


def ingest_pdf(file_bytes: bytes, filename: str) -> IngestedSource:
    max_bytes = config.UPLOAD_MAX_PDF_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise IngestionError(
            f"PDF is too large ({len(file_bytes) / 1e6:.1f} MB) - "
            f"max {config.UPLOAD_MAX_PDF_MB} MB."
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise IngestionError(f"Couldn't read this PDF: {e}") from e

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise IngestionError("This PDF is password-protected and can't be read.")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", i, filename, e)
            text = ""
        if text.strip():
            pages.append({"page": i, "text": text})

    if not pages:
        raise IngestionError(
            "No extractable text was found in this PDF - it may be a scanned "
            "image without OCR text, which isn't supported yet."
        )

    chunks = chunk_pages(
        pages, max_chars=config.CHUNK_MAX_CHARS, overlap_chars=config.CHUNK_OVERLAP_CHARS
    )
    if not chunks:
        raise IngestionError("This PDF had extractable text, but no usable chunks came out of it.")

    source_id = hash_bytes(file_bytes)
    title = os.path.splitext(filename)[0]
    full_text = "\n\n".join(p["text"].strip() for p in pages)

    return IngestedSource(
        source_id=source_id,
        title=title,
        source_type="pdf_document",
        origin=filename,
        full_text=full_text,
        raw_segments=pages,
        chunks=chunks,
    )
