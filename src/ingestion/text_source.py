"""Plain text file ingestion: decodes the file and chunks it with
chunker.chunk_text_by_chars() - no timestamps or pages, just the text."""
import logging
import os

import config
from chunker import chunk_text_by_chars

from .errors import IngestionError
from .hashing import hash_bytes
from .types import IngestedSource

logger = logging.getLogger(__name__)


def _decode(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: never fail on encoding alone, just replace bad bytes.
    return file_bytes.decode("latin-1", errors="replace")


def ingest_text_upload(file_bytes: bytes, filename: str) -> IngestedSource:
    max_bytes = config.UPLOAD_MAX_TEXT_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise IngestionError(
            f"Text file is too large ({len(file_bytes) / 1e6:.1f} MB) - "
            f"max {config.UPLOAD_MAX_TEXT_MB} MB."
        )

    text = _decode(file_bytes).strip()
    if not text:
        raise IngestionError("This text file appears to be empty.")

    chunks = chunk_text_by_chars(
        text, max_chars=config.CHUNK_MAX_CHARS, overlap_chars=config.CHUNK_OVERLAP_CHARS
    )
    if not chunks:
        raise IngestionError("This text file had content, but no usable chunks came out of it.")

    source_id = hash_bytes(file_bytes)
    title = os.path.splitext(filename)[0]

    return IngestedSource(
        source_id=source_id,
        title=title,
        source_type="text_document",
        origin=filename,
        full_text=text,
        raw_segments=[{"text": text, "start": None, "end": None, "page": None}],
        chunks=chunks,
    )
