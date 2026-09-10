"""Source ingestion layer: converts YouTube URLs and uploaded files (PDF,
text, audio/video) into a common IngestedSource - normalized text/segments
ready for the existing chunk -> embed -> ChromaDB -> hybrid-retrieval ->
RAG pipeline, unchanged regardless of where the content came from.

app.py should only ever call the functions re-exported here (plus
vector_store.save_source/source_exists) - it should never need to import
from an individual `*_source` module directly, which is what keeps
source-specific logic out of app.py.
"""
import os

from .audio_source import SUPPORTED_EXTENSIONS as _AUDIO_EXTENSIONS
from .audio_source import ingest_audio_upload
from .errors import IngestionError
from .hashing import hash_bytes
from .pdf_source import ingest_pdf
from .text_source import ingest_text_upload
from .types import IngestedSource
from .youtube_source import ingest_youtube_url

# Extensions (without the leading dot) accepted by st.file_uploader's
# `type=` filter and validated again in ingest_uploaded_file().
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt"}
SUPPORTED_UPLOAD_EXTENSIONS = sorted(
    e.lstrip(".") for e in (PDF_EXTENSIONS | TEXT_EXTENSIONS | _AUDIO_EXTENSIONS)
)


def ingest_uploaded_file(
    filename: str, file_bytes: bytes, language: str = None, model_size: str = None
) -> IngestedSource:
    """Dispatches an uploaded file to the right ingestion handler by
    extension. Raises IngestionError for anything unsupported."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in PDF_EXTENSIONS:
        return ingest_pdf(file_bytes, filename)
    if ext in TEXT_EXTENSIONS:
        return ingest_text_upload(file_bytes, filename)
    if ext in _AUDIO_EXTENSIONS:
        return ingest_audio_upload(file_bytes, filename, language=language, model_size=model_size)

    raise IngestionError(
        f"Unsupported file type '{ext or filename}'. "
        f"Supported: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}."
    )


__all__ = [
    "IngestedSource",
    "IngestionError",
    "hash_bytes",
    "ingest_youtube_url",
    "ingest_uploaded_file",
    "SUPPORTED_UPLOAD_EXTENSIONS",
]
