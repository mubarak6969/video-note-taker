"""Audio/video file upload ingestion: reuses the exact same Whisper
transcription (transcriber.transcribe_audio) and chunker
(chunker.chunk_segments) as the YouTube pipeline - the only difference is
where the audio file comes from."""
import logging
import os

import config
from chunker import chunk_segments
from transcriber import transcribe_audio

from .errors import IngestionError
from .hashing import hash_bytes
from .types import IngestedSource

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg", ".flac"}


def ingest_audio_upload(
    file_bytes: bytes, filename: str, language: str = None, model_size: str = None
) -> IngestedSource:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported audio/video format: '{ext or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    max_bytes = config.UPLOAD_MAX_AUDIO_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise IngestionError(
            f"File is too large ({len(file_bytes) / 1e6:.1f} MB) - "
            f"max {config.UPLOAD_MAX_AUDIO_MB} MB."
        )

    source_id = hash_bytes(file_bytes)

    # Whisper needs a real file on disk; store uploads alongside YouTube
    # downloads (same git-ignored directory), named by content hash so a
    # re-uploaded duplicate reuses the same file instead of writing again.
    upload_dir = os.path.join(config.DOWNLOADS_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    stored_path = os.path.join(upload_dir, f"{source_id}{ext}")
    if not os.path.exists(stored_path):
        with open(stored_path, "wb") as f:
            f.write(file_bytes)

    transcript, raw_segments, _ = transcribe_audio(
        stored_path, language=language, model_size=model_size or config.WHISPER_MODEL_SIZE
    )

    chunks = chunk_segments(
        raw_segments,
        chunk_duration=config.CHUNK_DURATION_SECONDS,
        overlap=config.CHUNK_OVERLAP_SECONDS,
    )
    if not chunks:
        raise IngestionError("No speech was detected in this file.")

    title = os.path.splitext(filename)[0]

    return IngestedSource(
        source_id=source_id,
        title=title,
        source_type="audio_upload",
        origin=filename,
        full_text=transcript,
        raw_segments=raw_segments,
        chunks=chunks,
    )
