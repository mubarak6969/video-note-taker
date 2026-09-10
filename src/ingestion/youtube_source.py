"""YouTube ingestion - a thin wrapper around the existing downloader +
transcriber + chunker, unchanged from the pipeline that existed before
this package. Kept separate so app.py doesn't need to know which pipeline
a URL vs. an uploaded file goes through."""
import logging

import config
from chunker import chunk_segments
from downloader import download_audio
from transcriber import transcribe_audio

from .errors import IngestionError
from .types import IngestedSource

logger = logging.getLogger(__name__)


def ingest_youtube_url(url: str, language: str = None, model_size: str = None) -> IngestedSource:
    """Downloads and transcribes a YouTube video. Raises downloader.DownloadError
    if the download fails, or IngestionError if no speech was detected."""
    video_id, title, file_path = download_audio(url)

    transcript, raw_segments, _ = transcribe_audio(
        file_path, language=language, model_size=model_size or config.WHISPER_MODEL_SIZE
    )

    chunks = chunk_segments(
        raw_segments,
        chunk_duration=config.CHUNK_DURATION_SECONDS,
        overlap=config.CHUNK_OVERLAP_SECONDS,
    )
    if not chunks:
        raise IngestionError("No speech was detected in this video.")

    return IngestedSource(
        source_id=video_id,
        title=title,
        source_type="youtube_video",
        origin=url,
        full_text=transcript,
        raw_segments=raw_segments,
        chunks=chunks,
    )
