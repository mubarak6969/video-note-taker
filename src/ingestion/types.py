"""The common normalized representation every source type is converted
into, before the existing chunker/embedder/vector_store pipeline takes
over unchanged regardless of where the content came from."""
from dataclasses import dataclass, field


@dataclass
class IngestedSource:
    source_id: str
    """Stable identifier: a YouTube video id, or a content hash for
    uploads - so re-ingesting the same source (even under a different
    filename) is recognized as a duplicate."""

    title: str
    source_type: str
    """One of: 'youtube_video', 'pdf_document', 'text_document',
    'audio_upload'."""

    origin: str
    """The source's URL (YouTube) or original filename (uploads) - stored
    for display and saved as the library entry's 'url' field."""

    full_text: str
    """The complete extracted/transcribed text, used as the transcript
    argument to notes_generator.generate_notes()."""

    raw_segments: list
    """Fine-grained normalized segments (dicts with a 'text' key, plus
    optionally 'start'/'end' or 'page') describing the source at its most
    natural granularity - Whisper's per-utterance segments for time-based
    sources, per-page entries for PDFs, or a single whole-text entry for
    plain text. Passed to generate_notes() for labeled notes."""

    chunks: list = field(default_factory=list)
    """Retrieval-ready chunks (dicts with 'text' and optionally
    'start'/'end'/'page'), already split via chunker.chunk_segments() or
    chunker.chunk_pages()/chunk_text_by_chars() as appropriate for this
    source type. embed_chunks() adds 'embedding' to each before saving."""
