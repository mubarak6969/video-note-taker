import logging

import config
from llm_client import chat_completion

logger = logging.getLogger(__name__)


def _label_segment(segment) -> str:
    """Formats a segment's position marker: a timestamp for time-based
    sources (YouTube, audio/video uploads), a page number for page-based
    sources (PDF), or nothing for plain text with neither."""
    if segment.get("start") is not None:
        return f"[{round(segment['start'], 2)}s]: "
    if segment.get("page") is not None:
        return f"[p.{segment['page']}]: "
    return ""


def generate_notes(transcript, segments):
    """`segments` is any list of normalized segment dicts (a 'text' key,
    plus optionally 'start'/'end' or 'page') - the same shape produced by
    every src/ingestion/ source type, not just Whisper's YouTube output."""
    logger.info("Generating notes via Groq (%d segments)", len(segments))

    # The labeled text below already contains the full transcript, so we
    # don't send the plain `transcript` string too - the original prompt did,
    # roughly doubling token usage/cost for no benefit.
    timestamps = "\n".join(
        f"{_label_segment(segment)}{segment['text'].strip()}" for segment in segments
    )

    prompt = f"""
You are a helpful assistant that creates structured notes from source material
(a video transcript, a document, or an audio recording).

Here is the content, labeled with timestamps and/or page numbers where available:
{timestamps}

Please provide:
1. STRUCTURED NOTES: Key points organized by topic
2. KEY REFERENCES: Most important points with their timestamp or page reference
3. ACTION ITEMS: Things to do or remember from this source

Format your response clearly with these 3 sections, using Markdown.
"""

    notes = chat_completion(prompt, model=config.GROQ_MODEL)
    logger.info("Notes generated (%d chars)", len(notes))
    return notes
