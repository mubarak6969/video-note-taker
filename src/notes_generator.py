import logging

import config
from llm_client import chat_completion

logger = logging.getLogger(__name__)

SOURCE_TYPE_LABELS = {
    "youtube_video": "YouTube video",
    "audio_upload": "audio/video recording",
    "pdf_document": "PDF document",
    "text_document": "text document",
}

REFERENCE_HINTS = {
    "youtube_video": "timestamps (e.g. [12:34])",
    "audio_upload": "timestamps (e.g. [12:34])",
    "pdf_document": "page numbers (e.g. [p. 4])",
    "text_document": "the source text (no timestamps or pages are available)",
}


def _label_segment(segment) -> str:
    """Formats a segment's position marker: a timestamp for time-based
    sources (YouTube, audio/video uploads), a page number for page-based
    sources (PDF), or nothing for plain text with neither."""
    if segment.get("start") is not None:
        return f"[{round(segment['start'], 2)}s]: "
    if segment.get("page") is not None:
        return f"[p.{segment['page']}]: "
    return ""


def generate_notes(transcript, segments, source_type: str = "youtube_video", title: str = None):
    """`segments` is any list of normalized segment dicts (a 'text' key,
    plus optionally 'start'/'end' or 'page') - the same shape produced by
    every src/ingestion/ source type, not just Whisper's YouTube output.
    `source_type`/`title` steer the notes toward the right vocabulary and
    reference style (timestamps vs. pages vs. neither) for that source.
    """
    logger.info("Generating notes via Groq (%d segments, source_type=%s)", len(segments), source_type)

    # The labeled text below already contains the full transcript, so we
    # don't send the plain `transcript` string too - the original prompt did,
    # roughly doubling token usage/cost for no benefit.
    content = "\n".join(
        f"{_label_segment(segment)}{segment['text'].strip()}" for segment in segments
    )

    source_label = SOURCE_TYPE_LABELS.get(source_type, "source")
    reference_hint = REFERENCE_HINTS.get(source_type, "the source text")
    title_line = f'Title: "{title}"\n' if title else ""

    prompt = f"""You are an expert note-taker who produces study-quality notes from source material.

Source type: {source_label}
{title_line}
Here is the content, labeled with {reference_hint} where available:
{content}

Produce notes with exactly these five Markdown sections, using only information
present in the content above - never invent facts or references that aren't there:

## Overview
A 2-4 sentence summary of what this source covers and why it matters.

## Key Concepts
The core ideas, terms, or topics introduced - each a short bullet with a
one-line explanation.

## Important Points
The specific facts, claims, numbers, or details worth remembering, as
bullets. Where the source provides one, include its reference ({reference_hint}).

## Actionable Takeaways
Concrete things a reader/listener should do, try, remember, or follow up
on. If the source is purely informational with nothing actionable, say so
briefly instead of inventing items.

## Source References
A short list of the most important moments or sections, each with its
{reference_hint}. Omit this section entirely if the source has neither
timestamps nor page numbers.
"""

    notes = chat_completion(prompt, model=config.GROQ_MODEL)
    logger.info("Notes generated (%d chars)", len(notes))
    return notes
