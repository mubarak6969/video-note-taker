import logging

import config
from llm_client import chat_completion

logger = logging.getLogger(__name__)


def generate_notes(transcript, segments):
    logger.info("Generating notes via Groq (%d segments)", len(segments))

    # The timestamped text below already contains the full transcript, so we
    # don't send the plain `transcript` string too - the original prompt did,
    # roughly doubling token usage/cost for no benefit.
    timestamps = "\n".join(
        f"[{round(segment['start'], 2)}s]: {segment['text'].strip()}" for segment in segments
    )

    prompt = f"""
You are a helpful assistant that creates structured notes from video transcripts.

Here is the transcript with timestamps:
{timestamps}

Please provide:
1. STRUCTURED NOTES: Key points organized by topic
2. KEY TIMESTAMPS: Most important moments with their times
3. ACTION ITEMS: Things to do or remember from this video

Format your response clearly with these 3 sections, using Markdown.
"""

    notes = chat_completion(prompt, model=config.GROQ_MODEL)
    logger.info("Notes generated (%d chars)", len(notes))
    return notes
