import logging

import config
from llm_client import chat_completion

logger = logging.getLogger(__name__)


def _format_timestamp(seconds) -> str:
    total = int(seconds or 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _format_position(chunk) -> str:
    """A human-readable pointer back to where in the source this chunk
    came from: a timestamp range for time-based sources (YouTube, audio/
    video uploads), a page number for page-based sources (PDF), or
    nothing for plain text with neither."""
    if chunk.get("start") is not None and chunk.get("end") is not None:
        return f"{_format_timestamp(chunk['start'])}-{_format_timestamp(chunk['end'])}"
    if chunk.get("page") is not None:
        return f"page {chunk['page']}"
    return ""


def _format_context(chunks) -> str:
    """Renders retrieved chunks as labeled, citable sources - each with a
    stable [Source N] label, the source's title, and a timestamp or page
    reference where available - so the model (and the UI) can point back
    to exactly where an answer came from, even when chunks come from
    different sources of different types (video, PDF, text, audio)."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("video_id") or "unknown source"
        position = _format_position(chunk)
        label = f"[Source {i} | \"{title}\"" + (f" | {position}]" if position else "]")
        blocks.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(blocks)


def answer_question(query, top_chunks, chat_history=None):
    """
    Takes the user's question + retrieved chunks (from retrieval.retrieve(),
    single-video or library-wide) and asks the LLM to answer using ONLY
    that context, citing sources by their [Source N] label. `chat_history`
    is an optional list of (question, answer) tuples from earlier turns, so
    follow-up questions ("what about the second point?") can be understood
    in context.
    """
    if not top_chunks:
        return (
            "I don't have enough information in the processed video(s) to answer that. "
            "Try rephrasing the question, or process the video that covers this topic."
        )

    context = _format_context(top_chunks)

    history_block = ""
    if chat_history:
        history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in chat_history[-3:])
        history_block = f"\nPREVIOUS CONVERSATION (for context on follow-ups):\n{history_text}\n"

    prompt = f"""You are answering questions using ONLY the CONTEXT below, which was retrieved from one or more processed sources (video transcripts, documents, or audio recordings).

Rules:
- Ground your answer strictly in the CONTEXT below. Do not use outside knowledge, and do not guess.
- If the CONTEXT does not contain enough information to answer fully, say so plainly instead of filling gaps.
- When a statement in your answer comes from the CONTEXT, cite it by its source label, e.g. "(Source 2, 4:12-4:40)" or "(Source 2, page 5)".
- If sources disagree or only partially cover the question, note that explicitly.
{history_block}
CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    logger.info("Asking Groq: %s", query)
    return chat_completion(prompt, model=config.GROQ_MODEL)
