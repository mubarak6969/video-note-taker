def chunk_segments(segments, chunk_duration=30, overlap=5):
    """
    Groups Whisper segments into time-based chunks of ~chunk_duration seconds.

    Consecutive chunks share `overlap` seconds of trailing context so a
    sentence or idea that straddles a chunk boundary isn't split away from
    its context in both halves - this measurably improves retrieval quality
    for RAG search.
    """
    if not segments:
        return []

    chunks = []
    window = []
    window_start = segments[0]["start"]

    for segment in segments:
        window.append(segment)

        if segment["end"] - window_start >= chunk_duration:
            chunks.append({
                "text": " ".join(s["text"].strip() for s in window).strip(),
                "start": window_start,
                "end": segment["end"],
            })

            # Keep the trailing `overlap` seconds of segments as the start
            # of the next window instead of discarding them.
            cutoff = segment["end"] - overlap
            window = [s for s in window if s["end"] > cutoff]
            window_start = window[0]["start"] if window else segment["end"]

    if window:
        chunks.append({
            "text": " ".join(s["text"].strip() for s in window).strip(),
            "start": window_start,
            "end": segments[-1]["end"],
        })

    return chunks


def chunk_text_by_chars(text, max_chars=1200, overlap_chars=150, page=None):
    """
    Splits plain text into overlapping chunks by character count, breaking
    on a word boundary where possible.

    Used for non-timed sources (PDF pages, text file uploads) where the
    time-based chunk_segments() above doesn't apply - there is no
    start/end, only an optional page number. `page`, if given, is stamped
    onto every resulting chunk so citations can point back to it.
    """
    text = (text or "").strip()
    if not text:
        return []

    spans = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)

    chunks = []
    for span_start, span_end in spans:
        chunk_text = text[span_start:span_end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "start": None, "end": None, "page": page})
    return chunks


def chunk_pages(pages, max_chars=1200, overlap_chars=150):
    """
    Chunks a list of {"page": int, "text": str} entries (e.g. extracted PDF
    pages) into overlapping character-based chunks, each stamped with the
    page number it came from.
    """
    chunks = []
    for entry in pages:
        chunks.extend(
            chunk_text_by_chars(
                entry["text"], max_chars=max_chars, overlap_chars=overlap_chars, page=entry.get("page")
            )
        )
    return chunks
