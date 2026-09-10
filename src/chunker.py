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
