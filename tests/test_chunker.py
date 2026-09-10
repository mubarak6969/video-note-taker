from chunker import chunk_segments


def test_empty_segments_returns_empty_list():
    assert chunk_segments([]) == []


def test_single_short_segment_becomes_one_chunk():
    segments = [{"start": 0.0, "end": 5.0, "text": "hello world"}]
    chunks = chunk_segments(segments, chunk_duration=30, overlap=5)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 5.0


def test_splits_into_multiple_chunks_by_duration():
    segments = [
        {"start": 0, "end": 10, "text": "one"},
        {"start": 10, "end": 20, "text": "two"},
        {"start": 20, "end": 35, "text": "three"},
        {"start": 35, "end": 45, "text": "four"},
    ]
    chunks = chunk_segments(segments, chunk_duration=30, overlap=0)

    assert len(chunks) == 2
    assert chunks[0]["end"] == 35
    assert chunks[1]["end"] == 45


def test_overlap_carries_trailing_segments_into_next_chunk():
    segments = [
        {"start": 0, "end": 15, "text": "a"},
        {"start": 15, "end": 30, "text": "b"},
        {"start": 30, "end": 40, "text": "c"},
    ]
    chunks = chunk_segments(segments, chunk_duration=30, overlap=10)

    assert len(chunks) == 2
    assert chunks[0]["end"] == 30
    assert chunks[1]["start"] == 15
    assert "b" in chunks[0]["text"]
    assert "b" in chunks[1]["text"]
