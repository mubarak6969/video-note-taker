from chunker import chunk_pages, chunk_segments, chunk_text_by_chars


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


def test_chunk_text_by_chars_empty_text_returns_empty_list():
    assert chunk_text_by_chars("") == []
    assert chunk_text_by_chars("   \n  ") == []


def test_chunk_text_by_chars_short_text_becomes_one_chunk():
    chunks = chunk_text_by_chars("just a short sentence.", max_chars=1000, page=2)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "just a short sentence."
    assert chunks[0]["page"] == 2
    assert chunks[0]["start"] is None
    assert chunks[0]["end"] is None


def test_chunk_text_by_chars_splits_long_text_on_word_boundaries():
    text = " ".join(f"word{i}" for i in range(200))  # long text, plenty of spaces
    chunks = chunk_text_by_chars(text, max_chars=50, overlap_chars=10)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 50
        assert not c["text"].endswith(" ")


def test_chunk_text_by_chars_overlap_repeats_trailing_content():
    text = "AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD"
    chunks = chunk_text_by_chars(text, max_chars=22, overlap_chars=11)

    assert len(chunks) >= 2
    # The overlap window should carry some trailing text of chunk N into
    # the start of chunk N+1.
    assert chunks[0]["text"][-5:] in chunks[1]["text"]


def test_chunk_pages_stamps_each_chunk_with_its_source_page():
    pages = [
        {"page": 1, "text": "Introduction to the topic at hand."},
        {"page": 2, "text": "Deeper details continue on this page."},
    ]
    chunks = chunk_pages(pages, max_chars=1000)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2
    assert chunks[0]["start"] is None


def test_chunk_pages_skips_blank_pages():
    pages = [{"page": 1, "text": "   "}, {"page": 2, "text": "real content here"}]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 2
