import rag_chat


def test_describe_source_youtube_gets_clickable_timestamp_link():
    chunk = {"start": 90, "end": 125, "title": "Demo Talk", "video_id": "abc123", "source_type": "youtube_video"}
    info = rag_chat.describe_source(chunk)

    assert info["title"] == "Demo Talk"
    assert info["position"] == "1:30-2:05"
    assert info["link"] == "https://www.youtube.com/watch?v=abc123&t=90s"
    assert info["source_type"] == "youtube_video"


def test_describe_source_pdf_has_page_position_and_no_link():
    chunk = {"page": 5, "title": "Manual", "video_id": "pdf1", "source_type": "pdf_document"}
    info = rag_chat.describe_source(chunk)

    assert info["position"] == "page 5"
    assert info["link"] is None


def test_describe_source_plain_text_has_no_position_and_no_link():
    chunk = {"title": "Notes", "video_id": "txt1", "source_type": "text_document"}
    info = rag_chat.describe_source(chunk)

    assert info["position"] == ""
    assert info["link"] is None


def test_describe_source_audio_upload_has_timestamp_but_no_link():
    # Only YouTube chunks get a deep link - an uploaded audio file has no
    # public URL to link back to.
    chunk = {"start": 10, "end": 20, "title": "Podcast", "video_id": "aud1", "source_type": "audio_upload"}
    info = rag_chat.describe_source(chunk)

    assert info["position"] == "0:10-0:20"
    assert info["link"] is None


def test_extract_cited_sources_returns_only_cited_chunks_in_citation_order():
    chunks = [
        {"text": "first fact", "start": 0, "end": 10, "title": "A", "video_id": "a", "source_type": "youtube_video"},
        {"text": "second fact", "page": 4, "title": "B", "video_id": "b", "source_type": "pdf_document"},
        {"text": "third fact", "page": 9, "title": "C", "video_id": "c", "source_type": "pdf_document"},
    ]
    answer = "The second fact matters most (Source 2, page 4). Also relevant: (Source 1, 0:00-0:10)."

    cited = rag_chat.extract_cited_sources(answer, chunks)

    assert [c["title"] for c in cited] == ["B", "A"]


def test_extract_cited_sources_deduplicates_repeated_citations():
    chunks = [{"text": "x", "start": 0, "end": 1, "title": "X", "video_id": "x", "source_type": "youtube_video"}]
    answer = "As shown (Source 1, 0:00-0:01), and again (Source 1, 0:00-0:01)."

    cited = rag_chat.extract_cited_sources(answer, chunks)
    assert len(cited) == 1


def test_extract_cited_sources_falls_back_to_all_chunks_when_no_citation_found():
    chunks = [{"text": "a", "start": 0, "end": 1, "title": "X", "video_id": "x", "source_type": "youtube_video"}]
    cited = rag_chat.extract_cited_sources("An answer with no citation markers at all.", chunks)
    assert cited == chunks


def test_extract_cited_sources_ignores_out_of_range_citation_numbers():
    chunks = [{"text": "a", "start": 0, "end": 1, "title": "X", "video_id": "x", "source_type": "youtube_video"}]
    # The model hallucinated "Source 5" when only one chunk was provided.
    cited = rag_chat.extract_cited_sources("See (Source 5, 0:00-0:01).", chunks)
    assert cited == chunks


def test_extract_cited_sources_empty_chunks_returns_empty_list():
    assert rag_chat.extract_cited_sources("(Source 1)", []) == []


def test_answer_question_prompt_instructs_citation_and_grounding_and_followups():
    captured = {}

    def fake_chat_completion(prompt, model):
        captured["prompt"] = prompt
        return "ANSWER"

    import unittest.mock

    with unittest.mock.patch.object(rag_chat, "chat_completion", fake_chat_completion):
        chunks = [{"text": "the sky is blue", "start": 0, "end": 5, "title": "T", "video_id": "v", "source_type": "youtube_video"}]
        rag_chat.answer_question("what color is the sky?", chunks)

    prompt = captured["prompt"]
    assert "cite it by its source label" in prompt
    assert "follow-up question" in prompt.lower()
    assert "Ground your answer strictly" in prompt
