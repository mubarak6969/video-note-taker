import notes_generator
import rag_chat


def test_generate_notes_calls_llm_with_timestamps(monkeypatch):
    captured = {}

    def fake_chat_completion(prompt, model):
        captured["prompt"] = prompt
        captured["model"] = model
        return "NOTES OUTPUT"

    monkeypatch.setattr(notes_generator, "chat_completion", fake_chat_completion)

    segments = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = notes_generator.generate_notes("hello", segments)

    assert result == "NOTES OUTPUT"
    assert "[0.0s]: hello" in captured["prompt"]
    # The full plain-text transcript should NOT be duplicated into the
    # prompt anymore - only the timestamped version is sent.
    assert captured["prompt"].count("hello") == 1


def test_generate_notes_prompt_has_all_five_structured_sections(monkeypatch):
    captured = {}
    monkeypatch.setattr(notes_generator, "chat_completion", lambda prompt, model: captured.update(prompt=prompt) or "NOTES")

    segments = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    notes_generator.generate_notes("hello", segments)

    for heading in ("## Overview", "## Key Concepts", "## Important Points", "## Actionable Takeaways", "## Source References"):
        assert heading in captured["prompt"], f"missing section: {heading}"


def test_generate_notes_uses_page_reference_hint_for_pdf_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(notes_generator, "chat_completion", lambda prompt, model: captured.update(prompt=prompt) or "NOTES")

    segments = [{"page": 3, "text": "warranty terms apply for one year"}]
    notes_generator.generate_notes("warranty terms apply for one year", segments, source_type="pdf_document", title="Manual")

    prompt = captured["prompt"]
    assert "[p.3]: warranty terms apply for one year" in prompt
    assert "PDF document" in prompt
    assert "page numbers" in prompt
    assert 'Title: "Manual"' in prompt
    # A time-based reference hint should NOT leak into a page-based prompt.
    assert "timestamps (e.g" not in prompt


def test_generate_notes_uses_no_reference_hint_for_plain_text_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(notes_generator, "chat_completion", lambda prompt, model: captured.update(prompt=prompt) or "NOTES")

    segments = [{"text": "the team agreed to ship on Friday"}]
    notes_generator.generate_notes("the team agreed to ship on Friday", segments, source_type="text_document")

    prompt = captured["prompt"]
    assert "text document" in prompt
    assert "no timestamps or pages are available" in prompt


def test_answer_question_returns_fallback_when_no_chunks():
    result = rag_chat.answer_question("what?", [])
    assert "enough information" in result.lower()


def test_answer_question_uses_context_and_history(monkeypatch):
    captured = {}

    def fake_chat_completion(prompt, model):
        captured["prompt"] = prompt
        return "ANSWER"

    monkeypatch.setattr(rag_chat, "chat_completion", fake_chat_completion)

    chunks = [{"start": 0, "end": 5, "text": "the sky is blue"}]
    result = rag_chat.answer_question(
        "what color is the sky?", chunks, chat_history=[("hi", "hello")]
    )

    assert result == "ANSWER"
    assert "the sky is blue" in captured["prompt"]
    assert "PREVIOUS CONVERSATION" in captured["prompt"]


def test_answer_question_prompt_cites_sources_with_title_and_timestamp(monkeypatch):
    captured = {}

    def fake_chat_completion(prompt, model):
        captured["prompt"] = prompt
        return "ANSWER"

    monkeypatch.setattr(rag_chat, "chat_completion", fake_chat_completion)

    chunks = [
        {
            "start": 72,
            "end": 100,
            "text": "the launch date is March 3rd",
            "title": "Product Keynote",
            "video_id": "abc123",
            "source_type": "youtube_video",
        }
    ]
    rag_chat.answer_question("when does it launch?", chunks)

    prompt = captured["prompt"]
    assert "Source 1" in prompt
    assert "Product Keynote" in prompt
    assert "1:12-1:40" in prompt
    assert "GROUND" in prompt.upper()
