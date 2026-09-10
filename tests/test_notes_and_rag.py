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
