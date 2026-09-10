"""Deterministic RAG evaluation: exercises real retrieval (embedding +
ChromaDB + hybrid search + reranking) against a small golden-question
corpus, without ever calling the real Groq API. Two tests mock the LLM
client to inspect prompt construction (follow-up context, no-LLM-call
guarantee on empty context) - everything else asserts on retrieval
output directly.
"""
from embedder import embed_query

from .fixtures import build_eval_library
from .golden_dataset import GOLDEN_QUESTIONS


def _questions(category):
    return [q for q in GOLDEN_QUESTIONS if q["category"] == category]


def test_exact_keyword_retrieval_finds_the_right_evidence(tmp_path, monkeypatch):
    retrieval, _, _ = build_eval_library(tmp_path, monkeypatch)

    for q in _questions("exact_keyword"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=3)
        assert any(q["expected_text_substring"] in c["text"] for c in results), (
            f"{q['id']}: expected '{q['expected_text_substring']}' among top results, "
            f"got {[c['text'] for c in results]}"
        )


def test_semantic_retrieval_finds_paraphrased_content(tmp_path, monkeypatch):
    retrieval, _, _ = build_eval_library(tmp_path, monkeypatch)

    for q in _questions("semantic"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=3)
        assert any(q["expected_text_substring"] in c["text"] for c in results), (
            f"{q['id']}: semantic search should surface content mentioning "
            f"'{q['expected_text_substring']}' despite no exact keyword overlap in the question, "
            f"got {[c['text'] for c in results]}"
        )


def test_cross_source_retrieval_spans_multiple_sources(tmp_path, monkeypatch):
    retrieval, _, _ = build_eval_library(tmp_path, monkeypatch)

    for q in _questions("cross_source"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=6)
        found_sources = {c["video_id"] for c in results}
        assert q["expected_source_ids"] <= found_sources, (
            f"{q['id']}: expected chunks from {q['expected_source_ids']}, only found {found_sources}"
        )


def test_source_attribution_matches_expected_title_and_type(tmp_path, monkeypatch):
    retrieval, _, rag_chat = build_eval_library(tmp_path, monkeypatch)

    for q in _questions("source_attribution"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=3)
        assert results, f"{q['id']}: expected at least one retrieved chunk"

        top = results[0]
        assert top["source_type"] == q["expected_source_type"], q["id"]
        assert top["title"] == q["expected_title"], q["id"]

        info = rag_chat.describe_source(top)
        assert info["title"] == q["expected_title"], q["id"]
        assert info["source_type"] == q["expected_source_type"], q["id"]


def test_insufficient_context_never_calls_the_llm_and_returns_fallback(tmp_path, monkeypatch):
    retrieval, _, rag_chat = build_eval_library(tmp_path, monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM should not be called when retrieval finds no context")

    monkeypatch.setattr(rag_chat, "chat_completion", fail_if_called)

    for q in _questions("insufficient_context"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=3)
        assert results == [], f"{q['id']}: expected no chunks for an unprocessed source scope"

        answer = rag_chat.answer_question(q["question"], results)
        assert "enough information" in answer.lower()


def test_followup_question_prompt_carries_prior_conversation(tmp_path, monkeypatch):
    retrieval, _, rag_chat = build_eval_library(tmp_path, monkeypatch)

    captured = {}

    def fake_chat_completion(prompt, model):
        captured["prompt"] = prompt
        return "The byproduct is oxygen (Source 1, 0:45-1:10)."

    monkeypatch.setattr(rag_chat, "chat_completion", fake_chat_completion)

    for q in _questions("follow_up"):
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=3)
        assert results, f"{q['id']}: expected retrieval to find something even for a pronoun-only follow-up"

        rag_chat.answer_question(q["question"], results, chat_history=q["chat_history"])

        prior_question, prior_answer = q["chat_history"][0]
        assert "PREVIOUS CONVERSATION" in captured["prompt"]
        assert prior_question in captured["prompt"]
        assert prior_answer in captured["prompt"]


def test_overall_retrieval_hit_rate_meets_quality_gate(tmp_path, monkeypatch):
    """A simple, deterministic quality gate across every retrieval-graded
    golden question: this is the kind of check a RAG regression suite
    runs on every change to chunking/embedding/retrieval settings, to
    catch a quality regression before it ships."""
    retrieval, _, _ = build_eval_library(tmp_path, monkeypatch)

    graded_categories = {"exact_keyword", "semantic", "cross_source", "source_attribution"}
    graded = [q for q in GOLDEN_QUESTIONS if q["category"] in graded_categories]

    hits = 0
    for q in graded:
        embedding = embed_query(q["question"])
        results = retrieval.retrieve(q["question"], embedding, video_id=q["scope_source_id"], top_k=6)

        if q["category"] == "cross_source":
            ok = q["expected_source_ids"] <= {c["video_id"] for c in results}
        elif q["category"] == "source_attribution":
            ok = bool(results) and results[0]["title"] == q["expected_title"]
        else:
            ok = any(q["expected_text_substring"] in c["text"] for c in results)

        hits += int(ok)

    hit_rate = hits / len(graded)
    assert hit_rate >= 0.9, f"retrieval hit-rate {hit_rate:.0%} ({hits}/{len(graded)}) fell below the 90% quality gate"
