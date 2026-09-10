import reranker


def test_rerank_empty_candidates_returns_empty_list():
    assert reranker.rerank("anything", []) == []


def test_rerank_none_method_preserves_input_order_and_truncates():
    candidates = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    result = reranker.rerank("q", candidates, top_k=2, method="none")
    assert result == candidates[:2]


def test_heuristic_rerank_prefers_higher_combined_score():
    candidates = [
        {"text": "irrelevant filler text", "vector_score": 0.9, "keyword_score": 0.0},
        {"text": "revenue grew to 3.7 million dollars", "vector_score": 0.1, "keyword_score": 1.0},
    ]
    # min_relevance=0 disables the relevance gate (covered separately below)
    # so this test stays focused on ranking order, not evidence strength.
    result = reranker.rerank(
        "what was the revenue figure of 3.7 million?", candidates, top_k=2, method="heuristic", min_relevance=0
    )

    assert result[0]["text"].startswith("revenue grew")
    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] >= result[1]["rerank_score"]


def test_heuristic_rerank_respects_top_k():
    candidates = [
        {"text": "one", "vector_score": 0.9, "keyword_score": 0.0},
        {"text": "two", "vector_score": 0.5, "keyword_score": 0.0},
        {"text": "three", "vector_score": 0.1, "keyword_score": 0.0},
    ]
    result = reranker.rerank("q", candidates, top_k=1, method="heuristic", min_relevance=0)
    assert len(result) == 1
    assert result[0]["text"] == "one"


def test_cross_encoder_failure_falls_back_to_heuristic(monkeypatch):
    def broken_get_cross_encoder():
        raise RuntimeError("no model available offline")

    monkeypatch.setattr(reranker, "_get_cross_encoder", broken_get_cross_encoder)

    candidates = [
        {"text": "a", "vector_score": 0.2, "keyword_score": 0.0},
        {"text": "b", "vector_score": 0.9, "keyword_score": 0.0},
    ]
    result = reranker.rerank("q", candidates, top_k=2, method="cross_encoder", min_relevance=0)

    # Falls back to the heuristic scorer instead of raising.
    assert result[0]["text"] == "b"
    assert all("rerank_score" in c for c in result)


# ------------------------------------------------------- relevance gate --


def test_relevance_gate_rejects_a_candidate_below_threshold():
    candidates = [
        {"text": "completely unrelated filler", "vector_score": 0.1, "keyword_score": 0.0, "vector_similarity": 0.02},
    ]
    result = reranker.rerank("what is the launch date?", candidates, method="heuristic", min_relevance=0.3)
    assert result == []


def test_relevance_gate_accepts_a_candidate_above_threshold():
    candidates = [
        {"text": "the launch date is march third", "vector_score": 0.9, "keyword_score": 0.9, "vector_similarity": 0.8},
    ]
    result = reranker.rerank("what is the launch date?", candidates, method="heuristic", min_relevance=0.3)
    assert len(result) == 1


def test_relevance_gate_drops_only_the_weak_candidates_from_a_mixed_set():
    candidates = [
        {"text": "the launch date is march third", "vector_score": 0.9, "keyword_score": 0.9, "vector_similarity": 0.8},
        {"text": "completely unrelated filler about cooking", "vector_score": 0.1, "keyword_score": 0.0, "vector_similarity": 0.02},
    ]
    result = reranker.rerank("what is the launch date?", candidates, top_k=5, method="heuristic", min_relevance=0.3)
    assert len(result) == 1
    assert result[0]["text"].startswith("the launch date")


def test_relevance_gate_an_exact_keyword_match_survives_despite_low_vector_similarity():
    # A rare code with near-zero embedding similarity to the query should
    # still pass on the strength of an exact match alone - the gate must
    # not punish the very case hybrid retrieval exists to rescue.
    candidates = [
        {"text": "the activation code is zx9981q for this device", "vector_score": 0.0, "keyword_score": 1.0, "vector_similarity": 0.01},
    ]
    result = reranker.rerank("what is the activation code zx9981q?", candidates, method="heuristic", min_relevance=0.3)
    assert len(result) == 1


def test_relevance_gate_is_disabled_by_min_relevance_zero():
    candidates = [
        {"text": "completely unrelated filler", "vector_score": 0.1, "keyword_score": 0.0, "vector_similarity": 0.0},
    ]
    result = reranker.rerank("what is the launch date?", candidates, method="heuristic", min_relevance=0)
    assert len(result) == 1


def test_relevance_gate_is_bypassed_entirely_for_none_method():
    # method="none" exists to pass candidates through untouched for
    # debugging/comparison - it must never silently drop anything, even
    # with a high default threshold configured.
    candidates = [{"text": "anything at all"}]
    result = reranker.rerank("unrelated query", candidates, method="none", min_relevance=0.99)
    assert result == candidates


def test_relevance_gate_uses_config_default_when_not_overridden(monkeypatch):
    import config

    monkeypatch.setattr(config, "RAG_RELEVANCE_THRESHOLD", 0.9)
    candidates = [
        # No content-word overlap with the query, so relevance_score is
        # driven by vector_similarity alone: 0.5 won't clear a 0.9 bar.
        {"text": "unrelated passage about something else entirely", "vector_score": 0.9, "keyword_score": 0.0, "vector_similarity": 0.5},
    ]
    result = reranker.rerank("what is the launch date?", candidates, method="heuristic")
    assert result == []


def test_exact_overlap_ignores_stopwords_shared_with_an_unrelated_chunk():
    # Sharing only function words ("is", "the") with a chunk must not
    # register as evidence of relevance - this is what keeps a genuinely
    # off-topic question from slipping past the gate by coincidence.
    candidates = [
        {"text": "the reimbursement policy code is HB-4471", "vector_score": 0.0, "keyword_score": 0.0, "vector_similarity": 0.05},
    ]
    result = reranker.rerank("what is the capital of France?", candidates, method="heuristic", min_relevance=0.3)
    assert result == []


def test_sigmoid_normalizes_cross_encoder_scores_into_zero_one_range():
    assert reranker._sigmoid(0.0) == 0.5
    assert 0.0 < reranker._sigmoid(-10.0) < 0.01
    assert 0.99 < reranker._sigmoid(10.0) < 1.0
