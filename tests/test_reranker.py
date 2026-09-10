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
    result = reranker.rerank(
        "what was the revenue figure of 3.7 million?", candidates, top_k=2, method="heuristic"
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
    result = reranker.rerank("q", candidates, top_k=1, method="heuristic")
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
    result = reranker.rerank("q", candidates, top_k=2, method="cross_encoder")

    # Falls back to the heuristic scorer instead of raising.
    assert result[0]["text"] == "b"
    assert all("rerank_score" in c for c in result)
