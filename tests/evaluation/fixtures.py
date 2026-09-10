import importlib

from .golden_dataset import CORPUS


def build_eval_library(tmp_path, monkeypatch):
    """Seeds an isolated, on-disk ChromaDB-backed library from CORPUS using
    the real embedding model (deterministic at inference time - no
    dropout/sampling), pointed at a temp DATA_DIR so it never touches the
    app's real library. Returns (retrieval, vector_store, rag_chat)
    reloaded against that isolated store.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import config
    import rag_chat
    import retrieval
    import vector_store
    from embedder import embed_chunks

    importlib.reload(config)
    importlib.reload(vector_store)

    for source in CORPUS:
        chunks = embed_chunks([dict(c) for c in source["chunks"]])
        vector_store.save_source(
            source["source_id"],
            source["title"],
            source["source_id"],
            f"# Notes for {source['title']}",
            chunks,
            source_type=source["source_type"],
        )

    return retrieval, vector_store, rag_chat
