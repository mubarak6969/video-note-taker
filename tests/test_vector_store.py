import importlib


def _fresh_vector_store(tmp_path, monkeypatch):
    """Reimports config + vector_store pointed at an isolated temp DATA_DIR,
    so tests never touch the app's real on-disk library."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import config
    import vector_store

    importlib.reload(config)
    importlib.reload(vector_store)
    return vector_store


def test_save_list_search_and_delete_roundtrip(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    chunks = [
        {"text": "cats are great", "start": 0, "end": 10, "embedding": [1.0, 0.0, 0.0]},
        {"text": "dogs are great too", "start": 10, "end": 20, "embedding": [0.0, 1.0, 0.0]},
    ]
    vector_store.save_video("vid1", "My Video", "http://example.com", "# Notes", chunks)

    assert vector_store.video_exists("vid1")
    videos = vector_store.list_videos()
    assert videos[0]["video_id"] == "vid1"
    assert videos[0]["title"] == "My Video"
    assert vector_store.load_notes("vid1") == "# Notes"

    results = vector_store.search_chunks([1.0, 0.0, 0.0], "vid1", top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "cats are great"

    vector_store.delete_video("vid1")
    assert not vector_store.video_exists("vid1")
    assert vector_store.load_notes("vid1") is None


def test_save_video_rejects_empty_chunks(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    try:
        vector_store.save_video("vid2", "Empty", "http://example.com", "# Notes", [])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transcript_is_persisted_and_removed_with_the_source(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    chunks = [{"text": "hello world", "start": 0, "end": 5, "embedding": [1.0, 0.0, 0.0]}]
    vector_store.save_source(
        "vid3", "My Video", "http://example.com", "# Notes", chunks,
        full_text="This is the full raw transcript text.",
    )

    assert vector_store.load_transcript("vid3") == "This is the full raw transcript text."

    vector_store.delete_source("vid3")
    assert vector_store.load_transcript("vid3") is None


def test_transcript_is_optional_and_absent_by_default(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    chunks = [{"text": "hello world", "start": 0, "end": 5, "embedding": [1.0, 0.0, 0.0]}]
    vector_store.save_video("vid4", "My Video", "http://example.com", "# Notes", chunks)

    # save_video's backward-compatible signature doesn't require full_text -
    # older callers (and existing saved sources) simply have no transcript.
    assert vector_store.load_transcript("vid4") is None


# ------------------------------------------------ reliability & security --


def test_corrupted_library_index_is_treated_as_empty_not_a_crash(tmp_path, monkeypatch):
    """A corrupted data/library.json must not brick the whole app - every
    page load calls list_sources() unconditionally."""
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    with open(vector_store.LIBRARY_PATH, "w", encoding="utf-8") as f:
        f.write("{not valid json::")

    assert vector_store.list_sources() == []
    assert not vector_store.source_exists("anything")


def test_source_id_with_path_traversal_is_rejected(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)

    chunks = [{"text": "hello", "start": 0, "end": 5, "embedding": [1.0, 0.0, 0.0]}]
    malicious_id = "../../etc/passwd"

    for fn, args in [
        (vector_store.save_source, (malicious_id, "Title", "http://x", "# Notes", chunks)),
        (vector_store.delete_source, (malicious_id,)),
        (vector_store.load_notes, (malicious_id,)),
        (vector_store.load_transcript, (malicious_id,)),
    ]:
        try:
            fn(*args)
            assert False, f"{fn.__name__} should have rejected a path-traversal id"
        except ValueError:
            pass


def test_source_id_with_slash_is_rejected(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)
    try:
        vector_store.load_notes("abc/def")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_normal_ids_are_unaffected_by_sanitization(tmp_path, monkeypatch):
    vector_store = _fresh_vector_store(tmp_path, monkeypatch)
    # YouTube-style ids and hex content hashes must keep working normally.
    for ok_id in ("dQw4w9WgXcQ", "a1b2c3d4e5f6a7b8", "my_source-1"):
        assert vector_store.load_notes(ok_id) is None  # doesn't raise
