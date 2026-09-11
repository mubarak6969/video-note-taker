import importlib
import os

import pytest


def _fresh_audio_source(
    tmp_path, monkeypatch, fake_transcribe=None, max_audio_mb=None, max_audio_minutes=None, fake_duration=None
):
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path / "downloads"))
    monkeypatch.setenv("CHUNK_DURATION_SECONDS", "30")
    monkeypatch.setenv("CHUNK_OVERLAP_SECONDS", "5")
    if max_audio_mb is not None:
        monkeypatch.setenv("UPLOAD_MAX_AUDIO_MB", str(max_audio_mb))
    if max_audio_minutes is not None:
        monkeypatch.setenv("UPLOAD_MAX_AUDIO_MINUTES", str(max_audio_minutes))
    import config

    importlib.reload(config)
    import ingestion.audio_source as audio_source

    importlib.reload(audio_source)

    if fake_transcribe is not None:
        monkeypatch.setattr(audio_source, "transcribe_audio", fake_transcribe)
    # Default: pretend ffprobe is unavailable (None) unless a test wants a
    # specific duration - matches "skip the check rather than block" in
    # environments without ffmpeg, and keeps every other test unaffected.
    monkeypatch.setattr(audio_source, "get_duration_seconds", lambda path: fake_duration)
    return audio_source


def _default_fake_transcribe(calls):
    def fake(file_path, language=None, model_size=None):
        calls.append(file_path)
        segments = [
            {"text": "hello there", "start": 0.0, "end": 2.0},
            {"text": "general kenobi", "start": 2.0, "end": 4.0},
        ]
        return "hello there general kenobi", segments, file_path + "_transcript.txt"

    return fake


def test_ingest_audio_upload_happy_path(tmp_path, monkeypatch):
    calls = []
    audio_source = _fresh_audio_source(tmp_path, monkeypatch, _default_fake_transcribe(calls))

    source = audio_source.ingest_audio_upload(b"fake mp3 bytes", "meeting.mp3")

    assert source.source_type == "audio_upload"
    assert source.title == "meeting"
    assert source.origin == "meeting.mp3"
    assert len(source.chunks) >= 1
    assert all(c["start"] is not None for c in source.chunks)
    assert calls, "transcribe_audio should have been called with the stored file path"
    assert os.path.exists(calls[0])


def test_ingest_audio_upload_rejects_unsupported_extension(tmp_path, monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("transcribe_audio should not be called for a rejected extension")

    audio_source = _fresh_audio_source(tmp_path, monkeypatch, must_not_be_called)
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="Unsupported"):
        audio_source.ingest_audio_upload(b"bytes", "notes.docx")


def test_ingest_audio_upload_rejects_oversized_file(tmp_path, monkeypatch):
    audio_source = _fresh_audio_source(tmp_path, monkeypatch, _default_fake_transcribe([]), max_audio_mb=0)
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="too large"):
        audio_source.ingest_audio_upload(b"some audio bytes", "clip.mp3")


def test_ingest_audio_upload_raises_when_no_speech_detected(tmp_path, monkeypatch):
    def fake_no_speech(file_path, language=None, model_size=None):
        return "", [], file_path + "_transcript.txt"

    audio_source = _fresh_audio_source(tmp_path, monkeypatch, fake_no_speech)
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="No speech"):
        audio_source.ingest_audio_upload(b"silence bytes", "silence.wav")


def test_ingest_audio_upload_reuses_stored_file_for_identical_bytes(tmp_path, monkeypatch):
    calls = []
    audio_source = _fresh_audio_source(tmp_path, monkeypatch, _default_fake_transcribe(calls))

    data = b"identical audio bytes"
    audio_source.ingest_audio_upload(data, "first.mp3")
    audio_source.ingest_audio_upload(data, "second.mp3")

    # Same content -> same source_id -> same stored path both times.
    assert calls[0] == calls[1]


def test_ingest_audio_upload_rejects_file_over_duration_cap(tmp_path, monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("transcribe_audio should not run once the duration cap rejects the file")

    audio_source = _fresh_audio_source(
        tmp_path, monkeypatch, must_not_be_called, max_audio_minutes=10, fake_duration=15 * 60
    )
    from ingestion.errors import IngestionError

    with pytest.raises(IngestionError, match="minutes long"):
        audio_source.ingest_audio_upload(b"long audio bytes", "long.mp3")


def test_ingest_audio_upload_allows_file_under_duration_cap(tmp_path, monkeypatch):
    calls = []
    audio_source = _fresh_audio_source(
        tmp_path, monkeypatch, _default_fake_transcribe(calls), max_audio_minutes=10, fake_duration=5 * 60
    )
    source = audio_source.ingest_audio_upload(b"short audio bytes", "short.mp3")
    assert source.source_type == "audio_upload"
    assert calls


def test_ingest_audio_upload_proceeds_when_duration_is_unknown(tmp_path, monkeypatch):
    """ffprobe being unavailable/failing must not block ingestion - the
    duration cap is a safety net, not a hard requirement."""
    calls = []
    audio_source = _fresh_audio_source(
        tmp_path, monkeypatch, _default_fake_transcribe(calls), max_audio_minutes=10, fake_duration=None
    )
    source = audio_source.ingest_audio_upload(b"unknown duration bytes", "mystery.mp3")
    assert source.source_type == "audio_upload"
    assert calls
