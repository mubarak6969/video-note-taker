import importlib

import pytest

import ingestion.youtube_source as youtube_source
from ingestion.errors import IngestionError


def _fresh_youtube_source(monkeypatch, max_audio_minutes=None, fake_duration=None, segments=None):
    if max_audio_minutes is not None:
        monkeypatch.setenv("UPLOAD_MAX_AUDIO_MINUTES", str(max_audio_minutes))
    import config

    importlib.reload(config)
    importlib.reload(youtube_source)

    monkeypatch.setattr(youtube_source, "download_audio", lambda url: ("vid123", "Title", "vid123.mp3"))
    monkeypatch.setattr(youtube_source, "get_duration_seconds", lambda path: fake_duration)

    default_segments = segments if segments is not None else [{"text": "hello", "start": 0.0, "end": 2.0}]

    def fake_transcribe(file_path, language=None, model_size=None):
        return "hello", default_segments, file_path + "_transcript.txt"

    monkeypatch.setattr(youtube_source, "transcribe_audio", fake_transcribe)
    return youtube_source


def test_rejects_video_over_duration_cap(monkeypatch):
    mod = _fresh_youtube_source(monkeypatch, max_audio_minutes=10, fake_duration=20 * 60)

    def must_not_be_called(*a, **k):
        raise AssertionError("transcribe_audio should not run once the duration cap rejects the video")

    monkeypatch.setattr(mod, "transcribe_audio", must_not_be_called)

    with pytest.raises(IngestionError, match="minutes long"):
        mod.ingest_youtube_url("https://www.youtube.com/watch?v=vid123")


def test_allows_video_under_duration_cap(monkeypatch):
    mod = _fresh_youtube_source(monkeypatch, max_audio_minutes=10, fake_duration=5 * 60)
    source = mod.ingest_youtube_url("https://www.youtube.com/watch?v=vid123")
    assert source.source_type == "youtube_video"
    assert source.source_id == "vid123"


def test_proceeds_when_duration_is_unknown(monkeypatch):
    mod = _fresh_youtube_source(monkeypatch, max_audio_minutes=10, fake_duration=None)
    source = mod.ingest_youtube_url("https://www.youtube.com/watch?v=vid123")
    assert source.source_type == "youtube_video"
