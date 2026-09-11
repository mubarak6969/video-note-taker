import importlib
import os

import pytest

import downloader
from downloader import DownloadError


def test_rejects_non_youtube_url_before_ever_calling_yt_dlp(monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("yt_dlp should never be invoked for a non-YouTube URL")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", must_not_be_called)

    with pytest.raises(DownloadError, match="doesn't look like a YouTube URL"):
        downloader.download_audio("https://evil.example/?next=youtube.com/watch")


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL - writes a fake mp3 where the real
    downloader would have, and returns a canned info dict."""

    def __init__(self, info, expected_path=None):
        self._info = info
        self._expected_path = expected_path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=True):
        if self._expected_path:
            os.makedirs(os.path.dirname(self._expected_path), exist_ok=True)
            with open(self._expected_path, "wb") as f:
                f.write(b"fake mp3 data")
        return self._info


def test_rejects_malformed_video_id_from_yt_dlp(tmp_path, monkeypatch):
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path))
    import config
    importlib.reload(config)

    malicious_info = {"id": "../../etc/passwd", "title": "Evil"}
    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", lambda options: _FakeYDL(malicious_info))

    with pytest.raises(DownloadError, match="unexpected"):
        downloader.download_audio("https://www.youtube.com/watch?v=x")


def test_rejects_video_id_with_path_separator(tmp_path, monkeypatch):
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path))
    import config
    importlib.reload(config)

    info = {"id": "abc/def", "title": "Evil"}
    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", lambda options: _FakeYDL(info))

    with pytest.raises(DownloadError, match="unexpected"):
        downloader.download_audio("https://www.youtube.com/watch?v=x")


def test_accepts_well_formed_video_id_and_returns_its_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path))
    import config
    importlib.reload(config)

    expected_path = str(tmp_path / "dQw4w9WgXcQ.mp3")
    info = {"id": "dQw4w9WgXcQ", "title": "Never Gonna Give You Up"}
    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", lambda options: _FakeYDL(info, expected_path))

    video_id, title, file_path = downloader.download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert video_id == "dQw4w9WgXcQ"
    assert title == "Never Gonna Give You Up"
    assert file_path == expected_path
