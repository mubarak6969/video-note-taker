import subprocess

import media_probe


class _FakeResult:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_returns_duration_on_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "125.436000\n"))
    assert media_probe.get_duration_seconds("some/file.mp3") == 125.436


def test_returns_none_when_ffprobe_not_installed(monkeypatch):
    def raise_not_found(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", raise_not_found)
    assert media_probe.get_duration_seconds("some/file.mp3") is None


def test_returns_none_on_timeout(monkeypatch):
    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert media_probe.get_duration_seconds("some/file.mp3") is None


def test_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(1, ""))
    assert media_probe.get_duration_seconds("some/file.mp3") is None


def test_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "not-a-number"))
    assert media_probe.get_duration_seconds("some/file.mp3") is None


def test_invokes_ffprobe_with_list_args_not_a_shell_string(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeResult(0, "10.0")

    monkeypatch.setattr(subprocess, "run", fake_run)
    media_probe.get_duration_seconds("my file with spaces.mp3")

    assert isinstance(captured["args"], list)
    assert captured["args"][0] == "ffprobe"
    assert "my file with spaces.mp3" in captured["args"]
    assert not captured["kwargs"].get("shell")
