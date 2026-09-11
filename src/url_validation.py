"""URL validation shared between the UI (fast inline feedback) and the
ingestion layer (a hard backstop applied in downloader.py, so this can't
be bypassed by any caller that skips the UI)."""
from urllib.parse import urlparse

_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def is_youtube_url(url: str) -> bool:
    """True only if `url` actually points at a YouTube host.

    Deliberately parses the URL and checks its hostname rather than doing
    a substring match - a naive check for "youtube.com" anywhere in the
    string would accept something like "https://evil.example/?next=
    youtube.com/watch", which could be used to trick yt-dlp into fetching
    an attacker-controlled URL under the guise of a validated one.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return parsed.hostname in _ALLOWED_HOSTS if parsed.hostname else False
