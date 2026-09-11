"""Small, pure display-formatting helpers (no Streamlit imports, no
business logic - just turning stored values into readable strings)."""
import re
from datetime import datetime

_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\x00-\x1f]')


def format_processed_date(iso_timestamp: str) -> str:
    """Turns a stored ISO timestamp into a short, human date like
    'Sep 05, 2026'. Returns '' for anything unparseable rather than
    raising - this only ever feeds a caption, never a calculation."""
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return ""
    return dt.strftime("%b %d, %Y")


def safe_download_filename(title: str, default: str, extension: str) -> str:
    """Sanitizes a source title for use as a download filename. Titles
    come from external sources (a YouTube title, an uploaded filename)
    and can contain path separators or control characters - strip those
    rather than passing them through to a filesystem save-dialog / the
    Content-Disposition header Streamlit builds from `file_name`."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", (title or "").strip())
    cleaned = cleaned.strip(". ")[:150]
    return f"{cleaned or default}{extension}"
