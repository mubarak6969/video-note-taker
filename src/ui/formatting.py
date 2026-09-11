"""Small, pure display-formatting helpers (no Streamlit imports, no
business logic - just turning stored values into readable strings)."""
from datetime import datetime


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
