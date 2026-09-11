"""A minimal, single-process, per-session rate limiter - a safety net on
how much one browser session can cost in Whisper/Groq/download spend, not
a substitute for real authentication or a distributed rate limiter. Pure
functions operating on a plain list of timestamps, so the logic is fully
unit-testable without Streamlit or a clock mock framework.
"""
import time


def prune_old(timestamps: list, window_seconds: int) -> list:
    """Returns only the timestamps still within the trailing window."""
    cutoff = time.time() - window_seconds
    return [t for t in timestamps if t > cutoff]


def is_allowed(timestamps: list, limit: int, window_seconds: int = 3600) -> bool:
    """True if fewer than `limit` actions have happened in the trailing
    `window_seconds`. `limit` <= 0 means unlimited (always allowed)."""
    if limit <= 0:
        return True
    return len(prune_old(timestamps, window_seconds)) < limit


def record(timestamps: list, window_seconds: int = 3600) -> list:
    """Returns an updated timestamp list with 'now' appended and anything
    outside the window dropped, so the list never grows unbounded."""
    pruned = prune_old(timestamps, window_seconds)
    pruned.append(time.time())
    return pruned
