"""Thin, defensive wrapper around ffprobe for checking audio/video
duration before running expensive Whisper transcription. ffmpeg (which
bundles ffprobe) is already a hard dependency of the transcription
pipeline itself (see packages.txt), so this adds no new external tool -
just one extra, safe subprocess call (list-args, never a shell string).
"""
import logging
import subprocess

logger = logging.getLogger(__name__)


def get_duration_seconds(file_path: str):
    """Returns the media file's duration in seconds, or None if ffprobe
    isn't available or the file couldn't be probed. Callers should treat
    None as "unknown" and skip any duration-based check rather than
    blocking ingestion on a missing/broken ffprobe - the duration cap is
    a cost/time safety net, not something ingestion should hard-fail on
    just because the safety net itself couldn't run.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("ffprobe unavailable or failed (%s); skipping duration check.", e)
        return None

    if result.returncode != 0:
        logger.warning("ffprobe exited %d for %s; skipping duration check.", result.returncode, file_path)
        return None

    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return None
