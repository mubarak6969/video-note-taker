import logging
import os
import re

import yt_dlp

import config
from url_validation import is_youtube_url

logger = logging.getLogger(__name__)

# Real YouTube video ids are 11 chars of [A-Za-z0-9_-]; allow a little
# slack for format changes, but this is also our defense-in-depth check
# before the id ever reaches a filesystem path (see vector_store.py) -
# reject anything that isn't plainly an id, rather than trusting yt-dlp's
# parsed output unconditionally.
_SAFE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class DownloadError(Exception):
    """Raised when a video's audio can't be downloaded."""


def download_audio(youtube_url: str):
    """Downloads a YouTube video's audio as mp3.

    Returns (video_id, title, file_path). Files are named by video_id
    (not the raw title) to avoid collisions and illegal-filename/path-length
    issues with titles that contain special characters.
    """
    if not is_youtube_url(youtube_url):
        raise DownloadError("That doesn't look like a YouTube URL.")

    os.makedirs(config.DOWNLOADS_DIR, exist_ok=True)
    logger.info("Downloading audio from: %s", youtube_url)

    output_path = os.path.join(config.DOWNLOADS_DIR, "%(id)s.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    if os.path.exists("cookies.txt"):
        options["cookiefile"] = "cookies.txt"

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(str(e)) from e

    video_id = info.get("id")
    title = info.get("title") or video_id or "Unknown"

    if not video_id or not _SAFE_VIDEO_ID_RE.match(video_id):
        raise DownloadError(
            "This video's id looked unexpected and was rejected as a precaution."
        )

    file_path = os.path.join(config.DOWNLOADS_DIR, f"{video_id}.mp3")
    if not os.path.exists(file_path):
        raise DownloadError(
            "Download finished but the expected audio file was not found. "
            "The video may be private, age-restricted, or unavailable."
        )

    logger.info("Downloaded '%s' -> %s", title, file_path)
    return video_id, title, file_path
