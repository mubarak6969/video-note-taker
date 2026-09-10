import logging
import os

import yt_dlp

import config

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when a video's audio can't be downloaded."""


def download_audio(youtube_url: str):
    """Downloads a YouTube video's audio as mp3.

    Returns (video_id, title, file_path). Files are named by video_id
    (not the raw title) to avoid collisions and illegal-filename/path-length
    issues with titles that contain special characters.
    """
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
    file_path = os.path.join(config.DOWNLOADS_DIR, f"{video_id}.mp3")

    if not video_id or not os.path.exists(file_path):
        raise DownloadError(
            "Download finished but the expected audio file was not found. "
            "The video may be private, age-restricted, or unavailable."
        )

    logger.info("Downloaded '%s' -> %s", title, file_path)
    return video_id, title, file_path
