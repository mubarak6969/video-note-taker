import logging
import os

import whisper

logger = logging.getLogger(__name__)

# Whisper models are large (~100MB-1.5GB); load each size once per process
# and reuse it, instead of reloading from disk on every video.
_model_cache = {}


def _get_model(model_size: str):
    if model_size not in _model_cache:
        logger.info("Loading Whisper model: %s", model_size)
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]


def transcribe_audio(file_path, language=None, model_size="base"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = _get_model(model_size)
    logger.info("Transcribing: %s", file_path)

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(file_path, **options)

    transcript = result["text"]
    segments = result["segments"]

    transcript_path = os.path.splitext(file_path)[0] + "_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    logger.info("Transcription complete: %s", transcript_path)
    return transcript, segments, transcript_path
