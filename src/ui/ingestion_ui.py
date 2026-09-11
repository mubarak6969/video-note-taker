"""YouTube URL and Upload File ingestion tabs. Validation/errors are
always shown inline (st.warning/st.error) - never swallowed - per the
existing app behavior this UI layer preserves."""
import logging

import streamlit as st

import config
import vector_store
from downloader import DownloadError
from ingestion import (
    IngestionError,
    SUPPORTED_UPLOAD_EXTENSIONS,
    hash_bytes,
    ingest_uploaded_file,
    ingest_youtube_url,
)
from services import finish_ingestion
from ui import access_control
from ui.constants import LANGUAGES, WHISPER_SIZES
from ui.state import load_source
from url_validation import is_youtube_url

logger = logging.getLogger(__name__)

_DEFAULT_WHISPER_INDEX = (
    WHISPER_SIZES.index(config.WHISPER_MODEL_SIZE) if config.WHISPER_MODEL_SIZE in WHISPER_SIZES else 0
)


def render_ingestion_tabs():
    tab_youtube, tab_upload = st.tabs(["🔗 YouTube URL", "📁 Upload File"])
    with tab_youtube:
        _render_youtube_tab()
    with tab_upload:
        _render_upload_tab()


def _render_youtube_tab():
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox("Video language", list(LANGUAGES.keys()), key="yt_language")
    with col2:
        whisper_size = st.selectbox(
            "Transcription quality",
            WHISPER_SIZES,
            index=_DEFAULT_WHISPER_INDEX,
            key="yt_whisper_size",
            help="Larger models are slower but more accurate.",
        )

    if st.button("Process video", type="primary", key="process_youtube"):
        if not youtube_url:
            st.warning("Please paste a YouTube URL first.")
        elif not is_youtube_url(youtube_url):
            st.warning("That doesn't look like a YouTube URL.")
        elif not access_control.check_ingestion_allowed():
            pass  # the limit message was already shown
        else:
            try:
                with st.status("Processing video...", expanded=True) as status:
                    status.write("⬇️ Downloading and transcribing audio (this can take a minute)...")
                    source = ingest_youtube_url(youtube_url, language=LANGUAGES[language], model_size=whisper_size)
                    is_new = finish_ingestion(source, on_progress=status.write)
                    load_source(source.source_id, source.title)
                    status.update(
                        label=f"Done! Processed: {source.title}" if is_new else f"Loaded from library: {source.title}",
                        state="complete",
                    )
                access_control.record_ingestion()
                st.rerun()
            except DownloadError as e:
                st.error(f"❌ Couldn't download this video: {e}")
            except IngestionError as e:
                st.error(f"❌ {e}")
            except Exception:
                logger.exception("Video processing failed")
                st.error("❌ Something went wrong while processing this video. Please try again.")


def _render_upload_tab():
    uploaded_file = st.file_uploader(
        "Upload a PDF, text file, or audio/video file",
        type=SUPPORTED_UPLOAD_EXTENSIONS,
        help="PDF and text files are read directly; audio/video files are transcribed with Whisper.",
    )

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Language", list(LANGUAGES.keys()), key="up_language", help="Used for audio/video files only."
        )
    with col2:
        whisper_size = st.selectbox(
            "Transcription quality",
            WHISPER_SIZES,
            index=_DEFAULT_WHISPER_INDEX,
            key="up_whisper_size",
            help="Used for audio/video files only.",
        )

    if st.button("Process upload", type="primary", key="process_upload"):
        if uploaded_file is None:
            st.warning("Please choose a file first.")
        else:
            _process_upload(uploaded_file, language, whisper_size)


def _process_upload(uploaded_file, language, whisper_size):
    file_bytes = uploaded_file.getvalue()
    try:
        source_id = hash_bytes(file_bytes)

        if vector_store.source_exists(source_id):
            existing = vector_store.get_source(source_id)
            title = existing["title"] if existing else uploaded_file.name
            with st.status("Processing file...", expanded=True) as status:
                status.write("♻️ Already in your library - loading saved notes.")
                load_source(source_id, title)
                status.update(label=f"Loaded from library: {title}", state="complete")
            st.rerun()
            return

        if not access_control.check_ingestion_allowed():
            return

        with st.status("Processing file...", expanded=True) as status:
            status.write("📖 Reading and processing file...")
            source = ingest_uploaded_file(
                uploaded_file.name, file_bytes, language=LANGUAGES[language], model_size=whisper_size
            )
            finish_ingestion(source, on_progress=status.write)
            load_source(source.source_id, source.title)
            status.update(label=f"Done! Processed: {source.title}", state="complete")
        access_control.record_ingestion()
        st.rerun()
    except IngestionError as e:
        st.error(f"❌ {e}")
    except Exception:
        logger.exception("Upload processing failed")
        st.error("❌ Something went wrong while processing this file. Please try again.")
