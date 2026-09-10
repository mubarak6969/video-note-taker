import logging
import os
import re

import streamlit as st

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _has_secret(name: str) -> bool:
    try:
        return name in st.secrets
    except Exception:
        return False


# Write cookies from secrets to a local file (for cloud deployment)
if _has_secret("YOUTUBE_COOKIES"):
    with open("cookies.txt", "w") as f:
        f.write(st.secrets["YOUTUBE_COOKIES"])

import config
import vector_store
from downloader import DownloadError
from embedder import embed_chunks, embed_query
from ingestion import IngestionError, SUPPORTED_UPLOAD_EXTENSIONS, hash_bytes, ingest_uploaded_file, ingest_youtube_url
from notes_generator import generate_notes
from rag_chat import answer_question
from retrieval import retrieve

st.set_page_config(page_title="Deep-Dive Video Note Taker", page_icon="🎯", layout="wide")

YOUTUBE_URL_RE = re.compile(r"(youtube\.com/|youtu\.be/)", re.IGNORECASE)

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
}

WHISPER_SIZES = ["base", "small", "medium"]
DEFAULT_WHISPER_INDEX = (
    WHISPER_SIZES.index(config.WHISPER_MODEL_SIZE) if config.WHISPER_MODEL_SIZE in WHISPER_SIZES else 0
)

SOURCE_ICONS = {
    "youtube_video": "🎬",
    "pdf_document": "📄",
    "text_document": "📝",
    "audio_upload": "🎙️",
}


def _ensure_groq_key_configured():
    if os.getenv("GROQ_API_KEY") or _has_secret("GROQ_API_KEY"):
        return
    st.error(
        "⚠️ GROQ_API_KEY is not configured. Add it to a `.env` file locally, "
        "or to Streamlit secrets when deployed, then reload the app."
    )
    st.stop()


_ensure_groq_key_configured()

DEFAULTS = {
    "current_source_id": None,
    "current_title": None,
    "current_notes": None,
    "chat_history": [],
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _load_source_into_state(source_id: str, title: str):
    st.session_state.current_source_id = source_id
    st.session_state.current_title = title
    st.session_state.current_notes = vector_store.load_notes(source_id)
    st.session_state.chat_history = []


def _finish_ingestion(source, status):
    """Shared tail of the ingestion pipeline once a source has been turned
    into an IngestedSource, regardless of whether it came from a YouTube
    URL or an uploaded file: dedupe, chunk+embed, generate notes, save."""
    if vector_store.source_exists(source.source_id):
        status.write("♻️ Already in your library - loading saved notes.")
        _load_source_into_state(source.source_id, source.title)
        status.update(label=f"Loaded from library: {source.title}", state="complete")
        return

    status.write("✂️ Chunking and embedding...")
    chunks = embed_chunks(source.chunks)

    status.write("📝 Generating notes...")
    notes = generate_notes(source.full_text, source.raw_segments)

    status.write("💾 Saving to your library...")
    vector_store.save_source(
        source.source_id, source.title, source.origin, notes, chunks, source_type=source.source_type
    )
    _load_source_into_state(source.source_id, source.title)
    status.update(label=f"Done! Processed: {source.title}", state="complete")


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("📚 Library")
    sources = vector_store.list_sources()
    if not sources:
        st.caption("Nothing processed yet.")
    for src in sources:
        src_id = src.get("source_id") or src["video_id"]
        icon = SOURCE_ICONS.get(src.get("source_type"), "📄")
        row = st.columns([5, 1])
        label = f"{icon} " + src["title"][:40] + ("…" if len(src["title"]) > 40 else "")
        if row[0].button(label, key=f"open_{src_id}", use_container_width=True):
            _load_source_into_state(src_id, src["title"])
            st.rerun()
        if row[1].button("🗑️", key=f"del_{src_id}"):
            vector_store.delete_source(src_id)
            if st.session_state.current_source_id == src_id:
                st.session_state.current_source_id = None
                st.session_state.current_title = None
                st.session_state.current_notes = None
                st.session_state.chat_history = []
            st.rerun()

# ------------------------------------------------------------------ Title --
st.title("🎯 Deep-Dive Video Note Taker")
st.write("Add a YouTube video or upload a file to generate notes and ask questions about it.")

tab_youtube, tab_upload = st.tabs(["🔗 YouTube URL", "📁 Upload File"])

with tab_youtube:
    youtube_url = st.text_input("YouTube URL")

    col1, col2 = st.columns(2)
    with col1:
        yt_language = st.selectbox("Video Language", list(LANGUAGES.keys()), key="yt_language")
    with col2:
        yt_whisper_size = st.selectbox(
            "Transcription quality",
            WHISPER_SIZES,
            index=DEFAULT_WHISPER_INDEX,
            key="yt_whisper_size",
            help="Larger models are slower but more accurate.",
        )

    if st.button("Process Video", type="primary", key="process_youtube"):
        if not youtube_url:
            st.warning("Please paste a YouTube URL first.")
        elif not YOUTUBE_URL_RE.search(youtube_url):
            st.warning("That doesn't look like a YouTube URL.")
        else:
            try:
                with st.status("Processing video...", expanded=True) as status:
                    status.write("⬇️ Downloading and transcribing audio (this can take a minute)...")
                    source = ingest_youtube_url(
                        youtube_url, language=LANGUAGES[yt_language], model_size=yt_whisper_size
                    )
                    _finish_ingestion(source, status)
                st.rerun()
            except DownloadError as e:
                st.error(f"❌ Couldn't download this video: {e}")
            except IngestionError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                logger.exception("Video processing failed")
                st.error(f"❌ Something went wrong while processing this video: {e}")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a PDF, text file, or audio/video file",
        type=SUPPORTED_UPLOAD_EXTENSIONS,
    )

    col1, col2 = st.columns(2)
    with col1:
        up_language = st.selectbox(
            "Language",
            list(LANGUAGES.keys()),
            key="up_language",
            help="Used for audio/video files only.",
        )
    with col2:
        up_whisper_size = st.selectbox(
            "Transcription quality",
            WHISPER_SIZES,
            index=DEFAULT_WHISPER_INDEX,
            key="up_whisper_size",
            help="Used for audio/video files only.",
        )

    if st.button("Process Upload", type="primary", key="process_upload"):
        if uploaded_file is None:
            st.warning("Please choose a file first.")
        else:
            file_bytes = uploaded_file.getvalue()
            try:
                with st.status("Processing file...", expanded=True) as status:
                    status.write("🔍 Checking your library...")
                    source_id = hash_bytes(file_bytes)

                    if vector_store.source_exists(source_id):
                        existing = vector_store.get_source(source_id)
                        title = existing["title"] if existing else uploaded_file.name
                        status.write("♻️ Already in your library - loading saved notes.")
                        _load_source_into_state(source_id, title)
                        status.update(label=f"Loaded from library: {title}", state="complete")
                    else:
                        status.write("📖 Reading and processing file...")
                        source = ingest_uploaded_file(
                            uploaded_file.name,
                            file_bytes,
                            language=LANGUAGES[up_language],
                            model_size=up_whisper_size,
                        )
                        _finish_ingestion(source, status)
                st.rerun()
            except IngestionError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                logger.exception("Upload processing failed")
                st.error(f"❌ Something went wrong while processing this file: {e}")

# ------------------------------------------------------------------ Notes --
if st.session_state.current_notes:
    st.subheader(f"📝 Notes — {st.session_state.current_title}")
    st.markdown(st.session_state.current_notes)
    st.download_button(
        "⬇️ Download notes (.md)",
        data=st.session_state.current_notes,
        file_name=f"{st.session_state.current_title or 'notes'}.md",
        mime="text/markdown",
    )

# -------------------------------------------------------------------- Q&A --
has_library = bool(sources)

if st.session_state.current_source_id or has_library:
    st.subheader("💬 Ask Questions")

    scope_options = []
    if st.session_state.current_source_id:
        scope_options.append("current")
    if has_library:
        scope_options.append("library")

    if len(scope_options) > 1:
        scope = st.radio(
            "Search scope",
            options=scope_options,
            format_func=lambda o: "📄 This source" if o == "current" else "📚 Entire library",
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        scope = scope_options[0]

    scope_source_id = st.session_state.current_source_id if scope == "current" else None

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("📌 Sources used"):
                for i, chunk in enumerate(turn["sources"], start=1):
                    title = chunk.get("title") or chunk.get("video_id", "unknown")

                    position = None
                    if chunk.get("start") is not None and chunk.get("end") is not None:
                        position = f"`{chunk['start']}s → {chunk['end']}s`"
                    elif chunk.get("page") is not None:
                        position = f"page {chunk['page']}"

                    header = f"**Source {i}** — {title}" + (f" · {position}" if position else "")

                    ts_link = None
                    if (
                        chunk.get("source_type") == "youtube_video"
                        and chunk.get("video_id")
                        and chunk.get("start") is not None
                    ):
                        ts_link = f"https://www.youtube.com/watch?v={chunk['video_id']}&t={int(chunk['start'])}s"

                    st.markdown(f"{header} · [open ▶]({ts_link})" if ts_link else header)
                    st.caption(chunk["text"])

    question = st.chat_input(
        "Ask a question..." if scope == "current" else "Ask a question across your library..."
    )
    if question:
        try:
            with st.spinner("Searching and generating answer..."):
                query_embedding = embed_query(question)
                top_chunks = retrieve(
                    question, query_embedding, video_id=scope_source_id, top_k=config.RAG_TOP_K
                )
                history_pairs = [(t["question"], t["answer"]) for t in st.session_state.chat_history]
                answer = answer_question(question, top_chunks, chat_history=history_pairs)

            st.session_state.chat_history.append(
                {"question": question, "answer": answer, "sources": top_chunks}
            )
            st.rerun()
        except Exception as e:
            logger.exception("Q&A failed")
            st.error(f"❌ Something went wrong answering that question: {e}")
