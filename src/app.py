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
from chunker import chunk_segments
from downloader import DownloadError, download_audio
from embedder import embed_chunks, embed_query
from notes_generator import generate_notes
from rag_chat import answer_question
from retrieval import retrieve
from transcriber import transcribe_audio
from vector_store import (
    delete_video,
    list_videos,
    load_notes,
    save_video,
    video_exists,
)

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
    "current_video_id": None,
    "current_title": None,
    "current_notes": None,
    "chat_history": [],
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _load_video_into_state(video_id: str, title: str):
    st.session_state.current_video_id = video_id
    st.session_state.current_title = title
    st.session_state.current_notes = load_notes(video_id)
    st.session_state.chat_history = []


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("📚 Video Library")
    videos = list_videos()
    if not videos:
        st.caption("No videos processed yet.")
    for video in videos:
        row = st.columns([5, 1])
        label = video["title"][:42] + ("…" if len(video["title"]) > 42 else "")
        if row[0].button(label, key=f"open_{video['video_id']}", use_container_width=True):
            _load_video_into_state(video["video_id"], video["title"])
            st.rerun()
        if row[1].button("🗑️", key=f"del_{video['video_id']}"):
            delete_video(video["video_id"])
            if st.session_state.current_video_id == video["video_id"]:
                st.session_state.current_video_id = None
                st.session_state.current_title = None
                st.session_state.current_notes = None
                st.session_state.chat_history = []
            st.rerun()

# ------------------------------------------------------------------ Title --
st.title("🎯 Deep-Dive Video Note Taker")
st.write("Paste a YouTube URL to generate notes and ask questions about it.")

youtube_url = st.text_input("YouTube URL")

col1, col2 = st.columns(2)
with col1:
    selected_language = st.selectbox("Video Language", list(LANGUAGES.keys()))
with col2:
    whisper_size = st.selectbox(
        "Transcription quality",
        ["base", "small", "medium"],
        index=["base", "small", "medium"].index(config.WHISPER_MODEL_SIZE)
        if config.WHISPER_MODEL_SIZE in ["base", "small", "medium"]
        else 0,
        help="Larger models are slower but more accurate.",
    )

language_code = LANGUAGES[selected_language]

if st.button("Process Video", type="primary"):
    if not youtube_url:
        st.warning("Please paste a YouTube URL first.")
    elif not YOUTUBE_URL_RE.search(youtube_url):
        st.warning("That doesn't look like a YouTube URL.")
    else:
        try:
            with st.status("Processing video...", expanded=True) as status:
                status.write("⬇️ Downloading audio...")
                video_id, title, file_path = download_audio(youtube_url)

                if video_exists(video_id):
                    status.write("♻️ Already in your library - loading saved notes.")
                    _load_video_into_state(video_id, title)
                    status.update(label=f"Loaded from library: {title}", state="complete")
                else:
                    status.write("🎙️ Transcribing audio (this can take a minute)...")
                    transcript, segments, _ = transcribe_audio(
                        file_path, language=language_code, model_size=whisper_size
                    )

                    status.write("✂️ Chunking and embedding transcript...")
                    chunks = chunk_segments(
                        segments,
                        chunk_duration=config.CHUNK_DURATION_SECONDS,
                        overlap=config.CHUNK_OVERLAP_SECONDS,
                    )
                    if not chunks:
                        raise RuntimeError("No speech was detected in this video.")
                    chunks = embed_chunks(chunks)

                    status.write("📝 Generating notes...")
                    notes = generate_notes(transcript, segments)

                    status.write("💾 Saving to your library...")
                    save_video(video_id, title, youtube_url, notes, chunks)
                    _load_video_into_state(video_id, title)

                    status.update(label=f"Done! Processed: {title}", state="complete")
            st.rerun()
        except DownloadError as e:
            st.error(f"❌ Couldn't download this video: {e}")
        except Exception as e:
            logger.exception("Video processing failed")
            st.error(f"❌ Something went wrong while processing this video: {e}")

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
has_library = bool(videos)

if st.session_state.current_video_id or has_library:
    st.subheader("💬 Ask Questions")

    scope_options = []
    if st.session_state.current_video_id:
        scope_options.append("current")
    if has_library:
        scope_options.append("library")

    if len(scope_options) > 1:
        scope = st.radio(
            "Search scope",
            options=scope_options,
            format_func=lambda o: "📄 This video" if o == "current" else "📚 Entire library",
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        scope = scope_options[0]

    scope_video_id = st.session_state.current_video_id if scope == "current" else None

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("📌 Sources used"):
                for i, chunk in enumerate(turn["sources"], start=1):
                    title = chunk.get("title") or chunk.get("video_id", "unknown")
                    video_ref = chunk.get("video_id")
                    ts_link = (
                        f"https://www.youtube.com/watch?v={video_ref}&t={int(chunk['start'])}s"
                        if video_ref
                        else None
                    )
                    header = f"**Source {i}** — {title} · `{chunk['start']}s → {chunk['end']}s`"
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
                    question, query_embedding, video_id=scope_video_id, top_k=config.RAG_TOP_K
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
