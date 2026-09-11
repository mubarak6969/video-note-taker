"""Streamlit entry point. Kept intentionally thin: environment/secrets
setup, session-state init, and composing the page out of src/ui/*
renderers - no business logic and no direct backend calls live here."""
import logging
import os

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

from ui.chat_ui import render_chat
from ui.ingestion_ui import render_ingestion_tabs
from ui.onboarding import render_onboarding
from ui.sidebar import render_sidebar
from ui.state import init_session_state
from ui.workspace import render_notes_tab, render_source_content_tab, workspace_icon

st.set_page_config(page_title="Deep-Dive Knowledge Assistant", page_icon="🎯", layout="wide")


def _ensure_groq_key_configured():
    if os.getenv("GROQ_API_KEY") or _has_secret("GROQ_API_KEY"):
        return
    st.error(
        "⚠️ GROQ_API_KEY is not configured. Add it to a `.env` file locally, "
        "or to Streamlit secrets when deployed, then reload the app."
    )
    st.stop()


_ensure_groq_key_configured()
init_session_state()

sources = render_sidebar()

st.title("🎯 Deep-Dive Knowledge Assistant")
st.write(
    "Turn YouTube videos, PDFs, text files, and audio/video recordings into structured "
    "notes, then ask questions answered strictly from what you've added - with every "
    "claim traceable back to its source."
)

if not sources:
    render_onboarding()

render_ingestion_tabs()

has_current_source = bool(st.session_state.current_source_id)
has_library = bool(sources)

if has_current_source:
    st.subheader(f"{workspace_icon()} {st.session_state.current_title}")
    tab_notes, tab_content, tab_chat = st.tabs(["📝 Notes", "📄 Source Content", "💬 Ask Questions"])
    with tab_notes:
        render_notes_tab()
    with tab_content:
        render_source_content_tab()
    with tab_chat:
        render_chat(has_current_source=True, has_library=has_library)
elif has_library:
    st.subheader("💬 Ask Questions")
    st.caption(
        "Select a source from your library to see its notes, or ask a question "
        "across everything you've processed."
    )
    render_chat(has_current_source=False, has_library=True)
