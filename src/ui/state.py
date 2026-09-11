"""Session-state management for the Streamlit UI - the only module that
reads/writes these st.session_state keys directly, so the rest of the UI
layer treats them as an opaque "current source" concept."""
import streamlit as st

import vector_store

DEFAULTS = {
    "current_source_id": None,
    "current_title": None,
    "current_notes": None,
    "current_transcript": None,
    "chat_history": [],
    "qa_cache": {},
}


def init_session_state():
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def load_source(source_id: str, title: str):
    """Makes `source_id` the active source: loads its saved notes/
    transcript and starts a fresh conversation."""
    st.session_state.current_source_id = source_id
    st.session_state.current_title = title
    st.session_state.current_notes = vector_store.load_notes(source_id)
    st.session_state.current_transcript = vector_store.load_transcript(source_id)
    st.session_state.chat_history = []
    st.session_state.qa_cache = {}


def clear_current_source():
    st.session_state.current_source_id = None
    st.session_state.current_title = None
    st.session_state.current_notes = None
    st.session_state.current_transcript = None
    st.session_state.chat_history = []
    st.session_state.qa_cache = {}
