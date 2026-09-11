"""Streamlit presentation layer. Every module here may import streamlit
and read/write st.session_state; business logic (ingestion, retrieval,
answer generation) lives in src/services.py and is only ever called from
here, never reimplemented here.
"""
