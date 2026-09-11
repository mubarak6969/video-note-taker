"""Landing / empty-library state - explains what the app does and how to
start, shown only until the first source has been processed."""
import streamlit as st


def render_onboarding():
    st.info(
        "**New here?** This app turns YouTube videos, PDFs, text files, and "
        "audio/video recordings into structured notes, then answers your "
        "questions using only what's actually in them - with every claim "
        "traced back to its exact source.",
        icon="👋",
    )

    steps = [
        ("1. Add a source", "Paste a YouTube URL below, or switch to **Upload File** for a PDF, text file, or audio/video recording."),
        ("2. Get structured notes", "An overview, key concepts, important points, and action items - generated automatically, with timestamps or page references."),
        ("3. Ask grounded questions", "Every answer cites exactly where it came from, or says plainly when it doesn't know."),
    ]
    columns = st.columns(3)
    for column, (heading, body) in zip(columns, steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.caption(body)
