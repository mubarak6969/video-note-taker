import streamlit as st
from downloader import download_audio
from transcriber import transcribe_audio
from notes_generator import generate_notes
from chunker import chunk_segments
from embedder import embed_chunks
from embedder import embed_query
from vector_store import search_chunks
from rag_chat import answer_question

st.set_page_config(page_title="Deep-Dive Video Note Taker", page_icon="🎯")

st.title("🎯 Deep-Dive Video Note Taker")
st.write("Paste a YouTube URL to generate notes and ask questions about it.")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "notes" not in st.session_state:
    st.session_state.notes = None

youtube_url = st.text_input("YouTube URL")

# Language options: display name -> Whisper language code
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

selected_language = st.selectbox("Video Language", list(LANGUAGES.keys()))
language_code = LANGUAGES[selected_language]

if st.button("Process Video"):
    if not youtube_url:
        st.warning("Please paste a YouTube URL first.")
    else:
        with st.spinner("Downloading audio..."):
            title, file_path = download_audio(youtube_url)
        
        with st.spinner("Transcribing audio... (this can take a minute)"):
            transcript, segments, transcript_path = transcribe_audio(file_path, language=language_code)
        
        with st.spinner("Chunking and embedding transcript..."):
            chunks = chunk_segments(segments, chunk_duration=30)
            chunks = embed_chunks(chunks)
            st.session_state.chunks = chunks
        
        with st.spinner("Generating notes..."):
            notes = generate_notes(transcript, segments)
            st.session_state.notes = notes
        
        st.success(f"Done! Processed: {title}")

if st.session_state.notes:
    st.subheader("📝 Generated Notes")
    st.write(st.session_state.notes)

# Q&A Section
if st.session_state.chunks:
    st.subheader("💬 Ask Questions About This Video")
    question = st.text_input("Your question")
    
    if st.button("Ask"):
        if question:
            with st.spinner("Searching transcript and generating answer..."):
                query_embedding = embed_query(question)
                top_chunks = search_chunks(query_embedding, st.session_state.chunks, top_k=3)
                answer = answer_question(question, top_chunks)
            
            st.write("**Answer:**")
            st.write(answer)
            
            with st.expander("📌 Source chunks used"):
                for chunk in top_chunks:
                    st.write(f"[{chunk['start']}s → {chunk['end']}s]: {chunk['text']}")
        else:
            st.warning("Please type a question.")