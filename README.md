# 🎯 Deep-Dive Video Note Taker

A Retrieval-Augmented Generation (RAG) app that turns any YouTube video into
searchable notes: paste a URL, get structured notes with timestamps, and
ask follow-up questions answered from the video's own transcript.

## How it works

```
YouTube URL → yt-dlp (audio) → Whisper (transcript) → chunk + embed
                                                              │
                                                              ▼
                                          ChromaDB (persistent vector store)
                                                              │
                                          question (this video, or library-wide)
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                     vector search (semantic)        BM25 search (keyword)
                                              └───────────────┬───────────────┘
                                                               ▼
                                               fuse candidates (Reciprocal Rank Fusion)
                                                               ▼
                                                    rerank → top-k chunks
                                                               ▼
                                              Groq → grounded, cited answer
```

- **Download**: `yt-dlp` pulls the audio track as mp3.
- **Transcribe**: local `openai-whisper` produces a timestamped transcript.
- **Notes**: the transcript is sent to Groq to produce structured notes
  (key points, timestamps, action items).
- **Chunk + embed**: the transcript is split into overlapping ~30s chunks
  and embedded with `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Store**: chunks (with title/timestamp/source-type metadata), embeddings,
  notes, and a library index are persisted to disk with ChromaDB, so videos
  survive an app restart and don't need to be reprocessed.
- **Retrieve**: a question is answered using **hybrid retrieval** —
  [`src/retrieval.py`](src/retrieval.py) fuses a vector (semantic) search
  with a BM25 (keyword) search via Reciprocal Rank Fusion, so exact names,
  numbers, and phrases aren't lost to semantic search alone. The fused
  candidates are then narrowed to the final context chunks by
  [`src/reranker.py`](src/reranker.py). This runs identically whether
  you're asking about the **current video** or searching your **entire
  library** at once - pick either from the UI.
- **Ask**: [`src/rag_chat.py`](src/rag_chat.py) answers strictly from the
  retrieved chunks, citing each claim by a `(Source N, timestamp)` label,
  and says so plainly when the context isn't enough to answer.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/) and make sure it's on your `PATH`
   (required by yt-dlp/Whisper for audio extraction).
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your
   [Groq API key](https://console.groq.com/keys):
   ```bash
   copy .env.example .env
   ```
4. Run the app:
   ```bash
   streamlit run src/app.py
   ```

## Running tests

Fast unit tests (no network, no GPU) live under `tests/` and mock the
Whisper/Groq/yt-dlp calls:

```bash
pip install -r requirements-dev.txt
pytest
```

The `src/test_*.py` scripts are manual smoke tests that hit the real
network and a real Groq API key — run them individually with
`python src/test_download.py` etc. when you want to sanity-check the
full pipeline against a live video.

## Configuration

All tunables live in [`src/config.py`](src/config.py) and can be overridden
via environment variables — see `.env.example` for the full list (Whisper
model size, chunk duration/overlap, Groq model, storage paths, and the
retrieval settings below).

**Retrieval / hybrid search:**

| Variable | Default | What it does |
|---|---|---|
| `RAG_TOP_K` | `3` | Chunks finally sent to the LLM as context |
| `RAG_CANDIDATE_COUNT` | `15` | Candidates considered before reranking |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` \| `vector` \| `keyword` |
| `RERANK_METHOD` | `heuristic` | `heuristic` (no extra download) \| `cross_encoder` (higher quality, downloads a small model) \| `none` |
| `RERANK_WEIGHT_VECTOR/KEYWORD/EXACT` | `0.45/0.35/0.20` | Heuristic reranker blend weights |
| `HYBRID_RRF_K` | `60` | Reciprocal Rank Fusion constant |

## Project layout

```
src/
  app.py              Streamlit UI
  config.py            environment-driven settings
  downloader.py        YouTube → mp3 (yt-dlp)
  transcriber.py        mp3 → transcript (Whisper, cached model)
  chunker.py            transcript → overlapping time-based chunks
  embedder.py           chunks → embeddings (sentence-transformers)
  vector_store.py       persistent storage: ChromaDB + video library
  retrieval.py           hybrid (vector + BM25) candidate retrieval
  reranker.py            reranks candidates → final top-k chunks
  text_utils.py          shared tokenizer for keyword search/reranking
  notes_generator.py    transcript → structured notes (Groq)
  rag_chat.py            grounded, cited question answering (Groq)
  llm_client.py          shared Groq client, retries, key validation
tests/                  fast unit tests (mocked)
```

Downloaded audio lives in `downloads/`, and persisted notes/embeddings/
library index live in `data/` — both are git-ignored.
