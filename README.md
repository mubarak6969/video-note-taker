# 🎯 Deep-Dive Video Note Taker

A Retrieval-Augmented Generation (RAG) app that turns YouTube videos, PDFs,
text documents, and audio/video files into searchable notes: add a source,
get structured notes with timestamp or page references, and ask follow-up
questions answered strictly from that source (or your whole library).

## How it works

```
YouTube URL ──── yt-dlp + Whisper ────┐
PDF upload  ──── pypdf (per page) ────┤
Text upload ──── read as-is ──────────┼──► IngestedSource (src/ingestion/)
Audio/video ──── Whisper ─────────────┘         (normalized text + segments)
                                                          │
                                                          ▼
                                              chunk + embed (unchanged)
                                                          │
                                                          ▼
                                      ChromaDB (persistent vector store)
                                                          │
                                      question (this source, or library-wide)
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

- **Ingest**: [`src/ingestion/`](src/ingestion/) converts any source - a
  YouTube URL, or an uploaded PDF/TXT/audio/video file - into one common
  `IngestedSource` (full text + normalized segments + ready-to-embed
  chunks). This is the only place that knows the differences between
  source types; everything downstream is unchanged regardless of where
  the content came from:
  - **YouTube**: `yt-dlp` downloads the audio, local `openai-whisper`
    transcribes it with timestamps.
  - **PDF**: `pypdf` extracts text per page; each chunk keeps its page
    number.
  - **Text file**: read directly, chunked by character count.
  - **Audio/video upload**: transcribed with the same Whisper pipeline as
    YouTube.
  - Uploads are identified by a **content hash**, not filename, so
    re-uploading the same file (even renamed) is recognized as a
    duplicate and skipped instead of reprocessed.
- **Notes**: the source's text is sent to Groq to produce structured notes
  (key points, timestamp/page references, action items).
- **Chunk + embed**: time-based sources get overlapping ~30s chunks
  ([`chunker.chunk_segments`](src/chunker.py)); page/text sources get
  overlapping character-based chunks ([`chunker.chunk_pages`](src/chunker.py)/
  `chunk_text_by_chars`). Every chunk is embedded with
  `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Store**: chunks (with title/timestamp-or-page/source-type metadata),
  embeddings, notes, and a library index are persisted to disk with
  ChromaDB, so sources survive an app restart and don't need to be
  reprocessed.
- **Retrieve**: a question is answered using **hybrid retrieval** —
  [`src/retrieval.py`](src/retrieval.py) fuses a vector (semantic) search
  with a BM25 (keyword) search via Reciprocal Rank Fusion, so exact names,
  numbers, and phrases aren't lost to semantic search alone. The fused
  candidates are then narrowed to the final context chunks by
  [`src/reranker.py`](src/reranker.py). This runs identically whether
  you're asking about the **current source** or searching your **entire
  library** (any mix of videos, PDFs, text, audio) at once - pick either
  from the UI.
- **Ask**: [`src/rag_chat.py`](src/rag_chat.py) answers strictly from the
  retrieved chunks, citing each claim by a `(Source N, timestamp-or-page)`
  label, and says so plainly when the context isn't enough to answer.

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

**Ingestion / uploads:**

| Variable | Default | What it does |
|---|---|---|
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `150` | Character-based chunking for PDF/text uploads |
| `UPLOAD_MAX_PDF_MB` | `25` | Max PDF size |
| `UPLOAD_MAX_TEXT_MB` | `5` | Max text file size |
| `UPLOAD_MAX_AUDIO_MB` | `200` | Max audio/video upload size |

## Supported sources

| Source | How | Chunk position metadata |
|---|---|---|
| YouTube URL | `yt-dlp` + Whisper | `start` / `end` (seconds) |
| PDF (`.pdf`) | `pypdf`, per page | `page` |
| Text (`.txt`) | read directly | none |
| Audio/video upload (`.mp3 .wav .m4a .mp4 .mov .webm .ogg .flac`) | Whisper | `start` / `end` (seconds) |

## Project layout

```
src/
  app.py              Streamlit UI (YouTube URL tab + Upload File tab)
  config.py            environment-driven settings
  ingestion/            source-ingestion abstraction (see below)
    types.py              IngestedSource - the common normalized shape
    errors.py             IngestionError - user-facing failure messages
    hashing.py             content-hash source ids (duplicate detection)
    youtube_source.py      YouTube URL → IngestedSource
    pdf_source.py           PDF upload → IngestedSource (pypdf, per-page)
    text_source.py          text upload → IngestedSource
    audio_source.py         audio/video upload → IngestedSource (Whisper)
  downloader.py         YouTube → mp3 (yt-dlp)
  transcriber.py         mp3/audio → transcript (Whisper, cached model)
  chunker.py             time-based AND character-based chunking
  embedder.py            chunks → embeddings (sentence-transformers)
  vector_store.py        persistent storage: ChromaDB + source library
  retrieval.py            hybrid (vector + BM25) candidate retrieval
  reranker.py             reranks candidates → final top-k chunks
  text_utils.py           shared tokenizer for keyword search/reranking
  notes_generator.py     source text → structured notes (Groq)
  rag_chat.py             grounded, cited question answering (Groq)
  llm_client.py           shared Groq client, retries, key validation
tests/                  fast unit tests (mocked)
```

Downloaded/uploaded audio lives in `downloads/` (uploads under
`downloads/uploads/`), and persisted notes/embeddings/library index live
in `data/` — both are git-ignored.

### Adding a new source type

Everything past ingestion (chunking granularity aside) is source-agnostic,
so adding a new type - a web article, a Word doc, a Slack export - means
writing one new `src/ingestion/<type>_source.py` that returns an
`IngestedSource`, then adding one `elif` in
`ingestion.ingest_uploaded_file()`'s dispatcher. `app.py`,
`vector_store.py`, `retrieval.py`, `reranker.py`, and `rag_chat.py` need
no changes.
