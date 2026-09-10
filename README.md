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
                                  relevance gate: drop anything below
                                  RAG_RELEVANCE_THRESHOLD (weak evidence)
                                                           ▼
                                nothing survives?  ──►  "insufficient evidence"
                                       │                 (LLM never called)
                                       ▼
                                    Groq → answer, grounded + cited per claim
                                                           ▼
                                extract_cited_sources() → concise "Sources" panel
                                     (only the evidence actually used, each with
                                      title + timestamp/page + clickable link)
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
- **Notes**: [`src/notes_generator.py`](src/notes_generator.py) sends the
  source's text to Groq to produce study-quality structured notes with
  five fixed sections - **Overview**, **Key Concepts**, **Important
  Points**, **Actionable Takeaways**, **Source References** - each
  referencing timestamps or page numbers where the source has them, and
  worded for what the source actually is (a PDF gets "page" language, a
  video gets timestamps, plain text gets neither).
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
  [`src/reranker.py`](src/reranker.py), which also applies a **relevance
  gate** (see below) that drops evidence too weak to trust. This runs
  identically whether you're asking about the **current source** or
  searching your **entire library** (any mix of videos, PDFs, text,
  audio) at once - pick either from the UI.
- **Ask**: [`src/rag_chat.py`](src/rag_chat.py) answers strictly from the
  retrieved chunks, citing each claim by a `(Source N, timestamp-or-page)`
  label, and says so plainly when the context isn't enough to answer -
  the LLM is never even called if the relevance gate found nothing worth
  answering from. Recent conversation turns are passed along too, so a
  follow-up like "what about its byproduct?" resolves against the prior
  question. `extract_cited_sources()` then parses which `Source N` labels
  the answer actually used and maps them back to the real chunk metadata
  - never to a title, timestamp, or page the model merely typed in its
  prose - so the UI's "Sources" panel is both concise (only the evidence
  actually used) and trustworthy (always ground-truth, never hallucinated).

## When the app says it doesn't have enough evidence

Every retrieved chunk gets a **relevance_score** (0-1) alongside its
ranking score, and [`reranker.rerank()`](src/reranker.py) drops anything
below `RAG_RELEVANCE_THRESHOLD` (default `0.30`) before it ever reaches
the LLM. If that empties the result set, the app shows "no relevant
evidence found" instead of generating an answer - the exact same
code path as asking about a source that was never processed, so a weak
match and no match are treated identically: as insufficient context, not
as license to guess.

`relevance_score` is deliberately **not** the same thing as the score
used to order results (`rerank_score`). Ranking scores from hybrid
retrieval are Reciprocal-Rank-Fusion-based - they reflect a candidate's
*position* among this query's own pool, so even a barely-related chunk
can rank respectably if nothing better happened to be around. A gate
built on that would only ever detect "we found less than usual," not
"what we found isn't actually relevant." `relevance_score` instead uses
whichever signal genuinely measures relevance for the method in use:

- **`heuristic`** (default): `max(raw cosine similarity, exact keyword
  overlap)`. Either strong semantic similarity *or* a strong exact match
  is independently sufficient - this is what lets a rare product code
  with near-zero embedding similarity still count as good evidence
  (exactly the case hybrid retrieval exists to rescue), while a vague
  question sharing only function words ("is", "the", "what") with an
  unrelated chunk doesn't score a false positive - overlap is computed
  after stripping stopwords for exactly this reason.
- **`cross_encoder`**: the model's own relevance score, passed through a
  sigmoid so it lands in the same roughly-(0,1) range as the heuristic
  score - a cross-encoder's raw output is an unbounded logit, not a
  similarity, so without this a single global threshold wouldn't mean
  the same thing across methods.
- **`none`**: never gated - this method exists to pass candidates through
  untouched for debugging/comparison, so gating it would defeat the point.

Set `RAG_RELEVANCE_THRESHOLD=0` to disable the gate entirely (restores
pre-gate behavior: always answer from whatever was retrieved, however
weak).

## UI states

The Q&A flow distinguishes four outcomes rather than collapsing them into
one generic result:

| State | What the user sees |
|---|---|
| Relevant answer | Chat bubble + a "Sources (N)" panel of card-style citations (source type icon, title, timestamp/page, evidence preview, clickable link where supported) |
| Insufficient evidence | A distinct `st.info` note (not styled as a confident answer) explaining nothing relevant was found, with no sources panel |
| Retrieval failure | "Something went wrong while searching your library" - the question is never sent to the LLM |
| System/API failure | "The AI service didn't respond" - shown only when retrieval succeeded but the LLM call failed |

Progress within each stage is shown as plain status text ("Retrieving
relevant context…", "Found 3 relevant passage(s)", "Generating a grounded
answer…") via `st.status`, not a fabricated percentage bar - there's no
way to know how "done" a retrieval or an LLM call is partway through, so
the UI doesn't pretend otherwise. Errors never surface a raw exception
message or stack trace; the real one is logged server-side via
`logger.exception()`.

An empty library shows onboarding guidance instead of a blank page, and
re-asking the identical question in the identical scope within the same
session reuses the cached answer instead of spending another retrieval +
LLM call.

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

## RAG evaluation

[`tests/evaluation/`](tests/evaluation/) is a small, deterministic RAG
quality suite - it runs as part of the normal `pytest` run above, not as
a separate tool. It seeds a hand-written 3-source corpus (a YouTube-style
video, a PDF, and a text doc - [`golden_dataset.py`](tests/evaluation/golden_dataset.py))
through the **real** embedding model, ChromaDB, hybrid retrieval, and
reranker - only the Groq call is ever mocked, and only for the two checks
that inherently need it (see below). Nothing here costs an API call or a
network round-trip.

Nine golden questions cover the categories that matter for RAG quality:

| Category | What it checks |
|---|---|
| `exact_keyword` | An exact code/number is retrieved even from a whole-corpus BM25+vector search |
| `semantic` | A paraphrased question with **no keyword overlap** still retrieves the right chunk via embeddings |
| `cross_source` | A library-wide question retrieves evidence from more than one source |
| `source_attribution` | Retrieved chunks carry the correct title/source_type, and `describe_source()` renders them correctly |
| `insufficient_context` | Two cases, both returning `[]` and the canned fallback **without calling the LLM**: a question scoped to an unprocessed source (empty pool), and an off-topic question scoped to a source that *does* have chunks (the relevance gate rejecting weak evidence, not just an empty pool) |
| `follow_up` | A pronoun-only follow-up ("what about its byproduct?") still retrieves the right chunk *and clears the relevance gate*, and the prior Q/A pair is verified present in the constructed prompt (LLM mocked to capture the prompt, not to grade its reply) |

A final `test_overall_retrieval_hit_rate_meets_quality_gate` test computes
a hit-rate across every retrieval-graded question and asserts it stays
at or above 90% - a concrete, re-runnable regression gate for chunking/
embedding/retrieval-setting changes, not just a pass/fail per question.

This intentionally does **not** grade the LLM's prose (that would need a
live call and a judge model - out of scope for a free, deterministic,
CI-friendly suite); it grades the parts of the pipeline that are
deterministic and that the LLM's answer quality depends on: did
retrieval find the right evidence, is it correctly attributed, and does
the prompt actually carry that evidence and conversation history.

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
| `RAG_RELEVANCE_THRESHOLD` | `0.30` | Minimum `relevance_score` to keep a chunk as evidence; `0` disables the gate. See ["When the app says it doesn't have enough evidence"](#when-the-app-says-it-doesnt-have-enough-evidence) above. |

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
  notes_generator.py     source text → structured 5-section notes (Groq)
  rag_chat.py             grounded, cited Q&A + citation extraction (Groq)
  llm_client.py           shared Groq client, retries, key validation
tests/                  fast unit tests (mocked)
  evaluation/              deterministic RAG quality suite (see above)
```

Downloaded/uploaded audio lives in `downloads/` (uploads under
`downloads/uploads/`), and persisted notes/embeddings/library index/raw
transcripts live in `data/` (`chroma_db/`, `library.json`, `notes/`,
`transcripts/`) — both directories are git-ignored. Both the generated
notes and the raw extracted/transcribed text are downloadable from the
UI once a source is open.

### Adding a new source type

Everything past ingestion (chunking granularity aside) is source-agnostic,
so adding a new type - a web article, a Word doc, a Slack export - means
writing one new `src/ingestion/<type>_source.py` that returns an
`IngestedSource`, then adding one `elif` in
`ingestion.ingest_uploaded_file()`'s dispatcher. `app.py`,
`vector_store.py`, `retrieval.py`, `reranker.py`, and `rag_chat.py` need
no changes.
