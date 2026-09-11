# 🎯 Deep-Dive Knowledge Assistant

*(repo/directory name `video-note-taker` for history - the app itself is
titled "Deep-Dive Knowledge Assistant" now that it covers more than video)*

A Retrieval-Augmented Generation (RAG) app that turns YouTube videos, PDFs,
text documents, and audio/video files into searchable notes: add a source,
get structured notes with timestamp or page references, and ask follow-up
questions answered strictly from that source (or your whole library) -
never inventing an answer when the evidence isn't there.

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
   (required by yt-dlp/Whisper for audio extraction, and by `ffprobe` -
   bundled with every standard ffmpeg install - for the duration cap in
   `src/media_probe.py`).
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
| `UPLOAD_MAX_AUDIO_MINUTES` | `90` | Max duration (checked via `ffprobe` after the file is on disk, before transcribing) - also applied to YouTube videos, after download but before the expensive Whisper step |

**Public-access protection** (unset = open to anyone with the URL - see [Deploying publicly](#deploying-publicly)):

| Variable | Default | What it does |
|---|---|---|
| `APP_PASSWORD` | *(unset)* | If set, gates the whole app behind this single shared password |
| `RATE_LIMIT_INGESTIONS_PER_HOUR` | `10` | Max sources one browser session can process per hour; `0` disables |
| `RATE_LIMIT_QUESTIONS_PER_HOUR` | `60` | Max questions one browser session can ask per hour; `0` disables |

## Supported sources

| Source | How | Chunk position metadata |
|---|---|---|
| YouTube URL | `yt-dlp` + Whisper | `start` / `end` (seconds) |
| PDF (`.pdf`) | `pypdf`, per page | `page` |
| Text (`.txt`) | read directly | none |
| Audio/video upload (`.mp3 .wav .m4a .mp4 .mov .webm .ogg .flac`) | Whisper | `start` / `end` (seconds) |

## Security considerations

This app accepts three kinds of untrusted input - a URL, an uploaded
file, and (indirectly) whatever text ends up in a transcript/document -
and spends real money (Groq API calls) and real compute (Whisper,
embeddings) processing them. What's mitigated, and how:

| Risk | Mitigation |
|---|---|
| SSRF / URL spoofing via the YouTube field | [`url_validation.is_youtube_url()`](src/url_validation.py) parses the URL and checks its actual hostname against an allowlist - not a substring match, which a URL like `https://evil.example/?next=youtube.com/watch` would slip past. Enforced twice: once in the UI for fast feedback, and again as a hard backstop inside [`downloader.download_audio()`](src/downloader.py) itself, so it can't be bypassed by any other caller. |
| Path traversal via a source id | YouTube video ids are validated against `^[A-Za-z0-9_-]{1,32}$` right after download; every place a `source_id` becomes a filesystem path ([`vector_store.py`](src/vector_store.py)) re-validates it against the same shape before `os.path.join` ever sees it. Upload ids are SHA-256 hex hashes, which are inherently safe. |
| Unsafe download filenames | A source's display title (from an uploaded filename or a YouTube title) is sanitized ([`ui/formatting.safe_download_filename()`](src/ui/formatting.py)) before being used as a `st.download_button` filename, stripping path separators and control characters. |
| Oversized/malicious uploads | Per-type size ceilings (`UPLOAD_MAX_*_MB`) enforced before any parsing; a separate duration cap (`UPLOAD_MAX_AUDIO_MINUTES`, via `ffprobe`) catches a small-but-long file the size cap alone wouldn't. A maliciously-crafted PDF designed to be slow to parse (a "PDF bomb") is bounded by the size cap but not by a parse-time timeout - see Known limitations. |
| Prompt injection from retrieved content | A transcript or document is exactly the kind of thing an attacker could seed with text like "ignore previous instructions...". The RAG prompt ([`rag_chat.py`](src/rag_chat.py)) explicitly tells the model the CONTEXT is untrusted reference material to quote/cite, never instructions to follow - see the test in `tests/test_citations.py`. |
| XSS via LLM output or citations | Answers and citations render through `st.markdown`/Streamlit-native components only - `unsafe_allow_html` is never used anywhere in the UI, so even if a transcript (or the model echoing it) contained raw HTML/JS, Streamlit escapes it rather than executing it. |
| Secret exposure | `GROQ_API_KEY`/`APP_PASSWORD` are read from environment/`st.secrets` only, never logged, never echoed in an error message shown to the user, and never written to a file the app itself creates. `.env`, `cookies.txt`, and `data/` are git-ignored. |
| Corrupted local state crashing the app | A malformed `data/library.json` is caught and treated as an empty library (logged loudly server-side) rather than crashing on every page load - see `vector_store._load_library()`. |
| No accounts / open cost exposure | This app has no user accounts. Before any public deployment, set `APP_PASSWORD` (a single shared password, `hmac.compare_digest`-compared) and the `RATE_LIMIT_*` variables - see [Deploying publicly](#deploying-publicly). |

Subprocess use (`ffprobe` for duration checks; `ffmpeg`/`yt-dlp` internally
for audio extraction) always passes arguments as a list, never a shell
string, so there's no shell-injection surface from a filename or URL.

## Deploying publicly

The app has **no accounts and no per-user isolation** - it's built for a
single owner (you) to use, not for arbitrary strangers to share. Before
putting it on a public URL:

1. Set `APP_PASSWORD` in your host's secrets/environment. Every visitor
   sees a password prompt before anything else renders; leaving it unset
   means anyone with the link can trigger real Whisper compute and real
   Groq spend.
2. Set `RATE_LIMIT_INGESTIONS_PER_HOUR` / `RATE_LIMIT_QUESTIONS_PER_HOUR`
   to something you're comfortable paying for even if someone runs up
   against them repeatedly (the defaults - 10 sources/hour, 60
   questions/hour - are conservative starting points, not a hard
   recommendation).
3. Consider lowering `UPLOAD_MAX_AUDIO_MB`/`UPLOAD_MAX_AUDIO_MINUTES` and
   using `WHISPER_MODEL_SIZE=base` on a free/small hosting tier - a
   larger Whisper model or a long file can exceed typical platform
   request-timeout limits (see Known limitations).

This is a **single shared password**, not multi-user authentication -
everyone who has it sees the same library. That's an appropriate,
honest tradeoff for a portfolio deployment; it is not what you'd want
for a product with real, separate user accounts.

## Deployment

**Recommended: [Streamlit Community Cloud](https://streamlit.io/cloud).**
The repo is already shaped for it - `requirements.txt` (pinned) and
`packages.txt` (`ffmpeg`, installed via `apt-get` at build time) are
exactly Streamlit Cloud's native reproducibility mechanism, so no
Dockerfile or build script is needed. `packages.txt` lists only `ffmpeg`
deliberately - on Debian/Ubuntu (what Streamlit Cloud builds on) the
`ffmpeg` apt package already bundles `ffprobe`, and there is no separate
`ffprobe` apt package to add - listing one would fail the build. Steps:

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `src/app.py`.
3. In the app's **Secrets**, add `GROQ_API_KEY` (required) and, before
   sharing the link publicly, `APP_PASSWORD` (see above). Any other
   `.env.example` variable can go here too, using the same names.
4. Deploy. First load is slow (downloading the Whisper and embedding
   models); subsequent loads reuse the same container.

**Alternative: [Render](https://render.com)** (or any host with a
persistent disk), if you want your library to durably survive restarts -
see the persistence note below. Deploy as a native Python web service
(build command `pip install -r requirements.txt`, start command
`streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`);
add `ffmpeg` via Render's native/Aptfile buildpack support, or switch to
a container deploy if you'd rather pin the OS image directly. Attach a
persistent disk mounted at `data/` (and `downloads/` if you want cached
audio to survive too) for durable storage.

**Ruled out: Vercel** (and other edge/serverless-function platforms).
This app needs a persistent process, not a short-lived function: Whisper
and `sentence-transformers` are large ML dependencies that must stay
loaded in memory across requests to be fast (the whole point of the
model-caching in `transcriber.py`/`embedder.py`), transcription can run
for minutes on a long file, and ChromaDB needs a real writable
filesystem. Serverless platforms cap execution time (often 10-60s),
cap deployment size in ways `torch`+`whisper` will blow through, and give
each invocation a fresh, mostly-read-only filesystem - none of which
this architecture can work around without becoming a fundamentally
different (and much more complex) system. Streamlit Cloud/Render's
long-running-container model is what this app actually needs.

**Persistence, honestly:** ChromaDB + a JSON library index on local disk
is the right amount of infrastructure for a single-user app - there is
no need for a hosted database. But "local disk" means different things
per platform: on Streamlit Community Cloud, storage can be wiped when the
app sleeps from inactivity and wakes back up, so treat a Cloud deployment
as a **live demo of the app's capabilities**, not a durable personal
archive - a library built up over a session isn't guaranteed to survive
long idle periods. Render's persistent disk (or self-hosting on a VPS)
gives genuinely durable local storage. Neither **Supabase** nor **object
storage (S3-compatible)** is introduced here: Supabase would only earn
its keep with multi-user accounts and per-user data isolation, which
this app deliberately doesn't have yet; object storage would only be
worth the complexity on a platform with *zero* persistent filesystem
(the class of platform already ruled out above). If durability on
Streamlit Cloud specifically ever becomes a requirement, periodically
syncing `data/` to S3-compatible storage is a bounded, well-understood
next step - not something to build speculatively now.

**Docker:** not included, and not required for either recommended
platform - both consume `requirements.txt`/`packages.txt` natively, which
already pins every Python and OS-level dependency precisely enough to
reproduce the environment. A Dockerfile would be a reasonable addition
if you later want to deploy to a raw VPS or a container-based host
instead (a straightforward `python:3.11-slim` base + `apt-get install
ffmpeg` + `pip install -r requirements.txt` would cover it) - not added
speculatively here, since it isn't build-verified in this pass and
neither recommended platform needs it.

**Background processing:** ingestion is fully synchronous - the request
that clicks "Process video" blocks until download, transcription,
embedding, and note generation all finish. For this app's realistic
single-user/portfolio traffic, a queue (Celery/RQ + Redis, or similar)
would be meaningfully more infrastructure than the problem justifies, so
none is introduced. The practical mitigation already in place is the
`UPLOAD_MAX_AUDIO_MINUTES` duration cap, which bounds the worst case
before the expensive step even starts. If this ever needs true
background processing (e.g. because a hosting platform's request timeout
is shorter than a real transcription job), the smallest next step is a
background thread pool within the same process - not a distributed queue
- escalating further only if concurrent load actually demands it.

## Known limitations

- **No multi-user isolation.** One shared library, one optional shared
  password. Not a substitute for real per-user accounts.
- **Synchronous ingestion.** A long video/audio file blocks the request
  until it's fully transcribed; very long files may exceed a hosting
  platform's request-timeout even with the duration cap applied.
- **Storage durability depends on the host.** See the Persistence note
  above - Streamlit Community Cloud's free tier is not guaranteed durable
  across sleep/wake cycles.
- **PDF parsing has no timeout.** The size cap (`UPLOAD_MAX_PDF_MB`)
  bounds the worst case, but a pathologically-crafted PDF could still be
  slow to parse; no per-file wall-clock timeout is enforced.
- **BM25 keyword search rebuilds its index on every query**, scoped to
  the current search (one source or the whole library). Fine at
  portfolio scale (dozens to low hundreds of chunks); would need caching
  or a persistent inverted index at meaningfully larger scale.
- **No distributed rate limiting.** `RATE_LIMIT_*` is per-*browser-session*
  in a single process - restarting the app or opening a new session
  resets it. It's a cost safety net, not abuse-proof.
- **`RAG_RELEVANCE_THRESHOLD=0.30` is tuned for `all-MiniLM-L6-v2`** with
  this app's chunking. A different embedding model or very different
  content (e.g. much longer chunks) would likely need retuning - the
  evaluation suite (`tests/evaluation/`) is the tool to retune it with.

## Project layout

```
src/
  app.py              Streamlit entry point - thin composition root only:
                        env/secrets check, session-state init, and
                        wiring src/ui/* renderers together. No business
                        logic and no direct backend calls live here.
  services.py           UI ↔ backend orchestration (finish_ingestion,
                          answer_with_retrieval) - framework-agnostic,
                          no Streamlit imports, so it's easy to reason
                          about independent of the UI.
  ui/                    Streamlit presentation layer (see below)
    state.py               session-state init/load/clear
    sidebar.py              library list: select/delete a source
    onboarding.py            empty-library landing state
    ingestion_ui.py          YouTube URL + Upload File tabs
    workspace.py             Notes / Source Content tabs
    chat_ui.py               Ask Questions tab: scope, history, citations
    access_control.py        optional password gate + rate limiting (UI glue)
    constants.py             icons/labels/language list shared by the above
    formatting.py            pure display-formatting helpers
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
  url_validation.py      is_youtube_url() - hostname-based URL check
  media_probe.py          ffprobe-based duration check (fails open)
  rate_limiter.py          pure per-session rate-limit logic
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

The UI is a single-page workspace: an empty library shows onboarding and
the two ingestion tabs; once a source is selected, its notes, raw source
content, and a chat scoped to either that source or the whole library
sit in one tabbed workspace, so the read → ask → verify loop never
requires leaving the page. `.streamlit/config.toml` pins a fixed light
theme (indigo accent) for a consistent look regardless of viewer OS
settings.

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
