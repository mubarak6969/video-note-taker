"""Tiny shared text helpers used by both keyword search and reranking."""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# A short, standard list of English function words. Used only to filter
# the *query* side of exact-overlap scoring (reranker._exact_overlap_score)
# so that sharing "is"/"the"/"what" with a chunk doesn't count as evidence
# of relevance - without this, an off-topic question can score a
# deceptively high overlap purely by chance, and a genuine but short
# follow-up ("what about its byproduct?") gets its one real content word
# diluted by three meaningless ones. BM25 keyword search is untouched by
# this list - its own IDF weighting already down-weights common terms.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "about", "what", "who",
    "whom", "whose", "when", "where", "why", "how", "which", "its", "it",
    "this", "that", "these", "those", "and", "or", "but", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may",
    "might", "must", "i", "you", "he", "she", "we", "they", "my", "your",
    "his", "her", "our", "their", "as", "by", "from", "into", "than",
    "then", "so", "if", "not", "no", "yes", "there", "here", "up", "down",
    "out", "over", "under", "again", "further", "just", "also", "me",
    "him", "them", "us", "am", "having", "having",
})


def tokenize(text: str) -> list:
    """Lowercases and splits into alphanumeric tokens (no stemming - good
    enough for BM25/overlap scoring on short transcript chunks, and keeps
    this dependency-free)."""
    return _WORD_RE.findall((text or "").lower())


def content_tokens(text: str) -> list:
    """tokenize() with stopwords removed - the meaningful words a query
    or chunk is actually "about", used where common-word overlap would
    otherwise be mistaken for evidence of relevance."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
