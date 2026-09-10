import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Loaded once per process and reused for every embedding call.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    """
    Takes a list of chunks (with 'text' key) and adds an 'embedding' to each.
    """
    logger.info("Generating embeddings for %d chunks...", len(chunks))

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    logger.info("Generated %d embeddings.", len(chunks))
    return chunks


def embed_query(query):
    """
    Embeds a single question/query the same way.
    """
    return model.encode([query])[0]
