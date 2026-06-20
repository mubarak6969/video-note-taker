import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Measures how similar two vectors are (meaning-wise).
    Returns a score between -1 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def search_chunks(query_embedding, chunks, top_k=3):
    """
    Finds the most relevant chunks for a given query embedding.
    Returns the top_k most similar chunks.
    """
    scored_chunks = []
    
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))
    
    # Sort by score, highest first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Return just the top_k chunks
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    return top_chunks