from sentence_transformers import SentenceTransformer

# Load once, reuse for all embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    """
    Takes a list of chunks (with 'text' key) and adds an 'embedding' to each.
    """
    print("🧠 Generating embeddings...")
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    
    print(f"✅ Generated {len(chunks)} embeddings!")
    return chunks


def embed_query(query):
    """
    Embeds a single question/query the same way.
    """
    return model.encode([query])[0]