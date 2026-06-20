from transcriber import transcribe_audio
from chunker import chunk_segments
from embedder import embed_chunks, embed_query
from vector_store import search_chunks

file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

transcript, segments, transcript_path = transcribe_audio(file_path)
chunks = chunk_segments(segments, chunk_duration=30)
chunks = embed_chunks(chunks)

# Ask a question!
query = "What does the speaker promise to never do?"
query_embedding = embed_query(query)

top_chunks = search_chunks(query_embedding, chunks, top_k=2)

print(f"\n❓ Query: {query}\n")
print("🔍 Most relevant chunks found:\n")
for i, chunk in enumerate(top_chunks):
    print(f"Match {i+1} [{chunk['start']}s → {chunk['end']}s]:")
    print(chunk['text'])
    print("---")