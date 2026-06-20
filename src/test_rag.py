from transcriber import transcribe_audio
from chunker import chunk_segments
from embedder import embed_chunks, embed_query
from vector_store import search_chunks
from rag_chat import answer_question

file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

transcript, segments, transcript_path = transcribe_audio(file_path)
chunks = chunk_segments(segments, chunk_duration=30)
chunks = embed_chunks(chunks)

query = "What does the speaker promise to never do?"
query_embedding = embed_query(query)
top_chunks = search_chunks(query_embedding, chunks, top_k=2)

answer = answer_question(query, top_chunks)

print(f"\n❓ Question: {query}")
print(f"\n💡 Answer:\n{answer}")