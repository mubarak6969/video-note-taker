from transcriber import transcribe_audio
from chunker import chunk_segments
from embedder import embed_chunks

file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

transcript, segments, transcript_path = transcribe_audio(file_path)
chunks = chunk_segments(segments, chunk_duration=30)
chunks = embed_chunks(chunks)

print(f"\nFirst chunk text: {chunks[0]['text'][:50]}...")
print(f"Embedding shape: {chunks[0]['embedding'].shape}")
print(f"First 5 numbers: {chunks[0]['embedding'][:5]}")