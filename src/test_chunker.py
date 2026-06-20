from transcriber import transcribe_audio
from chunker import chunk_segments

file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

transcript, segments, transcript_path = transcribe_audio(file_path)

chunks = chunk_segments(segments, chunk_duration=30)

print(f"\n📦 Total chunks created: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{chunk['start']}s → {chunk['end']}s]:")
    print(chunk['text'])
    print("---")