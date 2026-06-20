from transcriber import transcribe_audio

file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

transcript, segments, transcript_path = transcribe_audio(file_path)

print("\n📝 FULL TRANSCRIPT:")
print(transcript)

print("\n⏱️ TIMESTAMPS (first 5 segments):")
for segment in segments[:5]:
    start = round(segment["start"], 2)
    end = round(segment["end"], 2)
    text = segment["text"]
    print(f"[{start}s → {end}s]: {text}")

print(f"\n💾 Saved to: {transcript_path}")