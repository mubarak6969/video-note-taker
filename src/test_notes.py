from transcriber import transcribe_audio
from notes_generator import generate_notes

# Use the same mp3 file
file_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3"

# Step 1: Transcribe
transcript, segments, transcript_path = transcribe_audio(file_path)

# Step 2: Generate Notes
notes = generate_notes(transcript, segments)

# Print the notes
print("\n📝 GENERATED NOTES:")
print(notes)