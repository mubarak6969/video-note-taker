import whisper
import os

def transcribe_audio(file_path, language=None):
    print(f"Loading Whisper model...")
    model = whisper.load_model("base")
    
    print(f"Transcribing: {file_path}")
    
    options = {}
    if language:
        options["language"] = language
    
    result = model.transcribe(file_path, **options)
    
    transcript = result["text"]
    segments = result["segments"]
    
    transcript_path = os.path.splitext(file_path)[0] + "_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print(f"✅ Transcription complete!")
    print(f"📁 Transcript saved at: {transcript_path}")
    return transcript, segments, transcript_path