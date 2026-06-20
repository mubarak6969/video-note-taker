def chunk_segments(segments, chunk_duration=30):
    """
    Groups Whisper segments into chunks based on time duration.
    Each chunk = ~30 seconds of speech combined into one block of text.
    """
    chunks = []
    current_chunk_text = ""
    chunk_start_time = segments[0]["start"]
    
    for segment in segments:
        current_chunk_text += segment["text"] + " "
        
        # If this chunk has covered enough duration, save it and start a new one
        if segment["end"] - chunk_start_time >= chunk_duration:
            chunks.append({
                "text": current_chunk_text.strip(),
                "start": chunk_start_time,
                "end": segment["end"]
            })
            current_chunk_text = ""
            chunk_start_time = segment["end"]
    
    # Add any leftover text as the final chunk
    if current_chunk_text.strip():
        chunks.append({
            "text": current_chunk_text.strip(),
            "start": chunk_start_time,
            "end": segments[-1]["end"]
        })
    
    return chunks