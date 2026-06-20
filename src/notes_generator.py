from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_notes(transcript, segments):
    print("🤖 Sending transcript to Groq...")
    
    # Format timestamps from segments
    timestamps = ""
    for segment in segments:
        start = round(segment["start"], 2)
        text = segment["text"]
        timestamps += f"[{start}s]: {text}\n"
    
    # The prompt we send to the LLM
    prompt = f"""
You are a helpful assistant that creates structured notes from video transcripts.

Here is the full transcript:
{transcript}

Here are the timestamps:
{timestamps}

Please provide:
1. STRUCTURED NOTES: Key points organized by topic
2. KEY TIMESTAMPS: Most important moments with their times
3. ACTION ITEMS: Things to do or remember from this video

Format your response clearly with these 3 sections.
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    notes = response.choices[0].message.content
    print("✅ Notes generated!")
    return notes