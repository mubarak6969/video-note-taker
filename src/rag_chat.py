from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query, top_chunks):
    """
    Takes the user's question + relevant chunks, asks the LLM to answer
    using ONLY that context.
    """
    print("🤖 Asking Groq...")
    
    context = ""
    for chunk in top_chunks:
        context += f"[{chunk['start']}s - {chunk['end']}s]: {chunk['text']}\n\n"
    
    prompt = f"""
Answer the question using ONLY the context below. If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content