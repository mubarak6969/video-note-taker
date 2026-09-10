"""Shared Groq client used by notes_generator and rag_chat.

Centralizing this avoids duplicating client setup / retry logic in every
module that talks to the LLM, and fails fast with a clear message if the
API key is missing instead of surfacing a cryptic error deep in a request.
"""
import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file locally, "
                "or to Streamlit secrets when deployed."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat_completion(prompt: str, model: str, max_retries: int = 2) -> str:
    """Sends a single-turn prompt to Groq, retrying transient failures."""
    client = get_client()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            logger.warning(
                "Groq request failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, e,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to get a response from the LLM: {last_error}") from last_error
