import os
import json
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rag import get_movie_candidates

load_dotenv()

# =====================================================================
# CONFIGURATION & PROMPTS
# =====================================================================

SYSTEM_INSTRUCTION = """
You are a film expert recommending movies. 

If the user's prompt is too vague (e.g., just a country, just a broad genre), ask 1 or 2 clarifying questions to narrow down their preferences.

FALLBACK RULE (CIRCUIT BREAKER):
You must not get stuck in an endless loop of questions. Ask a MAXIMUM of 2 clarifying questions per conversation. If the user is still vague after that, immediately make a "best guess" recommendation of diverse, highly-rated movies that loosely fit whatever minimal context they provided.

Once you have enough context, OR if the fallback rule is triggered, recommend between 1 and 5 specific movies that fit the criteria.

You MUST respond ONLY with a valid JSON object. Do not include markdown formatting or conversational filler outside the JSON.

If you need to ask a question, use this exact structure:
{
    "status": "clarifying",
    "message": "Your clarifying question to the user here"
}

If you are recommending movies, use this exact structure:
{
    "status": "success",
    "movies": [
        {
            "title": "Movie Title",
            "director": "Director Name",
            "reason": "A 1-sentence explanation of why it fits."
        }
    ]
}
"""

# =====================================================================
# AI CLIENT SETUP
# =====================================================================

_client: genai.Client | None = None

def get_genai_client() -> genai.Client:
    """Initialize and return a singleton Gemini API client."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=key)
    return _client


def create_movie_chat() -> Any:
    """Instantiate a new chat session with the configured system instructions."""
    return get_genai_client().chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
        ),
    )

# =====================================================================
# UTILITIES
# =====================================================================

def _strip_code_fence(text: str) -> str:
    """Remove markdown formatting (e.g., ```json ... ```) from the LLM output."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline != -1 and last_fence != -1 and last_fence > first_newline:
            text = text[first_newline + 1:last_fence].strip()
    return text

# =====================================================================
# CORE ORCHESTRATION
# =====================================================================

def movie_assistant_turn(chat_session: Any, user_message: str, user_id: str) -> dict:
    """
    Orchestrate the Two-Stage RAG pipeline.
    1. Retrieve candidates from Supabase.
    2. Prompt Gemini to rank and format the recommendations.
    """
    print("🔍 Fetching candidates from Supabase Vector DB...")
    
    # Stage 1: Retrieve compressed candidates (TOON format)
    toon_candidates = get_movie_candidates(user_message, user_id)
    
    rag_prompt = f"""
    User Query: {user_message}
    
    CANDIDATE MOVIES (Format: tmdb_id|Title|Genres):
    {toon_candidates}
    
    TASK:
    You are an expert movie recommender. Analyze the User Query and pick the 5 best matches 
    ONLY from the provided CANDIDATE MOVIES list. 
    
    Return your response using the strictly mandated JSON schema.
    """
    
    print("🤖 Sending candidates to Gemini for final ranking...")
    
    # Stage 2: Generate and parse the AI response
    try:
        response = chat_session.send_message(rag_prompt)
        clean_json = _strip_code_fence(response.text)
        return json.loads(clean_json)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "AI failed to return valid JSON.",
            "raw_response": response.text if response else ""
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }

# =====================================================================
# CLI TESTING MODE
# =====================================================================

def start_movie_assistant() -> None:
    """Run the AI directly in the terminal for local testing."""
    chat = create_movie_chat()
    
    # Mock user_id for terminal testing purposes
    test_user_id = "00000000-0000-0000-0000-000000000000"

    print("🎬 Movie Recommender AI (RAG Mode) started. Type 'quit' to exit.")
    user_input = input("You: ")

    while user_input.lower() not in ["quit", "exit"]:
        result = movie_assistant_turn(chat, user_input, test_user_id)
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        if result.get("status") == "clarifying":
            user_input = input("You: ")
        elif result.get("status") == "success":
            print("--- Ready for another prompt ---")
            chat = create_movie_chat()
            user_input = input("You: ")
        else:
            break

if __name__ == "__main__":
    start_movie_assistant()