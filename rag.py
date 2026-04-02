import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import Client, create_client

load_dotenv()

# =====================================================================
# CLIENT SETUP & INITIALIZATION
# =====================================================================

_supabase_client: Client | None = None
_genai_client: genai.Client | None = None


def get_supabase_client() -> Client:
    """Initialize and return a singleton Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")  # Use service_role key for backend operations
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_genai_client() -> genai.Client:
    """Initialize and return a singleton Gemini API client."""
    global _genai_client
    if _genai_client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _genai_client = genai.Client(api_key=key)
    return _genai_client

# =====================================================================
# CORE RETRIEVAL LOGIC (RAG)
# =====================================================================

def get_movie_candidates(user_query: str, user_id: str) -> str:
    """
    Takes the user's prompt, finds 50 unseen matches in Supabase via vector search, 
    and returns them in the highly-compressed TOON format.
    """
    supabase = get_supabase_client()
    genai_client = get_genai_client()
    
    # 1. Fetch the user's "Watched" list so we can exclude them
    watched_res = supabase.table("user_movies").select("tmdb_id").eq("user_id", user_id).eq("status", "watched").execute()
    exclude_ids = [row["tmdb_id"] for row in watched_res.data]
    
    # 2. Convert the user's English query into a 768-dimensional vector
    embedding_response = genai_client.models.embed_content(
        model='gemini-embedding-001', # <-- Updated Model Name
        contents=user_query,
        config=types.EmbedContentConfig(output_dimensionality=768) # <-- Compress to 768!
    )
    query_vector = embedding_response.embeddings[0].values
    
    # 3. Call your Supabase SQL function
    rpc_response = supabase.rpc(
        "match_movies",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.3,      # Adjust this to be more or less strict
            "match_count": 50,           # Give Gemini 50 options to pick from
            "exclude_tmdb_ids": exclude_ids
        }
    ).execute()
    
    candidates = rpc_response.data
    
    if not candidates:
        return "No matches found in database."

    # 4. Compress the results into TOON (Token-Oriented Notation) format
    # Example output: "155|The Dark Knight|Action,Crime"
    toon_lines = []
    for c in candidates:
        # Strip out newlines or pipes from descriptions to prevent formatting breaks
        clean_title = str(c.get("title", "")).replace("|", "")
        clean_genres = str(c.get("genres", "")).replace("|", "")
        toon_lines.append(f"{c['tmdb_id']}|{clean_title}|{clean_genres}")
        
    return "\n".join(toon_lines)