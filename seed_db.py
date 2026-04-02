import os
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

load_dotenv()

# =====================================================================
# INITIALIZATION
# =====================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Use Service Role Key!
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, TMDB_API_KEY]):
    raise ValueError("Missing required environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================================
# TMDB FETCHING LOGIC
# =====================================================================

def get_tmdb_genres() -> dict[int, str]:
    """Fetch the master list of TMDB genres to map IDs to text."""
    print("📡 Fetching genre map from TMDB...")
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}"
    response = httpx.get(url)
    response.raise_for_status()
    genres = response.json().get("genres", [])
    return {g["id"]: g["name"] for g in genres}


def get_popular_movies(pages: int = 5) -> list[dict]:
    """Fetch top movies from TMDB across multiple pages."""
    print(f"📡 Fetching {pages * 20} popular movies from TMDB...")
    movies = []
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}"
        response = httpx.get(url)
        response.raise_for_status()
        movies.extend(response.json().get("results", []))
    return movies

# =====================================================================
# EMBEDDING & DATABASE LOGIC
# =====================================================================

def process_and_seed_movies():
    """Main pipeline: Download -> Map Genres -> Embed -> Supabase"""
    genre_map = get_tmdb_genres()
    raw_movies = get_popular_movies(pages=5)  # 5 pages = 100 movies
    
    formatted_records = []
    
    print("🧠 Generating AI embeddings for movie plots...")
    for movie in raw_movies:
        # 1. Map TMDB's integer genre IDs to readable text strings
        genre_names = [genre_map.get(gid, "Unknown") for gid in movie.get("genre_ids", [])]
        genres_str = ", ".join(genre_names)
        
        title = movie.get("title", "")
        overview = movie.get("overview", "")
        tmdb_id = movie.get("id")
        
        if not overview or not title:
            continue # Skip movies with missing data
            
        # 2. Create the "Rich Text" block for the AI to embed
        # We combine title, genres, and overview so the vector captures everything
        content_to_embed = f"Title: {title}\nGenres: {genres_str}\nPlot: {overview}"
        
        # 3. Call Gemini to do the vector math
        embedding_res = genai_client.models.embed_content(
            model='gemini-embedding-001', # <-- Updated Model Name
            contents=content_to_embed,
            config=types.EmbedContentConfig(output_dimensionality=768) # <-- Compress to 768!
        )
        vector = embedding_res.embeddings[0].values
        
        # 4. Prepare the row for Supabase
        formatted_records.append({
            "tmdb_id": tmdb_id,
            "title": title,
            "genres": genres_str,
            "description": overview,
            "embedding": vector
        })
        
    print(f"💾 Pushing {len(formatted_records)} movies to Supabase Vector DB...")
    
    # 5. Upsert into Supabase (Upsert prevents crashes if a movie already exists)
    # We chunk it into batches of 50 to avoid hitting payload size limits
    batch_size = 50
    for i in range(0, len(formatted_records), batch_size):
        batch = formatted_records[i:i + batch_size]
        supabase.table("movies").upsert(batch).execute()
        print(f"   -> Inserted batch {i // batch_size + 1}")

    print("✅ Database seeding complete!")

if __name__ == "__main__":
    process_and_seed_movies()