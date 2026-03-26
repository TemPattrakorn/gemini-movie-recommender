import asyncio
import os
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import create_movie_chat, movie_assistant_turn

load_dotenv()

# Initialize IP-based rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Gemini Movie Recommender API")

# Register rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
_cors = os.environ.get("CORS_ORIGINS", FRONTEND_URL)
_origins = [o.strip() for o in _cors.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, Any] = {}
_session_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    result: dict[str, Any]


async def fetch_streaming_link(title: str) -> str | None:
    """Fetch JustWatch links concurrently via TMDB API."""
    tmdb_key = os.environ.get("TMDB_API_KEY")
    if not tmdb_key:
        return None

    # Async HTTP client
    async with httpx.AsyncClient() as client:
        try:
            # Search movie ID
            search_url = "https://api.themoviedb.org/3/search/movie"
            search_res = await client.get(search_url, params={"query": title, "api_key": tmdb_key})
            search_res.raise_for_status()
            search_data = search_res.json()

            if not search_data.get("results"):
                return None
            movie_id = search_data["results"][0]["id"]

            # Fetch watch providers
            providers_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
            providers_res = await client.get(providers_url, params={"api_key": tmdb_key})
            providers_res.raise_for_status()
            providers_data = providers_res.json()

            results = providers_data.get("results", {})
            
            # Prioritize TH region, fallback to US
            region_data = results.get("TH") or results.get("US")

            if region_data and "link" in region_data:
                return region_data["link"]
            return None
        except Exception as e:
            print(f"Error fetching TMDB data for {title}: {e}")
            return None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute") # Limit: 20 requests/minute per IP
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    async with _session_lock:
        sid = req.session_id
        if sid and sid in _sessions:
            chat_session = _sessions[sid]
        else:
            try:
                chat_session = create_movie_chat()
            except ValueError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            sid = str(uuid.uuid4())
            _sessions[sid] = chat_session

    try:
        result = await asyncio.to_thread(movie_assistant_turn, chat_session, req.message)
    except Exception as e:
        async with _session_lock:
            _sessions.pop(sid, None)
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e!s}") from e

    if result.get("status") == "success":
        async with _session_lock:
            _sessions.pop(sid, None)
            
        # TMDB Orchestration
        movies = result.get("movies", [])
        
        # Fetch streaming links concurrently
        tasks = [fetch_streaming_link(movie["title"]) for movie in movies]
        links = await asyncio.gather(*tasks)
        
        # Attach links to movie results
        for movie, link in zip(movies, links):
            movie["streamingLink"] = link

    return ChatResponse(session_id=sid, result=result)