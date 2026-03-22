import asyncio
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import create_movie_chat, movie_assistant_turn

load_dotenv()

app = FastAPI(title="Gemini Movie Recommender API")

_cors = os.environ.get("CORS_ORIGINS", "*")
_origins = [o.strip() for o in _cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins else ["*"],
    allow_credentials=False,
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    async with _session_lock:
        sid = req.session_id
        if sid and sid in _sessions:
            chat = _sessions[sid]
        else:
            try:
                chat = create_movie_chat()
            except ValueError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            sid = str(uuid.uuid4())
            _sessions[sid] = chat

    try:
        result = await asyncio.to_thread(movie_assistant_turn, chat, req.message)
    except Exception as e:
        async with _session_lock:
            _sessions.pop(sid, None)
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e!s}") from e

    if result.get("status") == "success":
        async with _session_lock:
            _sessions.pop(sid, None)

    return ChatResponse(session_id=sid, result=result)
