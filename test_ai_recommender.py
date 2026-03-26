import os
import pytest
from dotenv import load_dotenv
from google import genai
from fastapi.testclient import TestClient
from unittest.mock import patch

from app import create_movie_chat, movie_assistant_turn
from main import app as fastapi_app

# =====================================================================
# SETUP & CONFIGURATION
# =====================================================================

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai_client = genai.Client(api_key=api_key)
api_client = TestClient(fastapi_app)

_CLARIFY_REPLY = (
    "I am flexible—pick specific films you are confident about. "
    "Return recommendations with exact titles and director names."
)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def chat_until_success(initial_message: str, max_turns: int = 8) -> dict:
    """
    Simulate a multi-turn conversation, automatically supplying a generic 
    response to clarification requests until a 'success' state is reached.
    """
    chat = create_movie_chat()
    message = initial_message
    
    for _ in range(max_turns):
        result = movie_assistant_turn(chat, message)
        status = result.get("status")
        
        if status == "success":
            return result
        if status == "clarifying":
            message = _CLARIFY_REPLY
            continue
        if status == "error":
            pytest.fail(f"LLM validation error (JSON): {result!r}")
        if status == "invalid":
            pytest.fail(f"LLM validation error (Schema): {result!r}")
            
        pytest.fail(f"Unexpected status code {status!r}: {result!r}")
        
    pytest.fail(f"Failed to reach success state after {max_turns} turns.")

# =====================================================================
# INFRASTRUCTURE & SECURITY TESTS
# =====================================================================

def test_api_health():
    """Verify the health endpoint responds with 200 OK for uptime monitoring."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_rate_limiter_blocks_after_20_requests():
    """
    Verify the IP-based rate limiter enforces the 20 requests/minute quota.
    Utilizes a spoofed IP to prevent test-state bleed.
    """
    spoofed_client = TestClient(fastapi_app, client=("10.0.0.99", 50000))
    payload = {"message": "Rate limit test message."}

    # Mock the LLM to execute the test without consuming live API quotas
    with patch("main.movie_assistant_turn") as mock_ai:
        mock_ai.return_value = {"status": "clarifying", "message": "Mocked response"}

        # Execute 20 authorized requests
        for _ in range(20):
            response = spoofed_client.post("/chat", json=payload)
            assert response.status_code == 200

        # Verify the 21st request is explicitly blocked
        blocked_response = spoofed_client.post("/chat", json=payload)
        assert blocked_response.status_code == 429
        
        error_data = blocked_response.json()
        assert "Rate limit exceeded" in str(error_data)

# =====================================================================
# API INTEGRATION & STATE MANAGEMENT TESTS
# =====================================================================

def test_api_chat_endpoint():
    """Verify the /chat endpoint successfully processes inputs and yields valid session data."""
    response = api_client.post("/chat", json={"message": "A sci-fi action movie set in space."})
    assert response.status_code == 200
    
    data = response.json()
    assert "session_id" in data
    assert data["result"]["status"] in ["success", "clarifying"]

def test_api_session_memory():
    """Verify the backend successfully maintains conversational context across requests."""
    # Establish context
    res1 = api_client.post("/chat", json={"message": "I only like movies directed by Christopher Nolan."})
    assert res1.status_code == 200
    session_id = res1.json().get("session_id")
    
    # Follow-up request utilizing pronoun resolution
    res2 = api_client.post("/chat", json={"session_id": session_id, "message": "Recommend me his Batman movie."})
    assert res2.status_code == 200
    
    result2 = res2.json().get("result", {})
    if result2.get("status") == "success":
        movies = result2.get("movies", [])
        title_found = any("Dark Knight" in m["title"] or "Batman Begins" in m["title"] for m in movies)
        assert title_found, "AI failed to retain context of the director."

def test_api_streaming_link_integration():
    """Verify the BFF orchestrator successfully injects TMDB streaming links into the AI payload."""
    payload = {"message": "I want to watch the 1999 sci-fi movie The Matrix directed by the Wachowskis."}
    response = api_client.post("/chat", json=payload)
    
    assert response.status_code == 200
    result = response.json().get("result", {})
    
    if result.get("status") == "success":
        movies = result.get("movies", [])
        assert len(movies) > 0
        for movie in movies:
            assert "streamingLink" in movie, "TMDB Orchestration failed."

# =====================================================================
# AI QUALITY GATES & VALIDATION TESTS
# =====================================================================

def test_strict_json_schema():
    """Verify the AI output strictly adheres to the mandated JSON schema."""
    result = chat_until_success("A philosophical story about an amnesiac detective solving a crime.")

    assert result.get("status") == "success"
    movies = result.get("movies")
    
    assert isinstance(movies, list)
    assert 1 <= len(movies) <= 6

    for m in movies:
        assert isinstance(m, dict)
        for key in ("title", "director", "reason"):
            assert key in m and m[key], f"Missing or empty key: {key}"

def test_hallucination_llm_judge():
    """Implement an 'LLM-as-a-Judge' to cross-reference and detect AI hallucinations."""
    result = chat_until_success("A contemplative movie about a Tokyo toilet cleaner's daily routine.")
    
    first_movie = result["movies"][0]
    movie_title = first_movie.get("title")
    director = first_movie.get("director")

    judge_prompt = f"""
    Fact check this statement: Did the director '{director}' direct a real movie called '{movie_title}'?
    Respond ONLY with the word 'YES' if it is a real movie by that director, or 'NO' if it is made up.
    """

    judge_response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
    ).text.strip().upper()

    assert "YES" in judge_response, f"Hallucination Detected: '{movie_title}' by '{director}'."

def test_fallback_circuit_breaker():
    """Verify the circuit breaker forces a recommendation after maximum clarification attempts."""
    chat = create_movie_chat()
    
    res1 = movie_assistant_turn(chat, "Recommend me a movie")
    if res1.get("status") == "success": return
    
    res2 = movie_assistant_turn(chat, "I don't know, surprise me")
    if res2.get("status") == "success": return
    
    res3 = movie_assistant_turn(chat, "anything is fine")
    assert res3.get("status") == "success", "Circuit breaker failed to trigger."
    assert len(res3.get("movies", [])) >= 1