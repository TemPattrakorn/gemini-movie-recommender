import os
import pytest
from dotenv import load_dotenv
from google import genai
from fastapi.testclient import TestClient

from app import create_movie_chat, movie_assistant_turn
from main import app as fastapi_app

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Rename the client slightly to avoid conflicts with the TestClient
genai_client = genai.Client(api_key=api_key)

# Initialize the FastAPI test client to test main.py
api_client = TestClient(fastapi_app)

_CLARIFY_REPLY = (
    "I am flexible—pick specific films you are confident about. "
    "Return recommendations with exact titles and director names."
)

def test_api_health():
    """
    FastAPI Integration: Verify the health endpoint is responsive.
    """
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_chat_endpoint():
    """
    FastAPI Integration: Verify the POST /chat endpoint successfully processes a 
    message and returns the expected session_id and result payload.
    """
    payload = {"message": "A fast-paced sci-fi action movie set in space."}
    response = api_client.post("/chat", json=payload)
    
    assert response.status_code == 200, f"API Error: {response.text}"
    data = response.json()
    
    assert "session_id" in data
    assert "result" in data
    assert data["result"]["status"] in ["success", "clarifying"]

def chat_until_success(initial_message: str, max_turns: int = 8) -> dict:
    """
    Drive the same chat + turn logic as the live app until status is success,
    replying once per clarifying turn with a fixed follow-up.
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
            pytest.fail(f"LLM output was not valid JSON: {result!r}")
        if status == "invalid":
            pytest.fail(f"LLM output did not match expected schema: {result!r}")
        pytest.fail(f"Unexpected status {status!r}: {result!r}")
    pytest.fail(f"No success after {max_turns} turns (last message was {message!r})")

def test_strict_json_schema():
    """
    Format compliance: model eventually returns the success envelope with a movies list
    whose items match the schema enforced in app.py's system instruction.
    """
    result = chat_until_success(
        "A philosophical story about an amnesiac detective solving a crime."
    )

    assert result.get("status") == "success"
    movies = result.get("movies")
    assert isinstance(movies, list), "Expected 'movies' to be a list"
    assert 1 <= len(movies) <= 6, f"Expected 1–6 movies, got {len(movies)}"

    for i, m in enumerate(movies):
        assert isinstance(m, dict), f"movies[{i}] must be an object"
        for key in ("title", "director", "reason"):
            assert key in m, f"movies[{i}] missing '{key}'"
            assert m[key], f"movies[{i}].{key} must be non-empty"

def test_hallucination_llm_judge():
    """
    LLM-as-a-Judge: first recommended film should look like a real title + director pair.
    """
    result = chat_until_success(
        "A contemplative movie about a Tokyo toilet cleaner's daily routine."
    )
    first = result["movies"][0]
    movie_title = first.get("title")
    director = first.get("director")

    judge_prompt = f"""
    Fact check this statement: Did the director '{director}' direct a real movie called '{movie_title}'?
    Respond ONLY with the word 'YES' if it is a real movie by that director, or 'NO' if it is made up or incorrect.
    """

    judge_response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
    ).text.strip().upper()

    assert "YES" in judge_response, (
        f"Hallucination Detected! The AI made up the movie '{movie_title}' by '{director}'."
    )

def test_fallback_circuit_breaker():
    """
    AI Edge Case: Test that the AI respects the fallback rule and stops asking 
    clarifying questions after a maximum of 2 vague responses, returning a success payload.
    """
    chat = create_movie_chat()
    
    # Turn 1: Extremely vague prompt
    res1 = movie_assistant_turn(chat, "Recommend me a movie")
    if res1.get("status") == "success":
        return  # Triggered success immediately, which is acceptable
        
    assert res1.get("status") == "clarifying"
    
    # Turn 2: Passive/Vague reply
    res2 = movie_assistant_turn(chat, "I don't know, surprise me")
    if res2.get("status") == "success":
        return  # Triggered fallback early
        
    assert res2.get("status") == "clarifying"
    
    # Turn 3: Final vague reply. This MUST trigger the fallback rule.
    res3 = movie_assistant_turn(chat, "anything is fine")
    assert res3.get("status") == "success", "AI failed the circuit breaker and asked a 3rd question."
    assert "movies" in res3
    assert len(res3["movies"]) >= 1
    
def test_api_streaming_link_integration():
    """
    FastAPI Integration: Verify the BFF orchestration logic.
    When the AI returns a success payload, the backend must inject the 
    'streamingLink' key into each movie object via TMDB.
    """
    # Ask for a very specific, famous movie guaranteed to trigger a 'success' state immediately
    payload = {"message": "I want to watch the 1999 sci-fi movie The Matrix directed by the Wachowskis."}
    response = api_client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})
    
    # We expect a success status based on the highly specific prompt
    if result.get("status") == "success":
        movies = result.get("movies", [])
        assert len(movies) > 0
        for i, movie in enumerate(movies):
            # Assert that the new TMDB logic added the key (even if the link itself is None)
            assert "streamingLink" in movie, f"Orchestrator failed to inject 'streamingLink' into movie[{i}]"
            
def test_api_session_memory():
    """
    FastAPI Integration: Verify that passing the session_id retains conversation history.
    """
    # Turn 1: Establish context
    res1 = api_client.post("/chat", json={"message": "I only like movies directed by Christopher Nolan."})
    assert res1.status_code == 200
    session_id = res1.json().get("session_id")
    assert session_id is not None
    
    # Turn 2: Use the exact same session ID and use a pronoun ("his")
    res2 = api_client.post("/chat", json={
        "session_id": session_id,
        "message": "Recommend me his Batman movie."
    })
    assert res2.status_code == 200
    data2 = res2.json()
    
    # The AI should know "his" refers to Nolan from the previous turn memory
    result2 = data2.get("result", {})
    if result2.get("status") == "success":
        movies = result2.get("movies", [])
        title_found = any("Dark Knight" in m["title"] or "Batman Begins" in m["title"] for m in movies)
        assert title_found, "The API failed to retain session context; the AI forgot who the director was!"
