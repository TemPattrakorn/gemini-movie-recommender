import os

import pytest
from dotenv import load_dotenv
from google import genai

from app import create_movie_chat, movie_assistant_turn

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=api_key)

_CLARIFY_REPLY = (
    "I am flexible—pick specific films you are confident about. "
    "Return recommendations with exact titles and director names."
)


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

    judge_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
    ).text.strip().upper()

    assert "YES" in judge_response, (
        f"Hallucination Detected! The AI made up the movie '{movie_title}' by '{director}'."
    )
