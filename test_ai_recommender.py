import json
import pytest
import os
from dotenv import load_dotenv
from google import genai
from app import get_movie_recommendation

# 1. Setup the environment for the test suite
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=api_key)

def test_strict_json_schema():
    """
    Test 1: Format Compliance
    Verifies the LLM strictly adheres to the requested JSON schema and does not return conversational text.
    """
    # A complex query to test if the model gets confused
    query = "A philosophical story about an amnesiac detective solving a crime."
    
    result_str = get_movie_recommendation(query)
    
    # Assert 1: The output must be perfectly parseable JSON
    try:
        result_json = json.loads(result_str)
    except json.JSONDecodeError:
        pytest.fail(f"Test Failed: LLM output was not valid JSON. Output was: {result_str}")
        
    # Assert 2: The JSON must contain exactly these three keys
    assert "title" in result_json, "Missing 'title' key in LLM output"
    assert "director" in result_json, "Missing 'director' key in LLM output"
    assert "reason" in result_json, "Missing 'reason' key in LLM output"

def test_hallucination_llm_judge():
    """
    Test 2: LLM-as-a-Judge (Hallucination Check)
    Dynamically verifies if the movie recommended by the AI actually exists in the real world.
    """
    query = "A contemplative movie about a Tokyo toilet cleaner's daily routine."
    
    # 1. Get the target output
    result_str = get_movie_recommendation(query)
    result_json = json.loads(result_str)
    
    movie_title = result_json.get("title")
    director = result_json.get("director")
    
    # 2. Prompt the Judge to verify the facts
    judge_prompt = f"""
    Fact check this statement: Did the director '{director}' direct a real movie called '{movie_title}'?
    Respond ONLY with the word 'YES' if it is a real movie by that director, or 'NO' if it is made up or incorrect.
    """

    judge_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
    ).text.strip().upper()
    
    # Assert: The judge must confirm the movie exists
    assert "YES" in judge_response, f"Hallucination Detected! The AI made up the movie '{movie_title}' by '{director}'."