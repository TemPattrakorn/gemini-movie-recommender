import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

def get_movie_recommendation(user_preference: str) -> Dict[str, Any]:
    prompt = f"""
    You are a film expert. Based on the following user preference, recommend one movie.
    User preference: {user_preference}

    You must respond ONLY with a valid JSON object using this exact structure:
    {{
        "title": "Movie Title",
        "director": "Director Name",
        "reason": "A 1-sentence explanation of why it fits the preference."
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw = response.text.strip()

    # Handle responses wrapped in markdown code fences like ```json ... ```
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_newline != -1 and last_fence != -1 and last_fence > first_newline:
            raw = raw[first_newline + 1:last_fence].strip()

    return json.loads(raw)

print("Fetching recommendation...")

test_query = "A thai movie about a women who try to clear her house for a renovation into home office"
result = get_movie_recommendation(test_query)

print("\nAPI Response:")
print(json.dumps(result, indent=2))