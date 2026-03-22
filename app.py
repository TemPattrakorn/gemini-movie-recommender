import os
import json
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=key)
    return _client


def create_movie_chat() -> Any:
    return get_genai_client().chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        ),
    )

system_instruction = """
You are a film expert recommending movies. 
If the user's prompt is too vague (e.g., just a country, just a broad genre), ask 1 or 2 clarifying questions to narrow down their preferences (e.g., "What genre?", "Do you prefer classic or modern?").
Once you have enough context, recommend between 1 and 6 specific movies that fit the criteria perfectly. Do not recommend more than 6.

You MUST respond ONLY with a valid JSON object. Do not include markdown formatting or conversational filler outside the JSON.

If you need to ask a question, use this exact structure:
{
    "status": "clarifying",
    "message": "Your clarifying question to the user here"
}

If you have enough context to recommend movies, use this exact structure containing a list of movies:
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


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline != -1 and last_fence != -1 and last_fence > first_newline:
            text = text[first_newline + 1:last_fence].strip()
    return text


def movie_assistant_turn(chat: Any, user_message: str) -> dict[str, Any]:
    """
    One user message → one JSON-serializable dict for a client (CLI or HTTP).
    status is always one of: clarifying, success, error (invalid JSON), invalid (bad shape).
    """
    response = chat.send_message(user_message)
    raw = (response.text or "").strip()
    raw = _strip_code_fence(raw)
    try:
        ai_data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "code": "invalid_json",
            "message": "AI did not return valid JSON.",
            "raw": raw,
        }

    status = ai_data.get("status")
    if status in ("clarifying", "success"):
        return ai_data

    return {
        "status": "invalid",
        "message": "Model response did not match expected schema.",
        "payload": ai_data,
    }


def start_movie_assistant() -> None:
    chat = create_movie_chat()

    print("🎬 Movie Recommender AI started. Type 'quit' to exit.")
    user_input = input("You: ")

    while user_input.lower() != "quit":
        result = movie_assistant_turn(chat, user_input)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get("status") == "clarifying":
            user_input = input("\nYou: ")
        elif result.get("status") == "success":
            break
        else:
            break


if __name__ == "__main__":
    start_movie_assistant()
