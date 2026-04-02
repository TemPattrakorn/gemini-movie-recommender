import json
from dotenv import load_dotenv

from app import create_movie_chat, movie_assistant_turn

# =====================================================================
# INITIALIZATION
# =====================================================================

# Load environment variables (API keys, Supabase URLs, etc.)
load_dotenv()

# =====================================================================
# CLI ORCHESTRATION
# =====================================================================

def run_terminal_chat() -> None:
    """Run the interactive terminal chat session."""
    chat = create_movie_chat()
    
    # Create a dummy guest ID for local terminal testing
    test_user_id = "00000000-0000-0000-0000-000000000000"

    print("🎬 Welcome to the Gemini Movie CLI (RAG Mode)!")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C gracefully
            print("\nGoodbye! 🍿")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! 🍿")
            break

        print("\n🤖 Gemini is searching your RAG database...")
        
        try:
            # Pass the test_user_id into the core orchestration function
            response = movie_assistant_turn(chat, user_input, test_user_id)
        except Exception as e:
            print(f"\n❌ System Error: {e}\n")
            continue
        
        status = response.get("status")
        
        if status == "clarifying":
            print(f"\nAI: {response.get('message')}\n")
            
        elif status == "success":
            print("\n🍿 HERE ARE YOUR MOVIES:")
            for m in response.get("movies", []):
                title = m.get("title", "Unknown Title")
                director = m.get("director", "Unknown Director")
                reason = m.get("reason", "No reason provided.")
                
                print(f"- {title} (Directed by {director})")
                print(f"  Why: {reason}\n")
            
            # Reset chat for a new recommendation
            print("--- Ready for another prompt ---")
            chat = create_movie_chat()
            
        else:
            # Catch-all for unexpected JSON structures
            print("\n❌ Error or Unexpected Response:")
            print(json.dumps(response, indent=2))
            print()

if __name__ == "__main__":
    run_terminal_chat()