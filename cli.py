from dotenv import load_dotenv
from app import create_movie_chat, movie_assistant_turn

# Load your API keys from .env
load_dotenv()

def run_terminal_chat():
    print("🎬 Welcome to the Gemini Movie CLI!")
    print("Type 'exit' to quit.\n")
    
    # Initialize the memory
    chat = create_movie_chat()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        print("\n🤖 Gemini is thinking...")
        response = movie_assistant_turn(chat, user_input)
        
        status = response.get("status")
        
        if status == "clarifying":
            print(f"\nAI: {response.get('message')}\n")
            
        elif status == "success":
            print("\n🍿 HERE ARE YOUR MOVIES:")
            for m in response.get("movies", []):
                print(f"- {m['title']} (Directed by {m['director']})")
                print(f"  Why: {m['reason']}\n")
            
            # Reset chat for a new recommendation
            print("--- Ready for another prompt ---")
            chat = create_movie_chat()
            
        else:
            print(f"\n❌ Error: {response}\n")

if __name__ == "__main__":
    run_terminal_chat()