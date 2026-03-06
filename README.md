# AI QA Evaluation Suite: Gemini Movie Recommender

A proof-of-concept project demonstrating how to build, constrain, and rigorously test a Generative AI application. This project showcases automated Quality Assurance techniques specifically tailored for non-deterministic LLM outputs.

## 🎯 Objective
Traditional software testing asserts exact expected outcomes. AI testing requires evaluating output *quality* and *safety*. This project builds a Gemini-powered movie recommendation engine and an automated `pytest` framework to evaluate its responses for **Schema Compliance** and **Data Hallucination**.

## 🛠️ Tech Stack
* **Language:** Python
* **AI Model:** Google Gemini API (`gemini-2.5-flash`)
* **Testing Framework:** `pytest`
* **Architecture:** LLM-as-a-Judge

## 🧪 The QA Framework

The test suite (`test_ai_recommender.py`) implements two critical AI evaluation strategies:

### 1. Format & Schema Compliance
LLMs are prone to generating conversational filler (e.g., "Sure, here is your movie..."). 
* **The Test:** Verifies that the model strictly adheres to system instructions by outputting highly structured, parseable JSON data without markdown formatting or conversational text.
* **Why it matters:** Ensures the AI can be safely integrated into traditional frontend/backend architectures without breaking data parsers.

### 2. Hallucination Detection (LLM-as-a-Judge)
LLMs can confidently invent fake information (hallucinations). 
* **The Test:** Extracts the generated movie title and director, then dynamically spins up a *second* LLM prompt (the "Judge"). The Judge fact-checks the first AI's output against its own training data to verify that the recommended movie actually exists in the real world.
* **Why it matters:** Automates factual accuracy testing without needing to write thousands of manual assertions or maintain a static database of all movies.

## 🚀 Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/gemini-movie-recommender.git
   cd gemini-movie-recommender
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install google-genai pytest python-dotenv
   ```

3. **Set up your environment variables**

   - Create a `.env` file in the project root.
   - Add your Google Gemini API key:

     ```bash
     GEMINI_API_KEY="your_api_key_here"
     ```

## 🏃‍♂️ Running the Tests

Execute the automated QA evaluation suite using `pytest`:

```bash
pytest test_ai_recommender.py -v
```

## 🔮 Future Enhancements

- CI/CD Integration: Integrate this test suite into a Jenkins pipeline to automatically run regression testing on the prompt whenever the `app.py` system instructions are updated.
- UI Test Automation: Build a simple web interface and implement Playwright for end-to-end UI testing alongside the backend LLM evaluation.