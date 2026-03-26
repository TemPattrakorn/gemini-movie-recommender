# 🎬 Gemini Movie Recommender — Backend API

A robust, secure, and fully-tested FastAPI backend powering an AI-driven movie recommendation engine. It uses **Gemini 2.5 Flash** for multi-turn conversational AI and integrates with the **TMDB API** to surface real-time streaming availability links.

This service acts as the **Backend-for-Frontend (BFF)** in a decoupled web architecture, built with a strong emphasis on API security, LLM output validation, and automated quality gates.

---

## 🏗 Architecture & Security

This API operates on a **Zero-Trust Server-to-Server Proxy Architecture**. Multiple layers of defense protect against public abuse, quota exhaustion, and malicious payloads.

| Layer | Mechanism | Effect |
|---|---|---|
| **Auth** | `x-api-key` header validation | Blocks direct browser/Postman access with `403 Forbidden` |
| **Rate Limiting** | `slowapi` — 20 req/min per IP | Prevents DoS attacks |
| **Payload Size** | Pydantic — max 1,000 chars | Rejects bloated prompts with `422 Unprocessable Entity` |
| **CORS** | Env-based origin allowlist | Prevents CSRF |
| **Circuit Breaker** | Clarification turn limit | Forces a recommendation to stop infinite loops |

---

## 🚀 Core Features

- **Conversational Memory** — Async locks and UUID session IDs maintain conversation context across multiple HTTP requests.
- **Structured LLM Output** — Forces Gemini to adhere to a strict JSON schema (title, director, reason) for guaranteed, parsable responses.
- **TMDB Orchestration** — Intercepts AI recommendations and concurrently fetches real-time JustWatch streaming links before returning the payload.
- **CLI Testing Mode** — A terminal wrapper (`cli.py`) enables isolated LLM debugging without starting the web server.

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI / Python 3.10+ |
| AI Provider | Google Gemini (`gemini-2.5-flash`) via `google-genai` SDK |
| External APIs | TMDB (The Movie Database) |
| Testing | Pytest, Pytest-HTML, `unittest.mock` |
| CI/CD | GitHub Actions |

---

## 💻 Local Development

### 1. Clone & Install

```bash
git clone https://github.com/your-username/gemini-movie-recommender.git
cd gemini-movie-recommender
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# AI & Data APIs
GEMINI_API_KEY=your_gemini_api_key
TMDB_API_KEY=your_tmdb_api_key

# Security & CORS
FRONTEND_URL=http://localhost:3000
API_SECRET_KEY=my-local-secret
```

### 3. Start the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Run CLI Mode (No Web Server)

Test the core AI logic directly in your terminal:

```bash
python cli.py
```

---

## 🧪 Testing & Quality Gates

The test suite bypasses security gates via spoofed network IPs and authorization headers to enable proper infrastructure validation.

```bash
pytest -v
```

**Coverage highlights:**

- **LLM-as-a-Judge** — A secondary AI prompt cross-references the primary AI's recommendations to dynamically detect hallucinations.
- **Schema Validation** — Verifies the LLM returns lists of dicts matching the Pydantic spec.
- **Context Retention** — Simulates multi-turn conversations using pronouns to verify memory across turns.
- **Infrastructure** — Validates rate limit drops (`429`), payload rejections (`422`), auth failures (`401`/`403`), and TMDB data hydration.

---

## 📡 API Reference

### `POST /chat`

Generates the next conversational step or a final movie recommendation.

**Required Headers**

```
Content-Type: application/json
x-api-key: <YOUR_API_SECRET_KEY>
```

**Request Body**

```json
{
  "message": "A fast-paced sci-fi action movie set in space.",
  "session_id": null
}
```

> Pass `session_id` from a previous response to continue an existing conversation.

**Success Response** *(final recommendation state)*

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "status": "success",
    "movies": [
      {
        "title": "Edge of Tomorrow",
        "director": "Doug Liman",
        "reason": "It offers intense, fast-paced sci-fi action with a clever time-loop mechanic.",
        "streamingLink": "https://www.themoviedb.org/movie/137113/watch"
      }
    ]
  }
}
```

---

## 🤝 Ecosystem

This backend is designed to work exclusively with the [Gemini Movie Recommender Frontend](https://github.com/TemPattrakorn/gemini-movie-frontend). Refer to that repository for frontend setup, API reference, and security documentation.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.
