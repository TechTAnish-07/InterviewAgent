# 🤖 AI Voice Interview Agent & Resume Service

Standalone Python service containing:
1. **LiveKit AI Voice Agent Worker (`agent.py`)** — Full-duplex conversational interviewer.
2. **FastAPI Microservice (`app/`)** — Resume normalization, pre-flight relevance screening, and agent dispatch.

---

## 🚀 Features & Architecture

### 🎙️ LiveKit AI Voice Agent (`agent.py`)
- **Full-Duplex Speech Pipeline**:
  - **VAD (Voice Activity Detection)**: Local Silero VAD for low-latency speech detection.
  - **STT (Speech-to-Text)**: Groq hosted Whisper (`whisper-large-v3`).
  - **LLM**: Gemini / OpenAI via LiteLLM (`gemini-2.5-flash`).
  - **TTS (Text-to-Speech)**: Fish Audio model (`fishaudio/s2.1-pro-free`).
- **Bounded Working Memory (`memory.py`)**:
  - Maintains a sliding window of recent conversation turns and automatically folds older exchanges into a rolling summary.
- **Moderation & Warning Escalation (`moderation.py`)**:
  - Screens candidate turns for off-topic or inappropriate content.
  - Escalates through soft warnings before terminating policy-violating sessions.
- **Automated Feedback Generation**:
  - Generates comprehensive post-interview evaluation reports and posts them to the Spring Boot backend (`POST /api/ai-interview/{sessionId}/feedback`).

### ⚡ FastAPI Microservice (`app/`)
- **`POST /resume/normalize`**: Reconstructs scrambled raw PDF resume text into clean, structured Markdown.
- **`POST /resume/check-relevance`**: Evaluates candidate resumes against target job titles before launching an interview session.
- **`POST /dispatch-agent`**: Dispatches the `interview-agent` worker to LiveKit rooms using the LiveKit Admin SDK.

---

## 📁 Project Structure

```
InterviewAgent/
├── app/
│   ├── controllers/          # Endpoint Controllers / Routers
│   │   ├── health_controller.py
│   │   ├── resume_controller.py
│   │   └── dispatch_controller.py
│   ├── schemas/              # Pydantic Schemas
│   │   └── resume_schema.py
│   ├── services/             # Resume Normalization & Relevance LLM Service
│   │   └── resume_service.py
│   ├── config.py
│   └── main.py               # FastAPI App Entrypoint
├── agent.py                  # LiveKit Voice Agent Worker
├── memory.py                 # Bounded Conversation Memory & Summarization
├── moderation.py             # Off-topic & Inappropriate Input Moderation
├── prompts.py                # System Prompts & Feedback Generation Prompts
├── tools.py                  # Agent Tool Call Definitions
├── requirements.txt          # Dependencies
├── test_main.py              # Pytest Unit Test Suite
└── README.md
```

---

## ⚙️ Setup & Running

### 1. Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gemini/gemini-2.5-flash
PORT=8000
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
GROQ_API_KEY=your_groq_api_key
SPRING_BASE_URL=http://localhost:8080
INTERNAL_SERVICE_API_KEY=internal-secret-key
```

---

### 4. Run Services

#### FastAPI Service (Port 8000)
```bash
uvicorn main:app --reload --port 8000
```

#### LiveKit Voice Agent Worker
```bash
python agent.py dev
```

---

## 🧪 Running Unit Tests

```bash
pytest
```

---

## 🔗 API Endpoints

- **GET `/health`**: Healthcheck (`{"status": "ok"}`)
- **POST `/resume/normalize`**: Reconstruct raw text into clean markdown `{ "rawText": "..." }`
- **POST `/resume/check-relevance`**: Assess resume fit against job title `{ "resumeText": "...", "jobTitle": "..." }`
- **POST `/dispatch-agent`**: Trigger worker dispatch `{ "room": "...", "session_id": 123 }`
- **Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

