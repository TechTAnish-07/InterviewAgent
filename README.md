# 🤖 AI Voice Interview Agent & Resume Microservice

A standalone Python microservice and full-duplex conversational voice agent powering AI-driven technical job interviews.

---

## 🚀 Overview & Key Features

1. **LiveKit AI Voice Agent Worker (`agent.py`)** — Real-time voice interviewer with low-latency VAD, STT, LLM reasoning, Function Tool calling, and TTS.
2. **FastAPI Microservice (`app/`)** — Resume normalization (PDF text reconstruction), pre-flight job relevance screening, and explicit agent dispatching via LiveKit API.
3. **Bounded Working Memory (`memory.py`)** — Maintains a sliding window of recent turn pairs and folds older turns into a rolling summary to manage context size efficiently.
4. **Safety & Moderation (`moderation.py`)** — Fast local heuristics and LLM fallback to detect off-topic or inappropriate responses, with soft warning escalation and policy termination.
5. **Automated Evaluation Reporting** — Generates objective post-interview candidate feedback reports and posts them to the Spring Boot backend (`POST /api/ai-interview/{sessionId}/feedback`).

---

## 🔄 Agent Execution Flow

The voice agent executes a full-duplex conversational loop for every interview session:

```mermaid
flowchart TD
    A[Room Connect / Agent Dispatch] --> B[Fetch Context from Spring Boot Backend]
    B --> C[Initialize VAD + STT + TTS + Bounded Memory]
    C --> D{Candidate Silent 2.5s?}
    D -- Yes --> E[Speak Initial Warm Greeting]
    D -- No --> F[Candidate Speaks]
    E --> F
    
    F --> G[Silero VAD + Groq Whisper STT]
    G --> H[Moderation Check: Local Heuristics / LLM]
    
    H -- Off-Topic / Inappropriate --> I{Warning Count}
    I -- Count < 3 --> J[Speak Warning Reply & Log]
    I -- Count >= 3 --> K[Policy Violation -> End Interview]
    
    H -- Normal --> L[Build Context: Summary + Sliding Window]
    L --> M[LiteLLM / Gemini 2.5 Flash Call with Tools]
    
    M -- Tool: end_interview --> N[Execute end_interview_flow]
    M -- Tool: get_resume_context --> O[Fetch & Append Resume Context] --> M
    M -- Normal Reply --> P[Fish Audio / LiveKit Inference TTS]
    
    P --> Q[Fold Turn into Bounded Working Memory]
    Q --> F
    
    N --> R[Generate Post-Interview Feedback Report]
    K --> R
    R --> S[POST Feedback & Session End to Spring Boot]
    S --> T[Disconnect Room & End Session]
```

---

## 🏗️ System Architecture & Code Structure

```mermaid
graph TD
    Client[Web Frontend / Candidate] <-->|WebRTC Voice| LiveKit[LiveKit Server]
    LiveKit <-->|Agent Dispatch & Worker API| AgentWorker[agent.py - LiveKit Worker]
    
    Spring[Spring Boot Backend] <-->|REST API Context & Feedback| AgentWorker
    Spring <-->|REST Dispatch & Resume API| FastAPI[app/main.py - FastAPI Server]
    
    subgraph Agent Core Components
        AgentWorker --> Memory[memory.py - Bounded Working Memory]
        AgentWorker --> Moderation[moderation.py - Safety Classifier]
        AgentWorker --> Prompts[prompts.py - System Prompts]
        AgentWorker --> Tools[tools.py - Function Tools]
    end
    
    subgraph External LLM & Voice Services
        AgentWorker --> VAD[Silero VAD]
        AgentWorker --> STT[Groq Whisper STT]
        AgentWorker --> LLM[LiteLLM / Gemini 2.5 Flash]
        AgentWorker --> TTS[Fish Audio TTS]
    end
```

### 📁 Directory Layout

```
InterviewAgent/
├── app/
│   ├── controllers/          # FastAPI Router Endpoints
│   │   ├── health_controller.py      # Health Check Endpoint
│   │   ├── resume_controller.py      # Resume Normalization & Relevance Endpoints
│   │   └── dispatch_controller.py    # LiveKit Agent Dispatch Endpoint
│   ├── schemas/              # Pydantic Request & Response Schemas
│   │   └── resume_schema.py
│   ├── services/             # Resume Normalization & Relevance Service
│   │   └── resume_service.py
│   ├── config.py             # App Configuration & Environment Bindings
│   └── main.py               # FastAPI Microservice Entrypoint
├── agent.py                  # LiveKit Voice Agent Worker & Session Lifecycle
├── memory.py                 # Bounded Working Memory & Direct Turn Folding
├── moderation.py             # Local Heuristic & LLM Content Moderation
├── prompts.py                # System Prompts & Feedback Generation Templates
├── tools.py                  # LiveKit Function Tool Definitions (@llm.function_tool)
├── test_main.py              # Pytest Test Suite
├── requirements.txt          # Python Dependencies
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
Copy `.env.example` to `.env` and populate your credentials:
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
INTERNAL_API_KEY=internal-secret-key
```

---

### 4. Run Services

#### FastAPI Microservice (Port 8000)
```bash
uvicorn app.main:app --reload --port 8000
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

## 🔗 API Endpoints Summary

- **GET `/health`**: Healthcheck (`{"status": "ok"}`)
- **POST `/resume/normalize`**: Reconstruct scrambled PDF text into structured Markdown
- **POST `/resume/check-relevance`**: Check candidate resume fit against target job title
- **POST `/dispatch-agent`**: Dispatch agent worker to a LiveKit room
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
