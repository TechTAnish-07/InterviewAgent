# Resume Normalization Agent API

Standalone Python FastAPI service scoped to resume text normalization.

## Project Structure

```
.
├── app/
│   ├── controllers/          # Endpoint Controllers / Routers
│   │   ├── health_controller.py
│   │   └── resume_controller.py
│   ├── schemas/              # Request / Response Schemas
│   │   └── resume_schema.py
│   ├── services/             # Business Logic & LLM Services
│   │   └── resume_service.py
│   ├── config.py             # System Configurations & Prompts
│   └── main.py               # FastAPI App Initializer
├── agent.py                  # LiveKit Voice Agent Worker (VAD + STT)
├── main.py                   # Root entrypoint exporting app
├── requirements.txt          # Dependencies
├── .env.example              # Environment variables template
├── .env                      # Local environment configuration
├── .gitignore                # Git ignore rules
├── test_main.py              # Unit tests suite
└── README.md                 # Documentation
```

## Setup & Running

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
Update `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini/gemini-2.0-flash
PORT=8000
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Services

#### A. Resume Normalization FastAPI Service
```bash
uvicorn main:app --reload --port 8000
```

#### B. LiveKit Voice Agent Worker (VAD + STT)
```bash
python agent.py dev
```

## LiveKit Voice Agent Worker & Manual Testing

The `agent.py` script runs a standalone LiveKit Agents worker registered under `agent_name="interview-agent"`.
It performs:
- **Voice Activity Detection (VAD)** using local Silero VAD.
- **Speech-to-Text (STT)** using Groq's hosted Whisper API (`groq.STT`).

### Manual Testing Steps

1. **Start LiveKit Dev Server**:
   ```bash
   livekit-server --dev
   ```

2. **Start the Voice Agent Worker**:
   ```bash
   python agent.py dev
   ```

3. **Connect a Test Client**:
   - Open [LiveKit Agents Playground](https://agents-playground.livekit.io/).
   - Connect to your local LiveKit server instance (`ws://localhost:7880` with dev credentials) or cloud LiveKit project.
   - Speak into your microphone.

4. **Verify Transcript Output**:
   - The worker console will output finalized transcripts with timestamps:
     ```text
     [2026-08-09 02:45:00] [session=test-session] Candidate said: Hello, I am ready for the interview.
     ```

## API Endpoints

- **GET `/health`**: Returns `{"status": "ok"}`
- **POST `/resume/normalize`**: Accepts JSON `{ "rawText": "..." }` and returns `{ "cleanedText": "..." }`
- **Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
