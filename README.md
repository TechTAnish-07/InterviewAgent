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
Copy `.env.example` to `.env` and set your API key:
```bash
cp .env.example .env
```
Update `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini/gemini-2.0-flash
PORT=8000
```

### 4. Run the Service
```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

- **GET `/health`**: Returns `{"status": "ok"}`
- **POST `/resume/normalize`**: Accepts JSON `{ "rawText": "..." }` and returns `{ "cleanedText": "..." }`
- **Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
