import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_LLM = os.getenv("LOCAL_LLM", "false").lower() in ("true", "1", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "ollama/qwen3.5:9b" if LOCAL_LLM else "gemini/gemini-2.5-flash")
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "ollama/qwen3.5:9b" if LOCAL_LLM else "groq/llama-3.1-8b-instant")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
PORT = int(os.getenv("PORT", 8000))


def is_local_llm(model: str | None = None) -> bool:
    """Return True if local LLM is explicitly enabled or model points to ollama."""
    if LOCAL_LLM:
        return True
    if model and str(model).lower().startswith("ollama/"):
        return True
    return False

