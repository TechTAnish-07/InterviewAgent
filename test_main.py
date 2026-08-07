import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.resume_service import resume_service

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_normalize_text_too_short():
    response = client.post(
        "/resume/normalize",
        json={"rawText": "Short text"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "rawText must be at least 50 characters long"


def test_normalize_missing_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    response = client.post(
        "/resume/normalize",
        json={"rawText": "This is a long enough text that exceeds fifty characters limit for testing."}
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "LLM normalization failed"


@patch("app.services.resume_service.acompletion")
def test_normalize_success(mock_acompletion, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="```markdown\n## Summary\nCandidate summary\n```"))
    ]
    
    async def async_return(*args, **kwargs):
        return mock_response
        
    mock_acompletion.side_effect = async_return
    
    raw_text = "John Doe - Software Engineer - 5+ years experience building scalable Python web applications and microservices."
    response = client.post(
        "/resume/normalize",
        json={"rawText": raw_text}
    )
    
    assert response.status_code == 200
    assert response.json() == {"cleanedText": "## Summary\nCandidate summary"}


def test_strip_code_fences():
    assert resume_service.strip_code_fences("```markdown\n## Skills\nPython\n```") == "## Skills\nPython"
    assert resume_service.strip_code_fences("```\n## Skills\nPython\n```") == "## Skills\nPython"
    assert resume_service.strip_code_fences("## Skills\nPython") == "## Skills\nPython"
