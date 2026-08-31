import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.resume_service import resume_service

import os

client = TestClient(app, headers={"X-Internal-Api-Key": os.getenv("INTERNAL_API_KEY", "internal-secret-key")})



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
    monkeypatch.setenv("LOCAL_LLM", "false")
    monkeypatch.setenv("MODEL_NAME", "gemini/gemini-2.5-flash")
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
        MagicMock(message=MagicMock(content='''```json
{
  "candidateName": "John Doe",
  "summary": "Software Engineer with 5+ years experience.",
  "skills": ["Python", "FastAPI"],
  "suitableRoles": ["Software Engineer", "Backend Developer"],
  "experienceLevel": "Senior",
  "cleanedMarkdown": "## Summary\\nCandidate summary"
}
```'''))
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
    assert response.json() == {
        "cleanedText": "## Summary\nCandidate summary",
        "candidateName": "John Doe",
        "summary": "Software Engineer with 5+ years experience.",
        "skills": ["Python", "FastAPI"],
        "suitableRoles": ["Software Engineer", "Backend Developer"],
        "experienceLevel": "Senior"
    }


def test_strip_code_fences():
    assert resume_service.strip_code_fences("```markdown\n## Skills\nPython\n```") == "## Skills\nPython"
    assert resume_service.strip_code_fences("```\n## Skills\nPython\n```") == "## Skills\nPython"
    assert resume_service.strip_code_fences("## Skills\nPython") == "## Skills\nPython"


@patch("app.services.resume_service.acompletion")
def test_check_relevance_success(mock_acompletion, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="RELEVANT: YES\nREASON: Candidate has strong relevant technical experience."))
    ]

    async def async_return(*args, **kwargs):
        return mock_response

    mock_acompletion.side_effect = async_return

    response = client.post(
        "/resume/check-relevance",
        json={"resumeText": "Experienced Backend Software Engineer with Java and Python.", "jobTitle": "Backend Engineer"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "relevant": True,
        "reason": "Candidate has strong relevant technical experience."
    }


@patch("app.services.resume_service.acompletion")
def test_check_relevance_fail_open_on_error(mock_acompletion, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_acompletion.side_effect = Exception("LLM call timeout")

    response = client.post(
        "/resume/check-relevance",
        json={"resumeText": "Experienced Backend Software Engineer.", "jobTitle": "Backend Engineer"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "relevant": True,
        "reason": "Resume assessment completed."
    }


@patch("app.services.resume_service.acompletion")
def test_check_relevance_markdown_bold_parsing(mock_acompletion, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="**RELEVANT:** NO\n**REASON:** Marketing experience does not match Backend Engineer role."))
    ]

    async def async_return(*args, **kwargs):
        return mock_response

    mock_acompletion.side_effect = async_return

    response = client.post(
        "/resume/check-relevance",
        json={"resumeText": "Marketing Manager with 5 years experience.", "jobTitle": "Backend Engineer"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "relevant": False,
        "reason": "Marketing experience does not match Backend Engineer role."
    }


@patch("app.services.resume_service.acompletion")
def test_normalize_with_custom_api_key_and_provider(mock_acompletion):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='''```json
{
  "candidateName": "Jane Doe",
  "summary": "Full Stack Engineer",
  "skills": ["React", "Node.js"],
  "suitableRoles": ["Full Stack Developer"],
  "experienceLevel": "Mid",
  "cleanedMarkdown": "## Cleaned Resume"
}
```'''))
    ]

    captured_kwargs = {}

    async def async_return(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    mock_acompletion.side_effect = async_return

    raw_text = "Jane Doe - Full Stack Engineer with 4 years experience building web applications."
    response = client.post(
        "/resume/normalize",
        json={
            "rawText": raw_text,
            "apiKey": "sk-custom-fe-key",
            "provider": "openai",
            "modelName": "gpt-4o",
        },
    )

    assert response.status_code == 200
    assert response.json()["candidateName"] == "Jane Doe"
    assert captured_kwargs.get("model") == "openai/gpt-4o"
    assert captured_kwargs.get("api_key") == "sk-custom-fe-key"


@patch("app.services.resume_service.acompletion")
def test_check_relevance_with_custom_api_key(mock_acompletion):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="RELEVANT: YES\nREASON: Strong fullstack background."))
    ]

    captured_kwargs = {}

    async def async_return(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    mock_acompletion.side_effect = async_return

    response = client.post(
        "/resume/check-relevance",
        json={
            "resumeText": "Full Stack developer with React, Node, and Python experience.",
            "jobTitle": "Full Stack Engineer",
            "apiKey": "sk-custom-key-123",
            "modelName": "anthropic/claude-3-5-sonnet-20241022",
        },
    )

    assert response.status_code == 200
    assert response.json()["relevant"] is True
    assert captured_kwargs.get("model") == "anthropic/claude-3-5-sonnet-20241022"
    assert captured_kwargs.get("api_key") == "sk-custom-key-123"


@patch("app.controllers.dispatch_controller.api.LiveKitAPI")
def test_dispatch_agent_with_custom_api_key_and_provider(mock_livekit_api):
    import json
    mock_lk_instance = MagicMock()
    mock_dispatch = MagicMock(id="disp_12345")

    async def async_create_dispatch(req):
        return mock_dispatch

    async def async_close():
        pass

    mock_lk_instance.agent_dispatch.create_dispatch.side_effect = async_create_dispatch
    mock_lk_instance.aclose.side_effect = async_close
    mock_livekit_api.return_value = mock_lk_instance

    response = client.post(
        "/dispatch-agent",
        json={
            "room": "room_test_101",
            "session_id": "sess_test_101",
            "candidate_name": "Alice Smith",
            "job_title": "Senior Backend Engineer",
            "apiKey": "sk-user-key-abc",
            "provider": "groq",
            "modelName": "llama-3.3-70b-versatile",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "dispatch_id": "disp_12345",
        "room": "room_test_101",
        "session_id": "sess_test_101",
    }

    # Verify metadata sent to LiveKit
    call_args = mock_lk_instance.agent_dispatch.create_dispatch.call_args[0][0]
    meta = json.loads(call_args.metadata)
    assert meta["sessionId"] == "sess_test_101"
    assert meta["candidateName"] == "Alice Smith"
    assert meta["apiKey"] == "sk-user-key-abc"
    assert meta["provider"] == "groq"
    assert meta["modelName"] == "groq/llama-3.3-70b-versatile"


def test_agent_extract_context_from_metadata():
    import json
    from agent import extract_context_from_metadata

    raw_meta = json.dumps({
        "candidateName": "Bob Ross",
        "jobTitle": "Painter & Engineer",
        "summary": "Creative developer",
        "skills": "['Art', 'Python']",
        "apiKey": "sk-test-metadata-key",
        "provider": "openai",
        "modelName": "gpt-4o-mini",
    })

    ctx = extract_context_from_metadata(raw_meta)
    assert ctx is not None
    assert ctx["candidateName"] == "Bob Ross"
    assert ctx["jobTitle"] == "Painter & Engineer"
    assert ctx["apiKey"] == "sk-test-metadata-key"
    assert ctx["provider"] == "openai"
    assert ctx["modelName"] == "openai/gpt-4o-mini"


