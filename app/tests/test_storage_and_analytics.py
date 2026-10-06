import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.evaluation_schema import (
    CompetencyScores,
    InterviewEvaluationResult,
    TopicAssessment,
)
from app.services.storage_service import StorageService
from agent import _parse_feedback_response


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_interviews.db"
        storage = StorageService(db_path=db_path)
        yield storage


def test_storage_save_and_retrieve_evaluation(temp_storage):
    eval_result = InterviewEvaluationResult(
        session_id="sess_101",
        user_id="user_abc",
        candidate_name="John Doe",
        job_role="Senior Backend Engineer",
        overall_score=8.5,
        competencies=CompetencyScores(
            technical_depth=8.0,
            problem_solving=9.0,
            communication=8.5,
            code_quality=8.0,
        ),
        strong_topics=[
            TopicAssessment(
                topic="System Design: Caching & Redis",
                category="System Design",
                score=9.0,
                status="strong",
                evidence="Explained cache-aside and write-through patterns clearly.",
            )
        ],
        weak_topics=[
            TopicAssessment(
                topic="SQL Indexing & Query Tuning",
                category="Database",
                score=5.0,
                status="weak",
                evidence="Uncertain about composite index leftmost prefix rule.",
                recommendation="Review B-Tree index scan patterns.",
            )
        ],
        overview="Strong system design performance with minor database gaps.",
        markdown_feedback="## Overview\nGood job!",
    )

    save_res = temp_storage.save_evaluation(eval_result)
    assert save_res["status"] == "success"
    assert save_res["session_id"] == "sess_101"

    fetched = temp_storage.get_evaluation("sess_101")
    assert fetched is not None
    assert fetched["session_id"] == "sess_101"
    assert fetched["overall_score"] == 8.5
    assert len(fetched["strong_topics"]) == 1
    assert fetched["strong_topics"][0]["topic"] == "System Design: Caching & Redis"
    assert len(fetched["weak_topics"]) == 1
    assert fetched["weak_topics"][0]["topic"] == "SQL Indexing & Query Tuning"


def test_storage_longitudinal_mastery_and_analytics(temp_storage):
    user_id = "candidate_xyz"

    # Session 1: Candidate is weak in SQL Indexing (4.0) and strong in Redis (8.5)
    sess1 = InterviewEvaluationResult(
        session_id="sess_001",
        user_id=user_id,
        candidate_name="Alice",
        job_role="Backend Developer",
        overall_score=6.5,
        competencies=CompetencyScores(technical_depth=6.0, problem_solving=7.0, communication=7.0),
        strong_topics=[
            TopicAssessment(topic="Redis Caching", category="System Design", score=8.5, status="strong", evidence="Great answer")
        ],
        weak_topics=[
            TopicAssessment(topic="SQL Indexing", category="Database", score=4.0, status="weak", evidence="Struggled with EXPLAIN", recommendation="Study EXPLAIN")
        ],
        overview="First interview session.",
        markdown_feedback="## Report 1",
    )
    temp_storage.save_evaluation(sess1)

    weak_topics_first = temp_storage.get_user_weak_topics_for_interview(user_id)
    assert "SQL Indexing" in weak_topics_first

    analytics1 = temp_storage.get_user_analytics(user_id)
    assert analytics1.total_interviews == 1
    assert analytics1.average_overall_score == 6.5
    assert len(analytics1.top_weaknesses) == 1
    assert analytics1.top_weaknesses[0].topic == "SQL Indexing"
    assert analytics1.top_weaknesses[0].mastery_level == "NEEDS_WORK"
    assert len(analytics1.top_strengths) == 1
    assert analytics1.top_strengths[0].topic == "Redis Caching"

    # Session 2: Candidate practiced and improved in SQL Indexing (8.0), and did great in Microservices (9.0)
    sess2 = InterviewEvaluationResult(
        session_id="sess_002",
        user_id=user_id,
        candidate_name="Alice",
        job_role="Backend Developer",
        overall_score=8.5,
        competencies=CompetencyScores(technical_depth=8.5, problem_solving=8.5, communication=8.0),
        strong_topics=[
            TopicAssessment(topic="SQL Indexing", category="Database", score=8.0, status="strong", evidence="Much improved on composite indexes"),
            TopicAssessment(topic="Microservices", category="Architecture", score=9.0, status="strong", evidence="Mastered circuit breakers"),
        ],
        weak_topics=[],
        overview="Second interview session - excellent progress.",
        markdown_feedback="## Report 2",
    )
    temp_storage.save_evaluation(sess2)

    analytics2 = temp_storage.get_user_analytics(user_id)
    assert analytics2.total_interviews == 2
    assert analytics2.average_overall_score == 7.5  # (6.5 + 8.5) / 2
    # SQL Indexing average is now (4.0 + 8.0) / 2 = 6.0 (DEVELOPING)
    sql_topic = next(t for t in analytics2.all_topic_mastery if t.topic == "SQL Indexing")
    assert sql_topic.attempts_count == 2
    assert sql_topic.average_score == 6.0
    assert sql_topic.mastery_level == "DEVELOPING"


def test_parse_feedback_response_json():
    json_text = """```json
    {
      "overall_score": 8.0,
      "competencies": {
        "technical_depth": 8.5,
        "problem_solving": 7.5,
        "communication": 8.0,
        "code_quality": 8.0
      },
      "strong_topics": [
        {
          "topic": "Python Generators",
          "category": "Language Internals",
          "score": 9.0,
          "status": "strong",
          "evidence": "Accurately detailed yield and memory efficiency."
        }
      ],
      "weak_topics": [
        {
          "topic": "Thread Safety & GIL",
          "category": "Concurrency",
          "score": 5.0,
          "status": "weak",
          "evidence": "Confused GIL with multithreading synchronization.",
          "recommendation": "Review threading.Lock and multiprocessing."
        }
      ],
      "all_topics": [],
      "overview": "Strong overall candidate.",
      "markdown_feedback": "## Overview\\nCandidate showed solid Python understanding."
    }
    ```"""

    eval_result, md = _parse_feedback_response(
        raw_text=json_text,
        candidate_name="Bob",
        job_role="Python Developer",
        session_id="sess_json_1",
        user_id="user_bob",
    )

    assert eval_result.overall_score == 8.0
    assert len(eval_result.strong_topics) == 1
    assert eval_result.strong_topics[0].topic == "Python Generators"
    assert len(eval_result.weak_topics) == 1
    assert eval_result.weak_topics[0].topic == "Thread Safety & GIL"
    assert "Candidate showed solid Python understanding." in md


def test_parse_feedback_response_fallback_markdown():
    plain_md = "## Overview\nCandidate performed well in general communication."
    eval_result, md = _parse_feedback_response(
        raw_text=plain_md,
        candidate_name="Bob",
        job_role="Python Developer",
        session_id="sess_fallback_1",
        user_id="user_bob",
    )

    assert eval_result.session_id == "sess_fallback_1"
    assert eval_result.overall_score == 7.0
    assert md == plain_md


def test_analytics_controller_endpoints():
    client = TestClient(app)

    payload = {
        "session_id": "api_test_sess",
        "user_id": "api_user_1",
        "candidate_name": "Charlie",
        "job_role": "Fullstack Engineer",
        "overall_score": 8.0,
        "competencies": {
            "technical_depth": 8.0,
            "problem_solving": 8.0,
            "communication": 8.0
        },
        "strong_topics": [
            {
                "topic": "React Hooks",
                "category": "Frontend",
                "score": 8.5,
                "status": "strong",
                "evidence": "Deep knowledge of useEffect dependencies."
            }
        ],
        "weak_topics": [
            {
                "topic": "CSS Grid & Flexbox",
                "category": "Frontend",
                "score": 4.5,
                "status": "weak",
                "evidence": "Could not align items properly.",
                "recommendation": "Practice Flexbox layouts."
            }
        ],
        "overview": "Solid front-end interview.",
        "markdown_feedback": "## Overview\nSolid performance."
    }

    # 1. Post evaluation
    post_resp = client.post("/analytics/evaluations", json=payload)
    assert post_resp.status_code == 200
    assert post_resp.json()["session_id"] == "api_test_sess"

    # 2. Get session evaluation
    get_sess_resp = client.get("/analytics/evaluations/api_test_sess")
    assert get_sess_resp.status_code == 200
    assert get_sess_resp.json()["candidate_name"] == "Charlie"

    # 3. Get user analytics
    user_resp = client.get("/analytics/users/api_user_1")
    assert user_resp.status_code == 200
    user_data = user_resp.json()
    assert user_data["user_id"] == "api_user_1"
    assert user_data["total_interviews"] >= 1
    assert len(user_data["top_strengths"]) >= 1
    assert len(user_data["top_weaknesses"]) >= 1

    # 4. Get weak topics
    weak_resp = client.get("/analytics/users/api_user_1/weak-topics")
    assert weak_resp.status_code == 200
    assert "CSS Grid & Flexbox" in weak_resp.json()
