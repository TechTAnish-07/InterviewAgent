import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query

from app.schemas.evaluation_schema import InterviewEvaluationResult, UserAnalyticsSummary
from app.services.storage_service import storage_service

logger = logging.getLogger("analytics-controller")

router = APIRouter(prefix="/analytics", tags=["Analytics & Topic Performance"])


@router.post("/evaluations", response_model=Dict[str, Any])
async def save_interview_evaluation(evaluation: InterviewEvaluationResult):
    """
    Save or update an interview evaluation record and update the candidate's longitudinal topic mastery profile.
    """
    try:
        result = storage_service.save_evaluation(evaluation)
        return result
    except Exception as e:
        logger.error("Failed to save evaluation for session %s: %s", evaluation.session_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to persist evaluation: {e}")


@router.get("/evaluations/{session_id}", response_model=Dict[str, Any])
async def get_session_evaluation(session_id: str):
    """
    Retrieve full evaluation details for a specific interview session.
    """
    eval_data = storage_service.get_evaluation(session_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Evaluation for session '{session_id}' not found.")
    return eval_data


@router.get("/users/{user_id}", response_model=UserAnalyticsSummary)
async def get_user_analytics_summary(user_id: str):
    """
    Retrieve comprehensive topic analytics for a candidate:
    - Overall average score & session count
    - Top strengths (best mastered topics)
    - Top weaknesses (weakest topics needing practice)
    - Full topic mastery matrix & competency averages
    - Recent interview history
    """
    try:
        return storage_service.get_user_analytics(user_id)
    except Exception as e:
        logger.error("Failed to fetch analytics for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to compute user analytics: {e}")


@router.get("/users/{user_id}/evaluations", response_model=List[Dict[str, Any]])
async def get_user_evaluations_history(user_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """
    Retrieve all historical interview evaluations for a candidate.
    """
    return storage_service.get_user_evaluations(user_id, limit=limit)


@router.get("/users/{user_id}/weak-topics", response_model=List[str])
async def get_candidate_weak_topics(user_id: str, limit: int = Query(default=3, ge=1, le=10)):
    """
    Retrieve top weak topics for a candidate to tailor future interview questions.
    """
    return storage_service.get_user_weak_topics_for_interview(user_id, limit=limit)
