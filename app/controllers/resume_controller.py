from fastapi import APIRouter
from app.schemas.resume_schema import (
    ResumeNormalizeRequest,
    ResumeNormalizeResponse,
    ResumeCheckRelevanceRequest,
    ResumeCheckRelevanceResponse,
)
from app.services.resume_service import resume_service

router = APIRouter(prefix="/resume", tags=["Resume"])


@router.post("/normalize", response_model=ResumeNormalizeResponse)
async def normalize_resume(request: ResumeNormalizeRequest):
    cleaned = await resume_service.normalize_resume(request.rawText)
    return ResumeNormalizeResponse(cleanedText=cleaned)


@router.post("/check-relevance", response_model=ResumeCheckRelevanceResponse)
async def check_relevance(request: ResumeCheckRelevanceRequest):
    res = await resume_service.check_relevance(request.resumeText, request.jobTitle)
    return ResumeCheckRelevanceResponse(relevant=res["relevant"], reason=res["reason"])

