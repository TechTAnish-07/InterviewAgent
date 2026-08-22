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
    custom_api_key = request.apiKey or request.api_key
    model_name = request.modelName or request.model_name or request.model
    res = await resume_service.normalize_resume(
        request.rawText,
        api_key=custom_api_key,
        provider=request.provider,
        model_name=model_name,
    )
    return ResumeNormalizeResponse(
        cleanedText=res.get("cleanedText", ""),
        candidateName=res.get("candidateName"),
        summary=res.get("summary"),
        skills=res.get("skills", []),
        suitableRoles=res.get("suitableRoles", []),
        experienceLevel=res.get("experienceLevel"),
    )


@router.post("/check-relevance", response_model=ResumeCheckRelevanceResponse)
async def check_relevance(request: ResumeCheckRelevanceRequest):
    custom_api_key = request.apiKey or request.api_key
    model_name = request.modelName or request.model_name or request.model
    res = await resume_service.check_relevance(
        request.resumeText,
        request.jobTitle,
        request.suitableRoles,
        api_key=custom_api_key,
        provider=request.provider,
        model_name=model_name,
    )
    return ResumeCheckRelevanceResponse(relevant=res["relevant"], reason=res["reason"])

