from pydantic import BaseModel


class ResumeNormalizeRequest(BaseModel):
    rawText: str
    apiKey: str | None = None
    api_key: str | None = None
    provider: str | None = None
    modelName: str | None = None
    model_name: str | None = None
    model: str | None = None


class ResumeNormalizeResponse(BaseModel):
    cleanedText: str
    candidateName: str | None = None
    summary: str | None = None
    skills: list[str] = []
    suitableRoles: list[str] = []
    experienceLevel: str | None = None


class ResumeCheckRelevanceRequest(BaseModel):
    resumeText: str
    jobTitle: str
    suitableRoles: list[str] | None = None
    apiKey: str | None = None
    api_key: str | None = None
    provider: str | None = None
    modelName: str | None = None
    model_name: str | None = None
    model: str | None = None


class ResumeCheckRelevanceResponse(BaseModel):
    relevant: bool
    reason: str

