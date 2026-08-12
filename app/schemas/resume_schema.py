from pydantic import BaseModel


class ResumeNormalizeRequest(BaseModel):
    rawText: str


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


class ResumeCheckRelevanceResponse(BaseModel):
    relevant: bool
    reason: str

