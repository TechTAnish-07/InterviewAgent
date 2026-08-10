from pydantic import BaseModel


class ResumeNormalizeRequest(BaseModel):
    rawText: str


class ResumeNormalizeResponse(BaseModel):
    cleanedText: str


class ResumeCheckRelevanceRequest(BaseModel):
    resumeText: str
    jobTitle: str


class ResumeCheckRelevanceResponse(BaseModel):
    relevant: bool
    reason: str

