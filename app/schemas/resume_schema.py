from pydantic import BaseModel


class ResumeNormalizeRequest(BaseModel):
    rawText: str


class ResumeNormalizeResponse(BaseModel):
    cleanedText: str
