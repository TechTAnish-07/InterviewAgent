import logging
from livekit.agents import llm

logger = logging.getLogger("interview-agent.tools")


@llm.function_tool(
    name="get_resume_context",
    description="Retrieve candidate resume text, job title/role, and candidate name to ground interview questions.",
)
async def get_resume_context(context: dict | None = None) -> str:
    """
    Returns stored candidate resume details, job title, and name.
    """
    if not context:
        return "No resume context available for this candidate."

    candidate_name = context.get("candidateName") or "Candidate"
    job_role = context.get("jobTitle") or context.get("jobRole") or "Software Engineer"
    resume_text = context.get("resumeText") or "No resume text provided."

    return (
        f"Candidate Name: {candidate_name}\n"
        f"Job Role / Title: {job_role}\n"
        f"Resume Text:\n{resume_text}"
    )


@llm.function_tool(
    name="end_interview",
    description="Wrap up and conclude the live interview when enough ground has been covered or the candidate requests to finish.",
)
async def end_interview(reason: str = "interview_complete") -> str:
    """
    Signals that the interview should end.
    """
    logger.info("end_interview tool invoked with reason: %s", reason)
    return f"Interview conclusion requested with reason: {reason}"
