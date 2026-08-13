import json
import logging
import os
import aiohttp
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


@llm.function_tool(
    name="repeat_last_response",
    description=(
        "Repeat or recall the interviewer's most recent spoken response. "
        "Use this ONLY when the candidate explicitly asks you to repeat yourself, "
        "such as 'please repeat that', 'can you say that again', 'I didn't catch that', "
        "'could you repeat the question', or similar phrasing."
    ),
)
async def repeat_last_response() -> str:
    """
    Signals the agent to replay its last spoken response from session memory.
    The actual text retrieval and re-speaking is handled in agent.py.
    """
    logger.info("repeat_last_response tool invoked — agent will replay last spoken text.")
    return "REPEAT_LAST_RESPONSE"


@llm.function_tool(
    name="show_coding_question",
    description=(
        "Display a coding question on the candidate's screen in their code editor panel. "
        "Use this to present ONE technical coding question matched to their experience level."
    ),
)
async def show_coding_question(
    question_text: str,
    context: dict | None = None,
) -> str:
    """
    Publishes a { type: "coding_question", questionText: question_text } data message into the room
    and stores question_text in session userdata as current_question.
    """
    logger.info("show_coding_question tool invoked with question: %r", question_text[:80])
    if context is not None:
        context["current_question"] = question_text
        room = context.get("room")
        if room and hasattr(room, "local_participant") and room.local_participant:
            try:
                payload = json.dumps({
                    "type": "coding_question",
                    "questionText": question_text,
                })
                await room.local_participant.publish_data(payload, reliable=True, topic="interview_events")
                logger.info("Successfully published coding_question data packet to room.")
            except Exception as e:
                logger.error("Failed to publish coding_question data packet: %s", e)

    return f"Coding question displayed on candidate screen: {question_text}"


@llm.function_tool(
    name="get_current_code",
    description=(
        "Retrieve the latest code currently written by the candidate in their editor along with "
        "their own execution results (stdout, stderr, status) if they ran it. "
        "Use this when the candidate signals they are finished with their coding solution."
    ),
)
async def get_current_code(context: dict | None = None) -> str:
    """
    Returns the latest_code and last_run_result currently stored in session userdata.
    """
    if not context:
        return "No session userdata context available."

    latest_code = context.get("latest_code", "")
    last_run_result = context.get("last_run_result")

    result_parts = []
    if latest_code and latest_code.strip():
        result_parts.append(f"Candidate's Latest Code in Editor:\n```\n{latest_code}\n```")
    else:
        result_parts.append("Candidate has not written any code in the editor yet.")

    if last_run_result and isinstance(last_run_result, dict):
        status = last_run_result.get("status", "Unknown")
        stdout = last_run_result.get("stdout", "")
        stderr = last_run_result.get("stderr", "")
        result_parts.append(
            f"Candidate's Own Run Result:\n"
            f"- Status: {status}\n"
            f"- Output (stdout): {stdout if stdout else '(no stdout)'}\n"
            f"- Errors (stderr): {stderr if stderr else '(no stderr)'}"
        )
    else:
        result_parts.append("Candidate has not clicked Run or executed their code yet.")

    return "\n\n".join(result_parts)


@llm.function_tool(
    name="run_code_check",
    description=(
        "Fallback tool: Execute the candidate's code on the backend with optional stdin to test an edge case they didn't try, "
        "or if they didn't run it themselves. Prefer discussing their own run results when available."
    ),
)
async def run_code_check(
    language: str,
    stdin: str = "",
    context: dict | None = None,
) -> str:
    """
    Calls POST /api/ai-interview/{sessionId}/execute-code on Spring Boot backend
    with the current latest_code from session userdata, given language, and optional stdin.
    """
    if not context:
        return "Error: No session context available."

    session_id = context.get("session_id")
    latest_code = context.get("latest_code", "")
    if not latest_code or not latest_code.strip():
        return "Error: Cannot execute code because the candidate has not written any code yet."

    spring_base_url = os.getenv("SPRING_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8080"
    internal_api_key = os.getenv("INTERNAL_API_KEY") or "internal-secret-key"

    url = f"{spring_base_url}/api/ai-interview/{session_id}/execute-code"
    headers = {
        "X-Internal-Api-Key": internal_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "code": latest_code,
        "language": language,
        "stdin": stdin,
    }

    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status = data.get("status") or data.get("statusDescription") or "Completed"
                    stdout = data.get("stdout") or data.get("output") or ""
                    stderr = data.get("stderr") or data.get("error") or ""
                    return (
                        f"Execution Result for {language} (stdin={stdin!r}):\n"
                        f"- Status: {status}\n"
                        f"- Output (stdout): {stdout if stdout else '(no stdout)'}\n"
                        f"- Error (stderr): {stderr if stderr else '(no stderr)'}"
                    )
                else:
                    err_body = await resp.text()
                    logger.error("Spring Boot execute-code endpoint returned status %s for session %s: %s", resp.status, session_id, err_body)
                    return f"Execution check failed with HTTP {resp.status}: {err_body}"
    except Exception as e:
        logger.error("Network error executing code check for session %s: %s", session_id, e)
        return f"Execution check failed due to network error: {e}"
