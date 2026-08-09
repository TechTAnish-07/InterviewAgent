import asyncio
import json
import logging
import os
from datetime import datetime

import aiohttp
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import silero, openai as lk_openai, groq

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8080"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or "internal-secret-key"


def extract_session_id(metadata_raw: str | None) -> str | None:
    """
    Extract sessionId from job metadata JSON string.
    Returns string sessionId if present, or None if missing or unparseable.
    """
    if not metadata_raw or not metadata_raw.strip():
        return None
    try:
        data = json.loads(metadata_raw)
        if isinstance(data, dict):
            sid = data.get("sessionId") or data.get("session_id")
            return str(sid) if sid is not None else None
        elif isinstance(data, (int, str)):
            return str(data)
        return None
    except Exception as e:
        logger.error("Failed to parse job metadata JSON: %s", e)
        return None


async def fetch_interview_context(session_id: str) -> dict | None:
    """
    Fetch interview context (candidateName, resumeText, jobRole) from Spring Boot backend.
    GET {SPRING_BASE_URL}/api/ai-interview/{sessionId}/context
    Headers: X-Internal-Api-Key: {INTERNAL_API_KEY}
    Returns context dict on success, or None on failure.
    """
    url = f"{SPRING_BASE_URL}/api/ai-interview/{session_id}/context"
    headers = {"X-Internal-Api-Key": INTERNAL_API_KEY}

    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    candidate_name = data.get("candidateName")
                    job_role = data.get("jobRole")
                    if candidate_name is None and job_role is None:
                        logger.error("Spring Boot context response missing required fields for session %s: %s", session_id, data)
                        return None
                    return data
                else:
                    error_body = await resp.text()
                    logger.error("Spring Boot returned non-200 status (%s) for session %s context: %s", resp.status, session_id, error_body)
                    return None
    except Exception as e:
        logger.error("Network error fetching interview context from Spring Boot for session %s: %s", session_id, e)
        return None


def handle_transcript(session_id: str, transcript_text: str) -> None:
    """
    Called whenever a finalized candidate transcript is received from STT.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_output = (
        f"\n{'=' * 65}\n"
        f" 🎙️  STT TRANSCRIPT RECEIVED [{timestamp}] (Session: {session_id})\n"
        f" 💬 Candidate said: \"{transcript_text}\"\n"
        f"{'=' * 65}\n"
    )
    print(formatted_output, flush=True)
    logger.info("[STT TRANSCRIPT] session=%s: %s", session_id, transcript_text)


async def entrypoint(ctx: agents.JobContext) -> None:
    session_id = extract_session_id(ctx.job.metadata)
    if not session_id:
        logger.error("Job metadata is missing or unparseable (metadata=%r). Disconnecting gracefully.", ctx.job.metadata)
        await ctx.connect()
        await ctx.room.disconnect()
        return

    logger.info("Agent starting explicit session %s in room %s...", session_id, ctx.room.name)

    # Connect to the LiveKit room
    await ctx.connect()

    # Fetch interview context from Spring Boot backend
    context = await fetch_interview_context(session_id)
    if context is None:
        logger.error("Could not obtain interview context for session %s. Disconnecting from room.", session_id)
        await ctx.room.disconnect()
        return

    logger.info(
        "Loaded session %s context: candidate='%s', jobRole='%s', resumePresent=%s",
        session_id,
        context.get("candidateName"),
        context.get("jobRole"),
        bool(context.get("resumeText")),
    )

    # Initialize AgentSession with Silero VAD and Groq STT (Whisper Large v3)
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3"),
        userdata={
            "session_id": session_id,
            "interview_context": context,
        },
    )

    session_ended = False

    async def end_interview(sid: str, reason: str) -> None:
        """
        Notify Spring Boot backend that the interview has ended.

        TODO: This function will be called by the main LLM logic later when it decides
        the interview is complete (e.g. after asking all questions).
        """
        nonlocal session_ended
        if session_ended:
            logger.info("End request already sent for session %s, skipping.", sid)
            return
        session_ended = True

        url = f"{SPRING_BASE_URL}/api/ai-interview/{sid}/end"
        headers = {
            "X-Internal-Api-Key": INTERNAL_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"reason": reason}

        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info("Successfully notified Spring Boot of interview end for session %s (reason: %s)", sid, reason)
                    else:
                        resp_text = await resp.text()
                        logger.error("Spring Boot /end endpoint returned status %s for session %s: %s", resp.status, sid, resp_text)
        except Exception as e:
            logger.error("Error sending /end request to Spring Boot for session %s: %s", sid, e)

    # Event handler for participant disconnection (end of interview signaling)
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant %s disconnected from room %s.", participant.identity, ctx.room.name)
        asyncio.create_task(end_interview(session_id, "candidate_disconnected"))

    # Event handler for candidate transcripts from STT
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev: agents.UserInputTranscribedEvent) -> None:
        if ev.transcript and ev.transcript.strip():
            if ev.is_final:
                handle_transcript(session_id, ev.transcript.strip())
            else:
                logger.debug("[STT Interim] session=%s: %s", session_id, ev.transcript.strip())

    # Start the agent session (silent agent listening to candidate speech)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=""),
    )

    logger.info("Session %s is live and listening for candidate speech in room %s.", session_id, ctx.room.name)


if __name__ == "__main__":
    agent_name = os.getenv("AGENT_NAME") or "interview-agent"
    logger.info("Starting LiveKit worker registered as agent_name: '%s' (explicit dispatch enabled)", agent_name)
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        )
    )
