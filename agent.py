import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents
from livekit.plugins import silero, groq

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")


def handle_transcript(session_id: str, transcript_text: str) -> None:
    """
    Placeholder function called when a finalized transcript is received.
    
    TODO: This is a placeholder. The main LLM interview logic will be
    implemented here in a later task — it will take this transcript,
    combine it with resume context and conversation history, and
    generate the interviewer's next response.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [session={session_id}] Candidate said: {transcript_text}")


def extract_session_id(metadata_raw: str | None) -> str:
    """Extract sessionId from job metadata, defaulting to 'test-session' if absent or invalid."""
    if not metadata_raw:
        return "test-session"
    try:
        data = json.loads(metadata_raw)
        if isinstance(data, dict):
            return data.get("sessionId") or data.get("session_id") or metadata_raw
        return str(data)
    except Exception:
        return metadata_raw


async def entrypoint(ctx: agents.JobContext) -> None:
    session_id = extract_session_id(ctx.job.metadata)
    logger.info("Agent starting session %s in room %s...", session_id, ctx.room.name)

    # Initialize LiveKit AgentSession with Silero VAD and Groq STT
    session = agents.AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(),
    )

    # Event handler for finalized user transcripts
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev: agents.UserInputTranscribedEvent) -> None:
        if ev.is_final and ev.transcript and ev.transcript.strip():
            handle_transcript(session_id, ev.transcript.strip())

    # Start agent session attached to room (this automatically connects ctx.room)
    await session.start(
        agent=agents.Agent(instructions=""),
        room=ctx.room,
    )


if __name__ == "__main__":
    agent_name = os.getenv("AGENT_NAME", "")
    logger.info("Starting worker with agent_name: '%s'", agent_name)
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        )
    )
