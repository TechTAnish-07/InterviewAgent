import asyncio
import json
import logging
import os
from datetime import datetime

import aiohttp
import litellm
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, inference
from livekit.agents import llm
from livekit.plugins import fishaudio, groq, openai, silero

from memory import ConversationMemory
from moderation import check_message_relevance
from prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    GREETING_INSTRUCTION,
    OFF_TOPIC_WARNING_INSTRUCTION,
    INAPPROPRIATE_WARNING_INSTRUCTION,
    FINAL_WARNING_INSTRUCTION,
    FEEDBACK_GENERATION_PROMPT,
)
from tools import get_resume_context, end_interview

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8080"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or "internal-secret-key"
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini/gemini-2.0-flash"


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
    Fetch interview context (candidateName, resumeText, jobTitle/jobRole) from Spring Boot backend.
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
                    job_role = data.get("jobTitle") or data.get("jobRole")
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


async def send_feedback_report(session_id: str, feedback_text: str) -> bool:
    """
    POST feedback report to Spring Boot backend:
    POST {SPRING_BASE_URL}/api/ai-interview/{sessionId}/feedback
    """
    url = f"{SPRING_BASE_URL}/api/ai-interview/{session_id}/feedback"
    headers = {
        "X-Internal-Api-Key": INTERNAL_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"feedback": feedback_text}

    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    logger.info("Successfully posted feedback report to Spring Boot for session %s", session_id)
                    return True
                else:
                    resp_text = await resp.text()
                    logger.error("Spring Boot /feedback endpoint returned status %s for session %s: %s", resp.status, session_id, resp_text)
                    return False
    except Exception as e:
        logger.error("Network error posting feedback report to Spring Boot for session %s: %s", session_id, e)
        return False


async def notify_session_end(session_id: str, reason: str) -> None:
    """
    POST session end status to Spring Boot backend:
    POST {SPRING_BASE_URL}/api/ai-interview/{sessionId}/end
    """
    url = f"{SPRING_BASE_URL}/api/ai-interview/{session_id}/end"
    headers = {
        "X-Internal-Api-Key": INTERNAL_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"reason": reason}

    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logger.info("Successfully notified Spring Boot of session end for session %s (reason: %s)", session_id, reason)
                else:
                    resp_text = await resp.text()
                    logger.error("Spring Boot /end endpoint returned status %s for session %s: %s", resp.status, session_id, resp_text)
    except Exception as e:
        logger.error("Network error notifying Spring Boot of session end for session %s: %s", session_id, e)


async def generate_and_save_feedback(
    session_id: str,
    context: dict | None,
    memory: ConversationMemory | None,
    reason: str
) -> None:
    """
    Generate post-interview feedback report using Gemini via LiteLLM and POST it to Spring Boot.
    Skips generation if policy_violation with minimal conversation.
    """
    if reason == "policy_violation" and (not memory or (not memory.recent_turns and not memory.rolling_summary)):
        logger.info("Skipping feedback generation for session %s due to policy_violation with minimal turns.", session_id)
        return

    candidate_name = (context.get("candidateName") if context else None) or "Candidate"
    job_role = (context.get("jobTitle") or context.get("jobRole") if context else None) or "Software Engineer"
    resume_text = (context.get("resumeText") if context else None) or "Not provided"
    full_conversation = memory.get_full_context_text() if memory else "No turns recorded."

    prompt = FEEDBACK_GENERATION_PROMPT.format(
        candidate_name=candidate_name,
        job_role=job_role,
        resume_text=resume_text[:1000],
        full_conversation_context=full_conversation,
    )

    messages = [
        {"role": "system", "content": "You are an expert technical interviewer writing an objective feedback report."},
        {"role": "user", "content": prompt},
    ]

    try:
        logger.info("Generating feedback report for session %s...", session_id)
        response = litellm.completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_tokens=600,
        )
        feedback_text = response.choices[0].message.content.strip()
        logger.info("Feedback report generated for session %s. Saving to Spring Boot...", session_id)
        await send_feedback_report(session_id, feedback_text)
    except Exception as e:
        logger.error("Error generating feedback report for session %s: %s", session_id, e)


class InterviewAgent(Agent):
    """
    Custom Agent subclass for conducting adaptive, resume-grounded AI voice interviews.
    """

    def __init__(
        self,
        session_id: str,
        context: dict | None,
        system_prompt: str,
        memory: ConversationMemory,
        room: rtc.Room,
    ) -> None:
        super().__init__(instructions=system_prompt)
        self._session_id = session_id
        self._context = context
        self._system_prompt = system_prompt
        self._memory = memory
        self._room = room
        self._warning_count = 0
        self.has_candidate_spoken = False
        self._is_ending = False

    async def end_interview_flow(self, reason: str) -> None:
        """
        Execute session end, generate feedback report, notify Spring Boot, and disconnect room.
        """
        if self._is_ending:
            return
        self._is_ending = True
        logger.info("Executing end_interview_flow for session %s (reason: %s)...", self._session_id, reason)

        # Generate feedback report and save to Spring Boot
        await generate_and_save_feedback(self._session_id, self._context, self._memory, reason)

        # Notify Spring Boot of session end
        await notify_session_end(self._session_id, reason)

        # Disconnect gracefully from room
        try:
            await self._room.disconnect()
        except Exception as e:
            logger.warn("Error disconnecting room for session %s: %s", self._session_id, e)

    async def generate_greeting(self) -> str:
        """
        Generate warm greeting prompt when candidate stays silent at room start.
        Uses static template to save LLM API quota on room initialization.
        """
        candidate_name = (self._context.get("candidateName") if self._context else None) or "there"
        job_role = (self._context.get("jobTitle") or self._context.get("jobRole") if self._context else None) or "Software Engineer"
        return f"Hello {candidate_name}! Welcome to your technical interview for the {job_role} position. Could you please introduce yourself and walk me through your background?"

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """
        Called by LiveKit Agents framework when candidate finishes speaking and turn is committed.
        Runs moderation check, warning escalation, memory updates, and LLM reply generation.
        """
        if self._is_ending:
            return

        # Candidate spoke; cancel silence greeting
        self.has_candidate_spoken = True

        transcript = new_message.text_content or ""
        if not transcript.strip():
            return

        handle_transcript(self._session_id, transcript)

        # 1. Moderation check on candidate message
        classification = check_message_relevance(transcript, self._memory.recent_turns)
        logger.info("[MODERATION] session=%s classification=%s text=%r", self._session_id, classification, transcript)

        # 2. Moderation Escalation Handling
        if classification in ("OFF_TOPIC", "INAPPROPRIATE"):
            self._warning_count += 1
            logger.warning("[WARNING] session=%s warning_count=%d class=%s", self._session_id, self._warning_count, classification)

            if self._warning_count == 1:
                instruction = (
                    OFF_TOPIC_WARNING_INSTRUCTION
                    if classification == "OFF_TOPIC"
                    else INAPPROPRIATE_WARNING_INSTRUCTION
                )
                warning_reply = self._generate_warning_reply(transcript, instruction)
                self._speak_and_log(warning_reply)
                return
            elif self._warning_count == 2:
                warning_reply = self._generate_warning_reply(transcript, FINAL_WARNING_INSTRUCTION)
                self._speak_and_log(warning_reply)
                return
            else:
                # warning_count >= 3: Auto-end for policy violation
                logger.error("[POLICY VIOLATION] Session %s reached max warnings. Ending interview.", self._session_id)
                await self.end_interview_flow("policy_violation")
                return

        # 3. Normal Reply Flow
        context_messages = self._memory.build_context_messages()
        full_messages = [
            {"role": "system", "content": self._system_prompt},
            *context_messages,
            {"role": "user", "content": transcript},
        ]

        try:
            # LiteLLM call with Gemini model and tools enabled
            response = litellm.completion(
                model=MODEL_NAME,
                messages=full_messages,
                temperature=0.7,
                max_tokens=200,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_context",
                            "description": get_resume_context.info.description,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "end_interview",
                            "description": end_interview.info.description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "reason": {
                                        "type": "string",
                                        "description": "Reason for ending interview (e.g. interview_complete)",
                                    }
                                },
                            },
                        },
                    },
                ],
            )

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            # Check if model invoked end_interview tool
            if tool_calls:
                for tc in tool_calls:
                    func_name = getattr(tc.function, "name", "")
                    if func_name == "end_interview":
                        args_raw = getattr(tc.function, "arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except Exception:
                            args = {}
                        reason = args.get("reason", "interview_complete")
                        logger.info("[TOOL CALL] Model invoked end_interview with reason: %s", reason)
                        await self.end_interview_flow(reason)
                        return
                    elif func_name == "get_resume_context":
                        logger.info("[TOOL CALL] Model invoked get_resume_context")
                        # Return resume text and retry completion
                        resume_info = await get_resume_context(self._context)
                        full_messages.append({"role": "system", "content": f"Resume Context:\n{resume_info}"})
                        retry_resp = litellm.completion(
                            model=MODEL_NAME,
                            messages=full_messages,
                            temperature=0.7,
                            max_tokens=200,
                        )
                        msg = retry_resp.choices[0].message

            reply_text = (msg.content or "").strip()
            reply_text = reply_text.replace("*", "").replace("#", "").replace("`", "")

            if not reply_text:
                reply_text = "Thank you for explaining that. Could you tell me more about your technical experience?"

            # Speak reply via TTS
            self._speak_and_log(reply_text)

            # Update working memory
            summary_before = self._memory.rolling_summary
            self._memory.add_turn(transcript, reply_text)
            summary_updated = self._memory.rolling_summary != summary_before

            logger.info(
                "[TURN LOG] session=%s classification=%s summary_updated=%s candidate=%r reply=%r",
                self._session_id,
                classification,
                summary_updated,
                transcript,
                reply_text,
            )

        except Exception as e:
            logger.error("Error in normal reply flow for session %s: %s", self._session_id, e)
            fallback_reply = "Thank you. Could you please elaborate on your technical background?"
            self._speak_and_log(fallback_reply)

    def _generate_warning_reply(self, candidate_text: str, instruction_prompt: str) -> str:
        """
        Generate warning response using Gemini via LiteLLM for off-topic/inappropriate inputs.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Candidate message: \"{candidate_text}\"\nInstruction: {instruction_prompt}"},
        ]
        try:
            response = litellm.completion(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                max_tokens=100,
            )
            reply = (response.choices[0].message.content or "").strip()
            reply = reply.replace("*", "").replace("#", "").replace("`", "")
            return reply or "Let's keep our conversation focused on the technical interview."
        except Exception as e:
            logger.error("Error generating warning reply: %s", e)
            return "Please keep your responses focused on the technical interview."

    def _speak_and_log(self, text: str) -> None:
        """
        Helper to print, log, and speak reply via TTS.
        """
        formatted_response = (
            f"\n{'*' * 65}\n"
            f" 🤖 AGENT SPOKEN RESPONSE (Session: {self._session_id})\n"
            f" 🗣️ Response: \"{text}\"\n"
            f"{'*' * 65}\n"
        )
        print(formatted_response, flush=True)
        logger.info("[AGENT RESPONSE] session=%s: %s", self._session_id, text)
        self.session.say(text)


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

    candidate_name = context.get("candidateName") or "Candidate"
    job_role = context.get("jobTitle") or context.get("jobRole") or "Software Engineer"
    logger.info("Loaded session %s context: candidate='%s', jobRole='%s'", session_id, candidate_name, job_role)

    # Format system prompt from SYSTEM_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        candidate_name=candidate_name,
        job_role=job_role,
    )

    # Initialize TTS using LiveKit Cloud Inference (Fish Audio model: fishaudio/s2.1-pro-free)
    if os.getenv("FISH_API_KEY"):
        logger.info("Using direct Fish Audio TTS plugin (voice: b347db033a6549378b48d00acb0d06cd)")
        tts_service = fishaudio.TTS(
            voice_id="b347db033a6549378b48d00acb0d06cd",
            speed=1.5,
            volume=1.2,
        )
    else:
        logger.info("Using LiveKit Cloud Inference Fish Audio TTS (model: fishaudio/s2.1-pro-free)")
        tts_service = inference.TTS(
            model="fishaudio/s2.1-pro-free",
            voice="b347db033a6549378b48d00acb0d06cd",
            language="en",
            extra_kwargs={"speed": 1.5, "volume": 1.2},
        )

    # Instantiate bounded working memory
    memory = ConversationMemory(window_size=5)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3"),
        tts=tts_service,
        userdata={
            "session_id": session_id,
            "interview_context": context,
            "memory": memory,
            "warning_count": 0,
        },
    )

    # Instantiate custom InterviewAgent
    agent = InterviewAgent(
        session_id=session_id,
        context=context,
        system_prompt=system_prompt,
        memory=memory,
        room=ctx.room,
    )

    # Event handler for participant disconnection (end of interview signaling)
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant %s disconnected from room %s.", participant.identity, ctx.room.name)
        asyncio.create_task(agent.end_interview_flow("candidate_disconnected"))

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info("Session %s is live — VAD + STT + TTS pipeline active.", session_id)

    # Silence-triggered greeting handling: wait 2.5 seconds; if candidate hasn't spoken, greeting fires
    await asyncio.sleep(2.5)
    if not agent.has_candidate_spoken and not agent._is_ending:
        logger.info("Candidate silent for 2.5s post-start. Generating first greeting...")
        greeting = await agent.generate_greeting()
        agent._speak_and_log(greeting)


if __name__ == "__main__":
    agent_name = os.getenv("AGENT_NAME") or "interview-agent"
    logger.info("Starting LiveKit worker registered as agent_name: '%s' (explicit dispatch enabled)", agent_name)
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        )
    )
