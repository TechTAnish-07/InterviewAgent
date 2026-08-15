import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterable

import numpy as np

import aiohttp
import litellm
from dotenv import load_dotenv

from livekit import agents, api, rtc
from livekit.agents import Agent, AgentSession, InterruptionOptions, ModelSettings, TurnHandlingOptions, inference
from livekit.agents import tts as agents_tts
from livekit.agents import llm
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins import fishaudio, groq, openai, silero


async def _stream_llm_response(
    response_stream: Any,
    full_text_container: list[str],
    tool_calls_container: list[dict],
) -> AsyncIterable[str]:
    """
    Consumes LiteLLM streaming completion.
    Yields word-buffered text chunks for sentence-level TTS streaming.
    Buffering to word (whitespace) boundaries prevents TTS from re-initializing
    its vocoder mid-phoneme, which was the cause of voice fluctuation/pitch shifts.
    Accumulates full text into full_text_container[0] and tool calls into tool_calls_container.
    """
    accumulated_text: list[str] = []
    accumulated_tool_calls: dict[int, dict] = {}
    word_buffer: str = ""
    try:
        async for chunk in response_stream:
            if not chunk or not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                content_clean = content.replace("*", "").replace("#", "").replace("`", "")
                if content_clean:
                    accumulated_text.append(content_clean)
                    word_buffer += content_clean
                    # Only yield when we have a complete word boundary (space/newline)
                    # This avoids mid-phoneme TTS synthesis which causes pitch fluctuation
                    if " " in word_buffer or "\n" in word_buffer:
                        parts = word_buffer.rsplit(" ", 1)
                        to_yield = parts[0] + " "
                        word_buffer = parts[1] if len(parts) > 1 else ""
                        yield to_yield

            t_calls = getattr(delta, "tool_calls", None)
            if t_calls:
                for tc in t_calls:
                    idx = getattr(tc, "index", 0) or 0
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {"name": "", "arguments": ""}
                    if hasattr(tc, "function") and tc.function:
                        if getattr(tc.function, "name", None):
                            accumulated_tool_calls[idx]["name"] += tc.function.name
                        if getattr(tc.function, "arguments", None):
                            accumulated_tool_calls[idx]["arguments"] += tc.function.arguments
    except Exception as e:
        logger.error("Error consuming LLM response stream: %s", e)

    # Flush remaining buffered text
    if word_buffer.strip():
        yield word_buffer

    full_text = "".join(accumulated_text).strip()
    # NOTE: If model only emitted tool calls (no text), full_text will be empty — that's fine.
    # Do NOT inject a fallback phrase here; the tool call handler or caller is responsible for
    # speaking any required follow-up. Injecting text here causes repetitive questions when
    # the model legitimately chose to call a tool without speaking.

    full_text_container.append(full_text)
    if accumulated_tool_calls:
        tool_calls_container.extend(accumulated_tool_calls.values())

from memory import ConversationMemory
from moderation import check_message_relevance
from questions import get_question_bank_prompt
from prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    GREETING_INSTRUCTION,
    WRAP_UP_INSTRUCTION,
    OFF_TOPIC_WARNING_INSTRUCTION,
    INAPPROPRIATE_WARNING_INSTRUCTION,
    FINAL_WARNING_INSTRUCTION,
    FEEDBACK_GENERATION_PROMPT,
    STATIC_OFF_TOPIC_WARNING,
    STATIC_INAPPROPRIATE_WARNING,
    STATIC_FINAL_WARNING,
)
from tools import (
    get_resume_context,
    end_interview,
    repeat_last_response,
    show_coding_question,
    get_current_code,
    run_code_check,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8080"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or "internal-secret-key"
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini/gemini-3.5-flash"

# Time and turn safety net limits (dynamic calculation based on total interview minutes)
DEFAULT_MAX_INTERVIEW_MINUTES: int = int(os.getenv("MAX_INTERVIEW_MINUTES", os.getenv("HARD_TIME_LIMIT_SECONDS", 1800) // 60))  # Default 30 minutes
DEFAULT_MAX_TIME_SECONDS: int = DEFAULT_MAX_INTERVIEW_MINUTES * 60
WARNING_WINDOW_SECONDS: int = int(os.getenv("WARNING_WINDOW_SECONDS", 300))  # 5 minutes before max time
HARD_TIME_LIMIT_SECONDS: int = DEFAULT_MAX_TIME_SECONDS
SOFT_TIME_LIMIT_SECONDS: int = max(0, DEFAULT_MAX_TIME_SECONDS - WARNING_WINDOW_SECONDS)
MAX_TURN_COUNT: int = int(os.getenv("MAX_TURN_COUNT", 40))                      # 40 turns max
GREETING_SILENCE_DELAY_SECONDS: float = float(os.getenv("GREETING_SILENCE_DELAY_SECONDS", 4.0))  # Seconds of silence before greeting fires

# Phrases that signal the candidate wants the agent to repeat its last response.
# Checked as substrings — add new variants here; order doesn't matter.
REPEAT_INTENT_TRIGGERS: frozenset[str] = frozenset({
    "repeat",           # "please repeat", "can you repeat", "repeat that"
    "say that again",   # "can you say that again"
    "say it again",     # "say it again please"
    "didn't catch",     # "I didn't catch that"
    "didn't hear",      # "I didn't hear you"
    "can't hear",       # "I can't hear you"
    "couldn't hear",    # "couldn't hear you"
    "can you say",      # "can you say that again"
    "could you say",    # "could you say that one more time"
    "come again",       # "come again?"
    "pardon",           # "pardon me?"
    "what did you say", # "what did you say?"
    "what was that",    # "what was that?"
    "one more time",    # "say that one more time"
    "once more",        # "once more please"
    "run that by me",   # "run that by me again"
    "say again",        # radio-style "say again"
    "didn't understand",# "I didn't understand"
    "didn't get that",  # "I didn't get that"
    "not hear",         # "could not hear"
    "not understand",   # "could not understand"
})

# Phrases that signal the candidate explicitly wants to end or conclude the interview/session.
# Fast-path: handled immediately with a warm closing message and room termination.
END_INTENT_TRIGGERS: frozenset[str] = frozenset({
    # Interview variants
    "end interview",
    "end the interview",
    "end this interview",
    "stop interview",
    "stop the interview",
    "finish interview",
    "finish the interview",
    "finish this interview",
    "conclude interview",
    "conclude the interview",
    "wrap up the interview",
    "wrap up interview",
    "terminate interview",
    "terminate the interview",
    "leave interview",
    "leave the interview",
    "exit interview",
    "exit the interview",
    "quit interview",
    "quit the interview",
    "done with the interview",
    "done with interview",
    # Session variants
    "end session",
    "end the session",
    "end this session",
    "stop session",
    "stop the session",
    "stop this session",
    "finish session",
    "finish the session",
    "finish this session",
    "conclude session",
    "conclude the session",
    "conclude this session",
    "wrap up session",
    "wrap up the session",
    "wrap up this session",
    "close session",
    "close the session",
    "close this session",
    "done with session",
    "done with the session",
    "finalize session",
    "finalize this session",
    # Call / conversation variants
    "end call",
    "end the call",
    "end this call",
    "hang up",
    "end conversation",
    "end the conversation",
    "end this conversation",
    # Intent phrases
    "i want to end",
    "i would like to end",
    "can we end",
    "can we wrap up",
    "can we finish",
    "let's end here",
    "let's stop here",
    "let's finish here",
    "let's wrap up",
    "wrap up here",
})


SESSION_CONTEXT_CACHE: dict[str, dict] = {}


def log_llm_cost(call_site: str, model: str, response) -> None:
    """
    Print token usage and estimated USD cost for every LiteLLM completion call.
    Uses litellm.completion_cost() for pricing (supports Gemini, OpenAI, etc.).
    """
    try:
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        try:
            cost_usd = litellm.completion_cost(completion_response=response)
        except Exception:
            cost_usd = 0.0
        print(
            f"\n{'\u2500'*65}\n"
            f"  \U0001f4b0 LLM COST LOG  [{call_site}]\n"
            f"  Model      : {model}\n"
            f"  Prompt     : {prompt_tokens:,} tokens\n"
            f"  Completion : {completion_tokens:,} tokens\n"
            f"  Total      : {total_tokens:,} tokens\n"
            f"  Estimated  : ${cost_usd:.6f} USD\n"
            f"{'\u2500'*65}\n",
            flush=True,
        )
        logger.info(
            "[COST] %s | model=%s prompt=%d completion=%d total=%d cost=$%.6f",
            call_site, model, prompt_tokens, completion_tokens, total_tokens, cost_usd,
        )
    except Exception as e:
        logger.warning("Cost logging failed for %s: %s", call_site, e)


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


def extract_context_from_metadata(metadata_raw: str | None) -> dict | None:
    """
    Extract pre-packaged candidate context directly from job metadata JSON string without HTTP/DB lookups.
    """
    if not metadata_raw or not metadata_raw.strip():
        return None
    try:
        data = json.loads(metadata_raw)
        if isinstance(data, dict):
            c_name = data.get("candidateName") or data.get("candidate_name")
            j_title = data.get("jobTitle") or data.get("job_title") or data.get("jobRole")
            if c_name and j_title:
                duration_val = data.get("durationMinutes") or data.get("duration_minutes") or data.get("maxInterviewMinutes") or data.get("max_interview_minutes")
                return {
                    "candidateName": c_name,
                    "jobTitle": j_title,
                    "jobRole": j_title,
                    "summary": data.get("summary") or "",
                    "skills": data.get("skills") or "[]",
                    "resumeText": data.get("resumeText") or data.get("resume_text") or "",
                    "durationMinutes": duration_val,
                }
    except Exception:
        pass
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
        {"role": "system", "content": "You are an expert technical interviewer. Write detailed, structured, honest, and actionable post-interview feedback reports. Always use the exact section headers requested. Be specific — reference what the candidate actually said."},
        {"role": "user", "content": prompt},
    ]

    try:
        logger.info("Generating feedback report for session %s...", session_id)
        response = litellm.completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_tokens=1200,
        )
        log_llm_cost("FEEDBACK_GEN", MODEL_NAME, response)
        feedback_text = response.choices[0].message.content.strip()
        logger.info("Feedback report generated for session %s. Saving to Spring Boot...", session_id)
        await send_feedback_report(session_id, feedback_text)
    except Exception as e:
        logger.error("Error generating feedback report for session %s: %s", session_id, e)


# Software PCM gain applied post-TTS synthesis, before WebRTC dispatch.
# Fish Audio TTS volume param is clamped server-side, so we amplify here instead.
# 2.5 ≈ +8 dB, safe for typical speech signals at Fish Audio output levels.
SOFTWARE_AUDIO_GAIN: float = 2.5


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
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._interview_start_time: float = time.time()
        self._soft_warning_sent: bool = False
        self._turn_count: int = 0
        self._is_processing_turn: bool = False

        # Calculate max duration and 5-minute warning threshold
        duration_min = None
        if context:
            duration_min = context.get("durationMinutes") or context.get("duration_minutes") or context.get("maxInterviewMinutes")
        if duration_min is not None:
            try:
                self._max_time_seconds: int = int(duration_min) * 60
            except (ValueError, TypeError):
                self._max_time_seconds = DEFAULT_MAX_TIME_SECONDS
        else:
            self._max_time_seconds = DEFAULT_MAX_TIME_SECONDS

        self._warning_time_seconds: int = max(0, self._max_time_seconds - WARNING_WINDOW_SECONDS)

        self._current_speech_handle: SpeechHandle | None = None
        self._current_turn_task: asyncio.Task | None = None
        self._greeting_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def _safe_interrupt(self) -> None:
        """Safely interrupt the current speech session without throwing if uninitialized."""
        try:
            if hasattr(self, "session") and self.session is not None:
                self.session.interrupt()
        except Exception as e:
            logger.debug("Interrupt ignored (session not ready or already completed): %s", e)

    def _record_warning(self, classification: str) -> int:
        """Atomically increment warning count and log escalation."""
        self._warning_count += 1
        logger.warning(
            "[WARNING] session=%s warning_count=%d class=%s",
            self._session_id,
            self._warning_count,
            classification,
        )
        return self._warning_count

    def _get_tool_context(self) -> dict:
        """Helper to return current session userdata context for tools."""
        if hasattr(self, "session") and self.session is not None and getattr(self.session, "userdata", None):
            self.session.userdata["turn_count"] = self._turn_count
            self.session.userdata["soft_warning_sent"] = self._soft_warning_sent
            self.session.userdata["interview_start_time"] = self._interview_start_time
            self.session.userdata["max_time_seconds"] = self._max_time_seconds
            self.session.userdata["warning_time_seconds"] = self._warning_time_seconds
            return self.session.userdata
        return {
            "session_id": self._session_id,
            "interview_context": self._context,
            "memory": self._memory,
            "room": self._room,
            "latest_code": "",
            "last_run_result": None,
            "current_question": None,
            "interview_start_time": self._interview_start_time,
            "max_time_seconds": self._max_time_seconds,
            "warning_time_seconds": self._warning_time_seconds,
            "soft_warning_sent": self._soft_warning_sent,
            "turn_count": self._turn_count,
        }

    async def end_interview_flow(self, reason: str) -> None:
        """
        Execute session end:
        1. Broadcast INTERVIEW_ENDED data event to room
        2. Generate feedback report and save to Spring Boot
        3. Notify Spring Boot of session end
        4. Delete LiveKit room on server (closes room for all participants including candidate)
        5. Disconnect agent WebRTC connection
        """
        if self._is_ending:
            return
        self._is_ending = True
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        logger.info("Executing end_interview_flow for session %s (reason: %s)...", self._session_id, reason)

        # Print cumulative cost summary for entire session
        self.log_session_cost_summary()

        # Broadcast interview end data event to frontend so client UI can immediately transition
        try:
            if self._room and hasattr(self._room, "local_participant") and self._room.local_participant:
                end_payload = json.dumps({
                    "type": "INTERVIEW_ENDED",
                    "sessionId": self._session_id,
                    "reason": reason,
                })
                await self._room.local_participant.publish_data(end_payload, reliable=True, topic="interview_events")
                logger.info("Broadcasted INTERVIEW_ENDED data event for session %s", self._session_id)
        except Exception as e:
            logger.warning("Error broadcasting INTERVIEW_ENDED event for session %s: %s", self._session_id, e)

        # Generate feedback report and save to Spring Boot
        await generate_and_save_feedback(self._session_id, self._context, self._memory, reason)

        # Notify Spring Boot of session end
        await notify_session_end(self._session_id, reason)

        # Delete LiveKit room on server to cleanly close room for all participants (candidate + agent)
        room_name = getattr(self._room, "name", None)
        if room_name and not isinstance(room_name, type(None)):
            livekit_url = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
            api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
            api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
            http_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
            try:
                lk_api = api.LiveKitAPI(http_url, api_key, api_secret)
                await lk_api.room.delete_room(api.DeleteRoomRequest(room=str(room_name)))
                await lk_api.aclose()
                logger.info("Successfully deleted LiveKit room %s on server for session %s", room_name, self._session_id)
            except Exception as e:
                logger.warning("Error deleting LiveKit room %s on server: %s", room_name, e)

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
        Cancels any prior in-flight turn task to avoid race conditions.
        """
        if self._is_ending:
            return

        # Candidate spoke; cancel silence greeting task if still scheduled
        self.has_candidate_spoken = True
        if self._greeting_task and not self._greeting_task.done():
            self._greeting_task.cancel()

        transcript = new_message.text_content or ""
        if not transcript.strip():
            return

        handle_transcript(self._session_id, transcript)

        # Cancel any in-flight turn processing task to prevent overlapping LLM completions / say() calls
        if self._current_turn_task and not self._current_turn_task.done():
            logger.info("[TURN CANCEL] Cancelling previous turn processing task for session %s", self._session_id)
            self._current_turn_task.cancel()

        # Interrupt any speech handle currently playing
        if self._current_speech_handle and not self._current_speech_handle.done():
            self._safe_interrupt()

        self._current_turn_task = asyncio.create_task(
            self._process_user_turn(transcript, turn_ctx)
        )

    async def _process_user_turn(
        self, transcript: str, turn_ctx: llm.ChatContext
    ) -> None:
        """
        Processes a committed user turn: early intent checks, moderation, LLM streaming, and tool calls.
        Tracks _is_processing_turn so the time watchdog never interrupts an in-flight conversation turn.
        """
        self._is_processing_turn = True
        # Turn count increment and safety net limits check
        self._turn_count += 1
        elapsed = time.time() - self._interview_start_time
        logger.info("[TURN %d] session=%s elapsed=%.1fs (max=%ds, warn_at=%ds)", self._turn_count, self._session_id, elapsed, self._max_time_seconds, self._warning_time_seconds)

        # 0. Hard limits check (Turn count & Time limit) — forced auto-end, bypassing model
        if self._turn_count >= MAX_TURN_COUNT:
            logger.warning("[MAX TURNS] Session %s reached max turn count (%d >= %d). Ending interview.", self._session_id, self._turn_count, MAX_TURN_COUNT)
            closing_msg = (
                "Thank you for your time today. That concludes our interview session, "
                "and your feedback report is now being prepared. Best of luck!"
            )
            handle = self._speak_and_log(closing_msg, candidate_text=transcript)
            try:
                await handle.wait_for_playout()
            except Exception:
                pass
            await self.end_interview_flow("max_turns_reached")
            return

        if elapsed >= self._max_time_seconds:
            logger.warning("[MAX TIME REACHED] Session %s reached max duration (%.1fs >= %ds). Ending interview smoothly.", self._session_id, elapsed, self._max_time_seconds)
            closing_msg = (
                "Thank you for your time today. We have reached the allotted time for this interview session, "
                "and your feedback report is now being prepared. Best of luck!"
            )
            handle = self._speak_and_log(closing_msg, candidate_text=transcript)
            try:
                await handle.wait_for_playout()
            except Exception:
                pass
            await self.end_interview_flow("time_limit_reached")
            return

        # 1. Early repeat detection — handle BEFORE moderation/LLM to be instant and zero-cost.
        # REPEAT_INTENT_TRIGGERS is a frozenset of substrings; any match triggers replay.
        # If the candidate uses a phrase NOT in the list, the LLM still has the
        # repeat_last_response tool as a fallback to catch intent via semantic understanding.
        if any(trigger in transcript.lower() for trigger in REPEAT_INTENT_TRIGGERS):
            last_reply = self._memory.get_last_response()
            if last_reply:
                logger.info("[REPEAT] Candidate asked to repeat. Replaying last response (%d chars) for session %s.", len(last_reply), self._session_id)
                # Pass candidate_text so this turn is properly logged in memory
                self._speak_and_log(last_reply, candidate_text=transcript)
                return
            else:
                logger.warning("[REPEAT] No prior response in memory for session %s — continuing to LLM.", self._session_id)
                # Fall through to normal LLM flow if nothing to repeat yet

        # 2. Early end-intent detection — fast-path for candidate asking to conclude/end the interview
        if any(trigger in transcript.lower() for trigger in END_INTENT_TRIGGERS):
            logger.info("[END INTENT] Candidate requested to end interview for session %s: %r", self._session_id, transcript)
            closing_msg = (
                "Thank you for taking the time to speak with me today. That concludes our interview session, "
                "and your feedback report is now being prepared. Best of luck!"
            )
            handle = self._speak_and_log(closing_msg, candidate_text=transcript)
            try:
                await handle.wait_for_playout()
            except Exception:
                pass
            await self.end_interview_flow("candidate_requested_end")
            return

        # 2.5 Early question re-display detection — fast-path for candidate asking where the question is
        QUESTION_QUERY_TRIGGERS = (
            "where is the question", "where is the problem", "can't find the question",
            "cannot find the question", "can't see the question", "cannot see the question",
            "share the question", "show the question", "show me the question",
            "send the question", "what is the question", "what's the question",
            "not able to see", "unable to see", "where question"
        )
        if any(trig in transcript.lower() for trig in QUESTION_QUERY_TRIGGERS):
            from questions import ALL_QUESTIONS
            ctx_dict = self._get_tool_context()
            current_q = ctx_dict.get("current_question") if ctx_dict else None
            if not current_q:
                current_q = ALL_QUESTIONS["E1"]["body"]

            logger.info("[QUESTION RESEND] Candidate asked for question. Re-publishing for session %s.", self._session_id)
            await show_coding_question(question_text=current_q, context=ctx_dict)
            reassurance_msg = (
                "I've displayed the coding problem on your screen in the code editor panel. "
                "Please take a look and let me know when you are ready to begin."
            )
            self._speak_and_log(reassurance_msg, candidate_text=transcript)
            return

        classification = check_message_relevance(transcript, self._memory.recent_turns)
        logger.info("[MODERATION] session=%s classification=%s text=%r", self._session_id, classification, transcript)

        # 3. Moderation Escalation Handling
        if classification in ("OFF_TOPIC", "INAPPROAPPROPRIATE", "INAPPROPRIATE"):
            warning_count = self._record_warning(classification)

            if warning_count == 1:
                instruction = (
                    OFF_TOPIC_WARNING_INSTRUCTION
                    if classification == "OFF_TOPIC"
                    else INAPPROPRIATE_WARNING_INSTRUCTION
                )
                warning_reply = self._generate_warning_reply(transcript, instruction)
                self._speak_and_log(warning_reply, candidate_text=transcript)
                return
            elif warning_count == 2:
                warning_reply = self._generate_warning_reply(transcript, FINAL_WARNING_INSTRUCTION)
                self._speak_and_log(warning_reply, candidate_text=transcript)
                return
            else:
                # warning_count >= 3: Auto-end for policy violation
                logger.error("[POLICY VIOLATION] Session %s reached max warnings (%d). Ending interview.", self._session_id, warning_count)
                await self.end_interview_flow("policy_violation")
                return

        # 4. Normal Reply Flow
        system_content = self._system_prompt
        time_remaining = self._max_time_seconds - elapsed
        if elapsed >= self._warning_time_seconds and not self._soft_warning_sent:
            self._soft_warning_sent = True
            remaining_mins = max(1, int(round(time_remaining / 60)))
            system_content += f"\n\n[TIME LIMIT APPROACHING: ~{remaining_mins} MINUTES REMAINING]\n{WRAP_UP_INSTRUCTION}"
            logger.info(
                "[TIME WARNING] Injected WRAP_UP_INSTRUCTION into system prompt for session %s (elapsed=%.1fs, remaining=%.1fs <= %ds)",
                self._session_id,
                elapsed,
                time_remaining,
                WARNING_WINDOW_SECONDS,
            )

        context_messages = self._memory.build_context_messages()
        full_messages = [
            {"role": "system", "content": system_content},
            *context_messages,
            {"role": "user", "content": transcript},
        ]

        try:
            tools_list = [
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
                {
                    "type": "function",
                    "function": {
                        "name": "repeat_last_response",
                        "description": repeat_last_response.info.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Reason for repeating the last response",
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "show_coding_question",
                        "description": show_coding_question.info.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question_text": {
                                    "type": "string",
                                    "description": "The coding question problem statement to display on the candidate's screen.",
                                }
                            },
                            "required": ["question_text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_code",
                        "description": get_current_code.info.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_code_check",
                        "description": run_code_check.info.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "language": {
                                    "type": "string",
                                    "description": "The programming language of the code (e.g. python, javascript, java, cpp)",
                                },
                                "stdin": {
                                    "type": "string",
                                    "description": "Optional standard input to feed to the program",
                                },
                            },
                            "required": ["language"],
                        },
                    },
                },
            ]

            response = await litellm.acompletion(
                model=MODEL_NAME,
                messages=full_messages,
                temperature=0.7,
                max_tokens=350,
                stream=True,
                tools=tools_list,
            )

            full_text_container: list[str] = []
            tool_calls_container: list[dict] = []
            text_stream = _stream_llm_response(response, full_text_container, tool_calls_container)

            # Stream LLM text chunks sentence-by-sentence to TTS via LiveKit session.say()
            handle = self._speak_and_log(
                text_stream,
                candidate_text=transcript,
                full_generated_text_container=full_text_container,
            )

            # Await playout of generated response before evaluating tool calls
            try:
                await handle.wait_for_playout()
            except Exception:
                pass

            for tc in tool_calls_container:
                func_name = tc.get("name", "")

                if func_name == "end_interview":
                    args_raw = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {}
                    reason = args.get("reason", "interview_complete")
                    logger.info("[TOOL CALL] Model invoked end_interview with reason: %s", reason)
                    # If model didn't speak a closing message (empty text), speak one
                    if not full_text_container or not full_text_container[0].strip():
                        closing_msg = (
                            "Thank you for your time today. That concludes our interview session, "
                            "and your feedback report is now being prepared. Best of luck!"
                        )
                        h = self._speak_and_log(closing_msg)
                        try:
                            await h.wait_for_playout()
                        except Exception:
                            pass
                    await self.end_interview_flow(reason)
                    return

                if func_name == "repeat_last_response":
                    logger.info("[TOOL CALL] Model invoked repeat_last_response for session %s", self._session_id)
                    last_reply = self._memory.get_last_response()
                    if last_reply:
                        logger.info("[REPEAT] Replaying last agent response (%d chars) for session %s", len(last_reply), self._session_id)
                        # Speak directly — no new LLM call, zero cost
                        self._speak_and_log(last_reply)
                    else:
                        logger.warning("[REPEAT] No prior response found in memory for session %s", self._session_id)
                        self._speak_and_log("I'm sorry, I don't have a previous response to repeat. Could you let me know what you'd like me to clarify?")
                    return

                if func_name == "show_coding_question":
                    args_raw = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {}
                    question_text = args.get("question_text", "").strip()

                    from questions import ALL_QUESTIONS
                    # If question_text is an ID like "E1", "M1", "H2", or title, resolve from ALL_QUESTIONS
                    if question_text in ALL_QUESTIONS:
                        question_text = ALL_QUESTIONS[question_text]["body"]
                    else:
                        for q_info in ALL_QUESTIONS.values():
                            if q_info["title"].lower() in question_text.lower():
                                question_text = q_info["body"]
                                break

                    # Fallback to spoken text if available
                    if not question_text and full_text_container and full_text_container[0].strip():
                        spoken = full_text_container[0].strip()
                        if len(spoken) > 80 or "\n" in spoken:
                            question_text = spoken

                    # Final fallback: Two Sum (E1)
                    if not question_text:
                        question_text = ALL_QUESTIONS["E1"]["body"]

                    logger.info("[TOOL CALL] Model invoked show_coding_question for session %s: %r", self._session_id, question_text[:80])
                    await show_coding_question(question_text=question_text, context=self._get_tool_context())

                    # If model didn't speak a transition message, announce the question aloud
                    if not full_text_container or not full_text_container[0].strip():
                        transition_msg = (
                            "I have displayed the coding question on your screen. "
                            "Take your time to read through it, and write out your solution in the code editor."
                        )
                        h = self._speak_and_log(transition_msg)
                        try:
                            await h.wait_for_playout()
                        except Exception:
                            pass

                if func_name in ("get_current_code", "run_code_check"):
                    logger.info("[TOOL CALL] Model invoked %s for session %s", func_name, self._session_id)
                    tool_result = ""
                    if func_name == "get_current_code":
                        tool_result = await get_current_code(context=self._get_tool_context())
                    elif func_name == "run_code_check":
                        args_raw = tc.get("arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except Exception:
                            args = {}
                        lang = args.get("language", "python")
                        stdin_val = args.get("stdin", "")
                        tool_result = await run_code_check(language=lang, stdin=stdin_val, context=self._get_tool_context())

                    # If model didn't speak a full evaluation (e.g. emitted only tool calls or minimal filler),
                    # perform follow-up completion with tool output so it speaks its evaluation
                    if not full_text_container or not full_text_container[0].strip() or len(full_text_container[0].strip()) < 25:
                        follow_up_messages = [
                            *full_messages,
                            {
                                "role": "assistant",
                                "content": full_text_container[0] if full_text_container and full_text_container[0].strip() else None,
                                "tool_calls": [
                                    {
                                        "id": tc.get("id") or f"call_{func_name}",
                                        "type": "function",
                                        "function": {
                                            "name": func_name,
                                            "arguments": tc.get("arguments", "{}") if isinstance(tc.get("arguments"), str) else json.dumps(tc.get("arguments", {})),
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id") or f"call_{func_name}",
                                "content": tool_result,
                            },
                        ]
                        fu_resp = await litellm.acompletion(
                            model=MODEL_NAME,
                            messages=follow_up_messages,
                            temperature=0.7,
                            max_tokens=350,
                            stream=True,
                            tools=tools_list,
                        )
                        fu_full_text: list[str] = []
                        fu_tool_calls: list[dict] = []
                        fu_stream = _stream_llm_response(fu_resp, fu_full_text, fu_tool_calls)
                        fu_handle = self._speak_and_log(
                            fu_stream,
                            candidate_text=transcript,
                            full_generated_text_container=fu_full_text,
                        )
                        try:
                            await fu_handle.wait_for_playout()
                        except Exception:
                            pass

        except asyncio.CancelledError:
            logger.info("[TURN CANCELLED] Turn processing task was cancelled for session %s.", self._session_id)
            raise
        except Exception as e:
            logger.error("Error in normal reply flow for session %s: %s", self._session_id, e)
            # Do NOT speak a canned background question as fallback — it causes the agent to
            # keep asking the same question repeatedly when errors occur. Silently fail instead
            # so the candidate can just speak again and trigger a fresh turn naturally.
        finally:
            self._is_processing_turn = False

    def _generate_warning_reply(self, candidate_text: str, instruction_prompt: str) -> str:
        """
        Return static warning redirect — zero LLM API cost.
        Uses pre-written templates from prompts.py instead of an LLM call.
        instruction_prompt is kept as a parameter for API compatibility but is no longer used.
        """
        text_lower = candidate_text.lower()
        from prompts import STATIC_OFF_TOPIC_WARNING, STATIC_INAPPROPRIATE_WARNING, STATIC_FINAL_WARNING

        if self._warning_count >= 2:
            return STATIC_FINAL_WARNING
        elif any(w in text_lower for w in ("fuck", "shit", "bitch", "bastard", "idiot", "hate", "shut up")):
            return STATIC_INAPPROPRIATE_WARNING
        else:
            return STATIC_OFF_TOPIC_WARNING

    def _accumulate_cost(self, response) -> None:
        """Accumulate token usage and cost across all LLM calls for this session."""
        try:
            usage = response.usage
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self._total_cost_usd += litellm.completion_cost(completion_response=response)
        except Exception:
            pass

    def log_session_cost_summary(self) -> None:
        """Print cumulative token usage and cost for the entire session."""
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens
        print(
            f"\n{'\u2550'*65}\n"
            f"  \U0001f4ca SESSION COST SUMMARY (Session: {self._session_id})\n"
            f"  Total Prompt Tokens     : {self._total_prompt_tokens:,}\n"
            f"  Total Completion Tokens : {self._total_completion_tokens:,}\n"
            f"  Total Tokens            : {total_tokens:,}\n"
            f"  Total Estimated Cost    : ${self._total_cost_usd:.6f} USD\n"
            f"{'\u2550'*65}\n",
            flush=True,
        )
        logger.info(
            "[SESSION COST] session=%s prompt=%d completion=%d total=%d cost=$%.6f",
            self._session_id, self._total_prompt_tokens, self._total_completion_tokens,
            total_tokens, self._total_cost_usd,
        )

    def _speak_and_log(
        self,
        text: str | AsyncIterable[str],
        candidate_text: str | None = None,
        full_generated_text_container: list[str] | None = None,
    ) -> SpeechHandle:
        """
        Helper to print, log, and speak reply via TTS.
        Supports both plain string text and streaming AsyncIterable[str] text streams.
        Attaches a done callback to track speech completion, interruption, and memory updates.
        Ensures active speech handles are managed safely without race conditions.
        """
        # Interrupt any previous speech handle that is still active
        if self._current_speech_handle is not None and not self._current_speech_handle.done():
            self._safe_interrupt()

        if isinstance(text, str):
            formatted_response = (
                f"\n{'*' * 65}\n"
                f" 🤖 AGENT SPOKEN RESPONSE (Session: {self._session_id})\n"
                f" 🗣️ Response: \"{text}\"\n"
                f"{'*' * 65}\n"
            )
            print(formatted_response, flush=True)
            logger.info("[AGENT RESPONSE] session=%s: %s", self._session_id, text)
        else:
            logger.info("[AGENT RESPONSE STREAM] session=%s: Initiated LLM-to-TTS sentence streaming.", self._session_id)

        handle = self.session.say(text)
        self._current_speech_handle = handle

        committed = False

        def _on_speech_done(h: SpeechHandle) -> None:
            nonlocal committed
            if committed:
                return
            committed = True

            if self._current_speech_handle is h:
                self._current_speech_handle = None

            full_gen_text = full_generated_text_container[0] if (full_generated_text_container and full_generated_text_container) else None
            actually_spoken = full_gen_text or ""
            if h.chat_items and h.chat_items[0].text_content:
                actually_spoken = h.chat_items[0].text_content
            elif isinstance(text, str):
                actually_spoken = text

            total_text = full_gen_text or (text if isinstance(text, str) else actually_spoken)

            if h.interrupted:
                logger.warning(
                    "[INTERRUPTION DETECTED] session=%s: Agent reply interrupted mid-sentence. "
                    "Delivered %d/%d chars before cutoff: %r",
                    self._session_id,
                    len(actually_spoken),
                    len(total_text),
                    actually_spoken,
                )
            else:
                logger.info("[SPEECH COMPLETED] session=%s: Full reply delivered (%d chars)", self._session_id, len(actually_spoken))

            if candidate_text is not None:
                summary_before = self._memory.rolling_summary
                self._memory.add_turn(candidate_text, actually_spoken)
                summary_updated = self._memory.rolling_summary != summary_before
                logger.info(
                    "[TURN LOG] session=%s summary_updated=%s candidate=%r reply=%r interrupted=%s",
                    self._session_id,
                    summary_updated,
                    candidate_text,
                    actually_spoken,
                    h.interrupted,
                )

        handle.add_done_callback(_on_speech_done)
        return handle

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Override Agent.tts_node to apply a software PCM amplitude gain after TTS synthesis.
        Fish Audio's 'volume' param is clamped server-side; boosting here is independent.
        """
        async for frame in Agent.default.tts_node(self, text, model_settings):
            # frame.data is a memoryview of int16 samples
            arr = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
            arr = np.clip(arr * SOFTWARE_AUDIO_GAIN, -32768.0, 32767.0)
            amplified = arr.astype(np.int16).tobytes()
            yield rtc.AudioFrame(
                data=amplified,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
            )


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

    # 1. Check in-memory SESSION_CONTEXT_CACHE first
    context = SESSION_CONTEXT_CACHE.get(str(session_id))

    # 2. Check metadata pre-packaged context if not in cache
    if context is None:
        context = extract_context_from_metadata(ctx.job.metadata)

    # 3. Fallback to Spring Boot HTTP endpoint if not present in metadata or cache
    if context is None:
        logger.info("Fetching interview context via HTTP endpoint for session %s...", session_id)
        context = await fetch_interview_context(session_id)

    if context is None:
        logger.error("Could not obtain interview context for session %s. Disconnecting from room.", session_id)
        await ctx.room.disconnect()
        return

    # Store in SESSION_CONTEXT_CACHE for fast in-memory lookup during call
    SESSION_CONTEXT_CACHE[str(session_id)] = context

    candidate_name = context.get("candidateName") or "Candidate"
    job_role = context.get("jobTitle") or context.get("jobRole") or "Software Engineer"
    summary = context.get("summary")
    skills = context.get("skills")
    logger.info("Loaded session %s context: candidate='%s', jobRole='%s'", session_id, candidate_name, job_role)

    extra_context = ""
    if summary and isinstance(summary, str) and summary.strip():
        extra_context += f"\n\nCandidate Executive Summary: {summary}"
    if skills and skills != "[]":
        extra_context += f"\nVerified Technical Skills: {skills}"

    # Format system prompt from SYSTEM_PROMPT_TEMPLATE + question bank
    # Appending the hardcoded question bank means the LLM always picks from a
    # pre-written, complete question body instead of generating one on the fly.
    # This eliminates the "empty question_text" bug and guarantees well-formed questions.
    system_prompt = (
        SYSTEM_PROMPT_TEMPLATE.format(
            candidate_name=candidate_name,
            job_role=job_role,
        )
        + extra_context
        + get_question_bank_prompt()
    )

    # Initialize TTS using LiveKit Cloud Inference / Fish Audio TTS
    tts_speed: float = float(os.getenv("TTS_SPEED", "1.1"))
    if os.getenv("FISH_API_KEY"):
        logger.info("Using direct Fish Audio TTS plugin (voice: b347db033a6549378b48d00acb0d06cd, speed=%.2f)", tts_speed)
        tts_service = fishaudio.TTS(
            voice_id="b347db033a6549378b48d00acb0d06cd",
            speed=tts_speed,
            volume=1.2,
        )
    else:
        logger.info("Using LiveKit Cloud Inference Fish Audio TTS (model: fishaudio/s2.1-pro-free, speed=%.2f)", tts_speed)
        tts_service = inference.TTS(
            model="fishaudio/s2.1-pro-free",
            voice="b347db033a6549378b48d00acb0d06cd",
            language="en",
            extra_kwargs={"speed": tts_speed, "volume": 1.2},
        )


    # Instantiate bounded working memory with local session log support
    memory = ConversationMemory(window_size=5, session_id=session_id)

    # Configure interruption behavior with tuned thresholds and options
    # - allow_interruptions=True: Candidate can interrupt mid-question.
    # - min_interruption_duration=0.8s: Prevents short pauses/breaths from triggering false interruptions.
    # - min_interruption_words=2: Guards against single-word filler sounds (e.g., "um", "uh").
    # Mode selection: We configure adaptive interruption mode in InterruptionOptions.
    # SDK Note: In livekit-agents v1.6.9, adaptive mode requires STT with streaming aligned transcripts.
    # Since groq.STT(model="whisper-large-v3") does not provide aligned transcripts, the framework
    # will automatically fall back to VAD mode using the tuned thresholds above (0.8s min_duration, 2 min_words).
    turn_handling = TurnHandlingOptions(
        interruption=InterruptionOptions(
            enabled=True,
            mode="adaptive",
            min_duration=0.8,
            min_words=2,
        )
    )

    # Build a Whisper prompt from candidate tech vocabulary to reduce hallucinations.
    # Whisper uses this (up to 224 tokens) to bias recognition toward known terms —
    # e.g. "Next.js" won't be heard as "Next JS" or "3xjs", "LiteLLM" won't be garbled, etc.
    stt_vocab_terms: list[str] = [candidate_name, job_role]
    if skills and isinstance(skills, str) and skills.strip() not in ("", "[]"):
        # skills may be a JSON array string like '["Spring Boot", "React"]' or plain comma-sep
        try:
            import json as _json
            parsed_skills = _json.loads(skills)
            if isinstance(parsed_skills, list):
                stt_vocab_terms.extend([s for s in parsed_skills if isinstance(s, str)])
        except Exception:
            # Fallback: treat as comma-separated string
            stt_vocab_terms.extend([s.strip() for s in skills.split(",") if s.strip()])
    # Add a broad base of common technical terms to cover any gaps
    stt_vocab_terms += [
        "Python", "Java", "Spring Boot", "React", "Next.js", "TypeScript", "JavaScript",
        "Node.js", "Flask", "FastAPI", "LangChain", "LiteLLM", "MCP", "Playwright",
        "LLM", "RAG", "Docker", "Kubernetes", "PostgreSQL", "MongoDB", "REST API",
        "GraphQL", "AWS", "GCP", "Azure", "Git", "GitHub", "CI/CD", "Groq", "Whisper",
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in stt_vocab_terms:
        if t and t not in seen:
            seen.add(t)
            unique_terms.append(t)
    stt_prompt = (
        f"This is a technical software engineering interview with {candidate_name} for a {job_role} role. "
        f"Technical terms that may appear: {', '.join(unique_terms[:40])}."
    )
    logger.info("[STT] Whisper prompt built (%d chars): %r", len(stt_prompt), stt_prompt[:120])

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3", language="en", prompt=stt_prompt),
        tts=tts_service,
        turn_handling=turn_handling,
        userdata={
            "session_id": session_id,
            "interview_context": context,
            "memory": memory,
            "warning_count": 0,
            "latest_code": "",
            "last_run_result": None,
            "current_question": None,
            "interview_start_time": time.time(),
            "soft_warning_sent": False,
            "turn_count": 0,
            "room": ctx.room,
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

    # Event handler for data messages (code editor updates & run results from candidate)
    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket) -> None:
        try:
            payload_str = data_packet.data.decode("utf-8")
            payload = json.loads(payload_str)
            if not isinstance(payload, dict):
                return
            msg_type = payload.get("type")
            if msg_type == "code_update":
                code = payload.get("code", "")
                session.userdata["latest_code"] = code
                logger.info("[DATA CHANNEL] Received code_update (length=%d) for session %s", len(code), session_id)
            elif msg_type == "code_run_result":
                code = payload.get("code", "")
                stdout = payload.get("stdout", "")
                stderr = payload.get("stderr", "")
                status = payload.get("status", "")
                run_result = {
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "status": status,
                }
                session.userdata["last_run_result"] = run_result
                if code:
                    session.userdata["latest_code"] = code
                logger.info("[DATA CHANNEL] Stored last_run_result (status=%s) for session %s", status, session_id)
        except Exception as e:
            logger.warning("[DATA CHANNEL] Error processing data packet for session %s: %s", session_id, e)

    # Event handler for participant disconnection (end of interview signaling)
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant %s disconnected from room %s.", participant.identity, ctx.room.name)
        task = asyncio.create_task(agent.end_interview_flow("candidate_disconnected"))
        agent._background_tasks.add(task)
        task.add_done_callback(agent._background_tasks.discard)

    # Event listeners on AgentSession for real-time VAD & transcript events:
    # Immediately cancel the opening greeting if the candidate starts speaking before the silence timeout.
    # This prevents the glitch where the agent starts speaking greeting right as user finishes speaking,
    # then abruptly interrupts itself.
    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        new_state = getattr(ev, "new_state", None)
        if new_state == "speaking":
            agent.has_candidate_spoken = True
            if agent._greeting_task and not agent._greeting_task.done():
                logger.info("[GREETING CANCEL] Candidate started speaking (user_state=speaking). Cancelling opening greeting.")
                agent._greeting_task.cancel()

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        transcript = getattr(ev, "transcript", "")
        if transcript and transcript.strip():
            agent.has_candidate_spoken = True
            if agent._greeting_task and not agent._greeting_task.done():
                logger.info("[GREETING CANCEL] Candidate speech transcribed (%r). Cancelling opening greeting.", transcript[:40])
                agent._greeting_task.cancel()

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info("Session %s is live — VAD + STT + TTS pipeline active.", session_id)

    # Silence-triggered greeting handling: wait GREETING_SILENCE_DELAY_SECONDS; if candidate hasn't spoken, greeting fires
    async def _greeting_worker() -> None:
        try:
            await asyncio.sleep(GREETING_SILENCE_DELAY_SECONDS)
            if not agent.has_candidate_spoken and not agent._is_ending:
                # Double-check real-time user state right before speaking to prevent race condition
                if getattr(session, "user_state", None) == "speaking":
                    logger.info("[GREETING CANCEL] Candidate is speaking at silence timeout. Aborting auto-greeting.")
                    agent.has_candidate_spoken = True
                    return

                logger.info("Candidate silent for %.1fs post-start. Generating first greeting...", GREETING_SILENCE_DELAY_SECONDS)
                greeting = await agent.generate_greeting()
                if not agent.has_candidate_spoken and not agent._is_ending and getattr(session, "user_state", None) != "speaking":
                    handle = agent._speak_and_log(greeting)

                    # Store the greeting in memory so repeat-detection can find it if the candidate
                    # immediately asks "can you repeat that?" before the first real turn.
                    def _store_greeting(h: SpeechHandle) -> None:
                        spoken = (h.chat_items[0].text_content if h.chat_items else None) or greeting
                        # Use empty string for candidate_text so add_turn stores it but won't
                        # inject it into LLM context as a user message (empty strings are skipped)
                        memory.recent_turns.append(("", spoken))
                        logger.info("[GREETING] Stored opening greeting in memory (%d chars) for session %s.", len(spoken), session_id)

                    handle.add_done_callback(_store_greeting)
        except asyncio.CancelledError:
            logger.info("Greeting task cancelled for session %s (candidate spoke before silence timeout).", session_id)
        except Exception as e:
            logger.warning("Error in greeting task for session %s: %s", session_id, e)

    agent._greeting_task = asyncio.create_task(_greeting_worker())

    # Periodic watchdog task: ensures time limit is enforced even if candidate is silent for long stretches
    async def _time_watchdog() -> None:
        try:
            while not agent._is_ending:
                await asyncio.sleep(5)
                elapsed = time.time() - agent._interview_start_time
                if elapsed >= agent._max_time_seconds and not agent._is_ending:
                    # Avoid race conditions: if candidate turn is actively being processed or agent is actively speaking, wait for that turn to complete cleanly
                    if agent._is_processing_turn or (agent._current_speech_handle and not agent._current_speech_handle.done()):
                        logger.debug("[WATCHDOG] Session %s reached max time (%.1fs >= %ds), waiting for active turn/speech to complete.", session_id, elapsed, agent._max_time_seconds)
                        continue

                    logger.warning("[WATCHDOG] Session %s exceeded max duration during silence/inactivity (%.1fs >= %ds). Concluding interview.", session_id, elapsed, agent._max_time_seconds)
                    closing_msg = (
                        "Thank you for your time today. We have reached the allotted time limit for this interview session, "
                        "so we will conclude here. Your feedback report is now being prepared. Best of luck!"
                    )
                    handle = agent._speak_and_log(closing_msg)
                    try:
                        await handle.wait_for_playout()
                    except Exception:
                        pass
                    await agent.end_interview_flow("time_limit_reached")
                    break
        except asyncio.CancelledError:
            logger.debug("[WATCHDOG] Time watchdog cancelled for session %s", session_id)
        except Exception as e:
            logger.error("[WATCHDOG] Error in time watchdog for session %s: %s", session_id, e)

    agent._watchdog_task = asyncio.create_task(_time_watchdog())
    agent._background_tasks.add(agent._watchdog_task)
    agent._watchdog_task.add_done_callback(agent._background_tasks.discard)


if __name__ == "__main__":
    agent_name = os.getenv("AGENT_NAME") or "interview-agent"
    logger.info("Starting LiveKit worker registered as agent_name: '%s' (explicit dispatch enabled)", agent_name)
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        )
    )
