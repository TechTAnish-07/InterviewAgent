import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
import litellm
from prompts import SUMMARIZE_INSTRUCTION

logger = logging.getLogger("interview-agent.memory")

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"

# Default configuration for live session memory
DEFAULT_WINDOW_SIZE = 10
DEFAULT_BATCH_FOLD_SIZE = 3
DEFAULT_MAX_RECENT_TOKENS = 3000
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL") or "groq/llama-3.1-8b-instant"


class ConversationMemory:
    """
    Bounded working memory for live voice interview conversation.
    - Maintains a sliding window of recent turn pairs (last 10 turns) for LLM completion calls.
    - Summarizes expired turns into rolling_summary using a separate fast Groq model in background tasks.
    - Batches expired turn pairs (3-5 pairs) before summarizing to reduce API overhead.
    - Includes token safety net (default 3000 tokens) to trigger early folds for long answers.
    - Guarded against overlapping background folds.
    - Persists complete un-truncated conversation history locally in conversations/{session_id}.json.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        session_id: str | None = None,
        max_summary_bullets: int = 8,
        batch_fold_size: int = DEFAULT_BATCH_FOLD_SIZE,
        max_recent_tokens: int = DEFAULT_MAX_RECENT_TOKENS,
        summarization_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.recent_turns: list[tuple[str, str]] = []  # [(user_text, agent_text), ...]
        self.full_history: list[dict] = []  # [{turn: 1, timestamp: "...", candidate: "...", interviewer: "..."}, ...]
        self.summary_bullets: list[str] = []  # List of concise turn summaries
        self.pending_fold_buffer: list[tuple[str, str]] = []  # [(user_text, agent_text), ...]
        self.rolling_summary: str = ""
        self.window_size: int = window_size
        self.max_summary_bullets: int = max_summary_bullets
        self.batch_fold_size: int = batch_fold_size
        self.max_recent_tokens: int = max_recent_tokens
        self.session_id: str | None = session_id
        self.summarization_model: str = summarization_model or SUMMARIZATION_MODEL
        self.api_key: str | None = api_key

        self._fold_lock: asyncio.Lock | None = None
        self._is_folding: bool = False

        # Ensure conversations directory exists
        try:
            CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create conversations directory: %s", e)

    def estimate_turns_tokens(self, turns: list[tuple[str, str]] | None = None) -> int:
        """
        Rough estimation of token count for recent turns (~4 characters per token).
        """
        target = turns if turns is not None else self.recent_turns
        total_chars = sum(len(cand or "") + len(agent or "") for cand, agent in target)
        return total_chars // 4

    def add_turn(self, user_text: str, agent_text: str) -> None:
        """
        Append a turn pair (candidate text, agent reply).
        - Appends to sliding window (recent_turns).
        - Checks window size (10) and token safety net (3000 tokens).
        - Moves evicted pairs to pending_fold_buffer.
        - Triggers background batch fold when batch_fold_size is reached or token threshold exceeded.
        """
        if not user_text and not agent_text:
            return

        turn_index = len(self.full_history) + 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        turn_record = {
            "turn": turn_index,
            "timestamp": timestamp,
            "candidate": user_text,
            "interviewer": agent_text,
        }
        self.full_history.append(turn_record)

        self.recent_turns.append((user_text, agent_text))

        # Check turn-pair window limit (10) and token safety net (3000 tokens)
        fold_reason = None
        while len(self.recent_turns) > self.window_size:
            oldest_pair = self.recent_turns.pop(0)
            self.pending_fold_buffer.append(oldest_pair)
            fold_reason = fold_reason or "batch_size_reached"

        # Token safety net check
        current_tokens = self.estimate_turns_tokens()
        if current_tokens > self.max_recent_tokens and self.recent_turns:
            logger.info(
                "[MEMORY TOKEN SAFETY NET] recent_turns tokens (%d) > threshold (%d). Triggering early fold.",
                current_tokens,
                self.max_recent_tokens,
            )
            while self.estimate_turns_tokens() > self.max_recent_tokens and len(self.recent_turns) > 1:
                oldest_pair = self.recent_turns.pop(0)
                self.pending_fold_buffer.append(oldest_pair)
            fold_reason = "token_safety_net"

        # Persist full history to local file
        self._save_to_local_file()

        # Log confirmation that current turn's spoken reply was NOT delayed waiting on a fold
        logger.info(
            "[NO DELAY] session=%s: Spoken reply logged instantly in 0ms (not waiting on fold). "
            "recent_turns=%d/%d, pending_folds=%d",
            self.session_id,
            len(self.recent_turns),
            self.window_size,
            len(self.pending_fold_buffer),
        )

        # Trigger background fold task if batch size threshold is reached or token safety net fired
        if len(self.pending_fold_buffer) >= self.batch_fold_size or fold_reason == "token_safety_net":
            reason = fold_reason or "batch_size_reached"
            self._schedule_background_fold(reason=reason)

    def _schedule_background_fold(self, reason: str = "batch_size_reached") -> None:
        """
        Schedule background fold task using asyncio.create_task if an event loop is running.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.trigger_background_fold(reason=reason))
        except RuntimeError:
            logger.debug("No running event loop for background fold task; will fold on next flush.")

    async def trigger_background_fold(self, reason: str = "batch_size_reached") -> None:
        """
        Non-blocking background task to summarize accumulated pending_fold_buffer pairs using Groq.
        Guarded against overlapping executions using _is_folding flag and asyncio.Lock.
        """
        if self._is_folding:
            logger.info(
                "[MEMORY FOLD] Fold task already in progress; batch fold queued in pending_fold_buffer (len=%d).",
                len(self.pending_fold_buffer),
            )
            return

        if not self.pending_fold_buffer:
            return

        if self._fold_lock is None:
            self._fold_lock = asyncio.Lock()

        async with self._fold_lock:
            self._is_folding = True
            try:
                await self._execute_batch_fold(reason=reason)
            finally:
                self._is_folding = False

    async def _execute_batch_fold(self, reason: str = "batch_size_reached") -> None:
        """
        Summarize all pending turn pairs in pending_fold_buffer in a single Groq call.
        """
        if not self.pending_fold_buffer:
            return

        batch_to_fold = self.pending_fold_buffer[:]
        self.pending_fold_buffer = []

        exchanges = []
        for candidate_text, agent_text in batch_to_fold:
            cand = candidate_text if candidate_text else "N/A"
            agent = agent_text if agent_text else "N/A"
            exchanges.append(f"Candidate: {cand}\nInterviewer: {agent}")

        exchange_text = "\n---\n".join(exchanges)
        prompt = SUMMARIZE_INSTRUCTION.format(exchange_text=exchange_text)

        start_time = time.perf_counter()
        logger.info(
            "[MEMORY FOLD TRIGGERED] session=%s reason=%s batch_size=%d model=%s (Groq)",
            self.session_id,
            reason,
            len(batch_to_fold),
            self.summarization_model,
        )

        try:
            acompletion_kwargs = {
                "model": self.summarization_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            }
            if self.api_key:
                acompletion_kwargs["api_key"] = self.api_key

            response = await litellm.acompletion(**acompletion_kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            summary = response.choices[0].message.content.strip()

            if summary:
                self.summary_bullets.append(summary)
                self._rebuild_rolling_summary()
                logger.info(
                    "[MEMORY FOLD COMPLETED] session=%s reason=%s duration=%.2fms model=%s (Groq confirmed) "
                    "summarized_%d_pairs summary_bullets=%d",
                    self.session_id,
                    reason,
                    elapsed_ms,
                    self.summarization_model,
                    len(batch_to_fold),
                    len(self.summary_bullets),
                )
                self._save_to_local_file()
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "[MEMORY FOLD FAILED] session=%s reason=%s duration=%.2fms model=%s error: %s. Restoring to buffer.",
                self.session_id,
                reason,
                elapsed_ms,
                self.summarization_model,
                e,
            )
            # Re-queue failed batch back into pending_fold_buffer
            self.pending_fold_buffer = batch_to_fold + self.pending_fold_buffer

    async def flush_pending_folds(self) -> None:
        """
        Flush any remaining items in pending_fold_buffer at the end of an interview session.
        """
        if self.pending_fold_buffer:
            logger.info("[MEMORY FLUSH] Flushing %d remaining pending fold pairs at interview end.", len(self.pending_fold_buffer))
            await self.trigger_background_fold(reason="interview_end_flush")

    def _rebuild_rolling_summary(self) -> None:
        """
        Rebuild self.rolling_summary from self.summary_bullets while enforcing max_summary_bullets cap.
        """
        if len(self.summary_bullets) > self.max_summary_bullets:
            self.summary_bullets = self.summary_bullets[-self.max_summary_bullets:]

        if self.summary_bullets:
            self.rolling_summary = "\n".join(f"- {b}" for b in self.summary_bullets)
        else:
            self.rolling_summary = ""

    def build_context_messages(self) -> list[dict[str, str]]:
        """
        Build message dicts for LLM completion calls:
        - System context message with rolling summary (if non-empty)
        - Followed by full recent_turns (sliding window of last 10 turns) in chronological order
        """
        messages: list[dict[str, str]] = []

        if self.rolling_summary:
            summary_msg = f"Prior Conversation Summary:\n{self.rolling_summary}"
            messages.append({"role": "system", "content": summary_msg})

        for candidate_text, agent_text in self.recent_turns:
            if candidate_text:
                messages.append({"role": "user", "content": candidate_text})
            if agent_text:
                messages.append({"role": "assistant", "content": agent_text})

        return messages

    def get_full_context_text(self) -> str:
        """
        Return text representation of complete conversation history for feedback generation.
        """
        lines = []
        if self.rolling_summary:
            lines.append("SUMMARY OF EARLIER CONVERSATION:")
            lines.append(self.rolling_summary)
            lines.append("\nCOMPLETE CONVERSATION TURNS:")

        if self.full_history:
            for item in self.full_history:
                lines.append(f"Turn {item['turn']} [{item['timestamp']}]:")
                lines.append(f"  Candidate: {item['candidate']}")
                lines.append(f"  Interviewer: {item['interviewer']}")
        else:
            for i, (candidate_text, agent_text) in enumerate(self.recent_turns, 1):
                lines.append(f"Turn {i}:")
                lines.append(f"  Candidate: {candidate_text}")
                lines.append(f"  Interviewer: {agent_text}")

        return "\n".join(lines) if lines else "No prior conversation history recorded."

    def get_last_response(self) -> str | None:
        """
        Return the most recent non-empty agent (interviewer) reply text.
        """
        for _, agent_text in reversed(self.recent_turns):
            if agent_text and agent_text.strip():
                return agent_text.strip()

        if self.session_id:
            file_path = CONVERSATIONS_DIR / f"{self.session_id}.json"
            try:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    history = data.get("full_history", [])
                    for turn in reversed(history):
                        agent_text = turn.get("interviewer", "").strip()
                        if agent_text:
                            logger.info(
                                "[REPEAT] Loaded last response from local file for session %s (turn %d)",
                                self.session_id, turn.get("turn", "?"),
                            )
                            return agent_text
            except Exception as e:
                logger.warning("Error reading local conversation file for repeat: %s", e)

        return None

    def _save_to_local_file(self) -> None:
        """
        Save conversation log to local JSON file under conversations/{session_id}.json.
        """
        if not self.session_id:
            return

        file_path = CONVERSATIONS_DIR / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "total_turns": len(self.full_history),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_bullets": self.summary_bullets,
            "pending_fold_buffer": self.pending_fold_buffer,
            "rolling_summary": self.rolling_summary,
            "full_history": self.full_history,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved local conversation log to %s", file_path)
        except Exception as e:
            logger.warning("Error saving local conversation log to %s: %s", file_path, e)



