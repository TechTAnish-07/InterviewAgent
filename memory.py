import json
import logging
import os
from datetime import datetime
from pathlib import Path
from prompts import SUMMARIZE_INSTRUCTION

logger = logging.getLogger("interview-agent.memory")

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"


class ConversationMemory:
    """
    Bounded working memory for live voice interview conversation.
    - Maintains a sliding window of recent turn pairs (last 5 turns) for LLM completion calls.
    - Summarizes expired turns into rolling_summary to keep LLM context size bounded.
    - Persists complete un-truncated conversation history locally in conversations/{session_id}.json.
    """

    def __init__(self, window_size: int = 5, session_id: str | None = None) -> None:
        self.recent_turns: list[tuple[str, str]] = []  # [(user_text, agent_text), ...]
        self.full_history: list[dict] = []  # [{turn: 1, timestamp: "...", candidate: "...", interviewer: "..."}, ...]
        self.rolling_summary: str = ""
        self.window_size: int = window_size
        self.session_id: str | None = session_id
        self.model_name: str = os.getenv("MODEL_NAME") or "gemini/gemini-2.5-flash"

        # Ensure conversations directory exists
        try:
            CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create conversations directory: %s", e)

    def add_turn(self, user_text: str, agent_text: str) -> None:
        """
        Append a turn pair (candidate text, agent reply).
        - Appends to sliding window (recent_turns).
        - If turn pairs exceed window_size (5), folds the oldest pair into rolling_summary.
        - Appends to full_history and persists to local file in conversations/.
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
        if len(self.recent_turns) > self.window_size:
            oldest_pair = self.recent_turns.pop(0)
            self.summarize_and_fold(oldest_pair)

        # Persist full conversation history to local file
        self._save_to_local_file()

    def summarize_and_fold(self, oldest_pair: tuple[str, str]) -> None:
        """
        Fold the oldest turn pair directly into rolling_summary without triggering an extra LLM API call.
        """
        candidate_text, agent_text = oldest_pair
        summary_piece = f"Candidate: \"{candidate_text[:120]}\" | Interviewer: \"{agent_text[:120]}\""
        if self.rolling_summary:
            self.rolling_summary += f"\n- {summary_piece}"
        else:
            self.rolling_summary = f"- {summary_piece}"
        logger.info("Updated rolling summary with folded turn.")

    def build_context_messages(self) -> list[dict[str, str]]:
        """
        Build message dicts for LLM completion calls:
        - System context message with rolling summary (if non-empty)
        - Followed by full recent_turns (sliding window of last 5 turns) in chronological order
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
        Scans recent_turns in reverse so it correctly skips the current turn
        (which has an empty agent text when the candidate just asked "please repeat").
        Priority:
          1. In-memory recent_turns (fastest, always current for live session)
          2. Local persisted JSON file (fallback if window was already flushed)
        Returns None if no prior response is found.
        """
        # 1. Scan in-memory sliding window in reverse for last non-empty agent reply
        for _, agent_text in reversed(self.recent_turns):
            if agent_text and agent_text.strip():
                return agent_text.strip()

        # 2. Fallback: read local JSON file for the most recent non-empty interviewer turn
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
            "rolling_summary": self.rolling_summary,
            "full_history": self.full_history,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved local conversation log to %s", file_path)
        except Exception as e:
            logger.warning("Error saving local conversation log to %s: %s", file_path, e)

