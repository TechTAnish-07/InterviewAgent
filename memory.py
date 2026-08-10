import logging
import os
import litellm
from prompts import SUMMARIZE_INSTRUCTION

logger = logging.getLogger("interview-agent.memory")


class ConversationMemory:
    """
    Bounded working memory for live voice interview conversation.
    Maintains a sliding window of recent turn pairs and a rolling summary of older turns.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.recent_turns: list[tuple[str, str]] = []  # [(user_text, agent_text), ...]
        self.rolling_summary: str = ""
        self.window_size: int = window_size
        self.model_name: str = os.getenv("MODEL_NAME") or "gemini/gemini-2.0-flash"

    def add_turn(self, user_text: str, agent_text: str) -> None:
        """
        Append a turn pair (candidate text, agent reply).
        If turn pairs exceed window_size, fold the oldest pair into rolling_summary.
        """
        if not user_text and not agent_text:
            return

        self.recent_turns.append((user_text, agent_text))
        if len(self.recent_turns) > self.window_size:
            oldest_pair = self.recent_turns.pop(0)
            self.summarize_and_fold(oldest_pair)

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
        - Followed by full recent_turns in chronological order
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
        Return text representation of rolling summary + recent turns for feedback generation.
        """
        lines = []
        if self.rolling_summary:
            lines.append("SUMMARY OF EARLIER CONVERSATION:")
            lines.append(self.rolling_summary)
            lines.append("\nRECENT CONVERSATION TURNS:")

        for i, (candidate_text, agent_text) in enumerate(self.recent_turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Candidate: {candidate_text}")
            lines.append(f"  Interviewer: {agent_text}")

        return "\n".join(lines) if lines else "No prior conversation history recorded."
