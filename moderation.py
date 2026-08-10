import logging
import os
import litellm

logger = logging.getLogger("interview-agent.moderation")

MODERATION_SYSTEM_PROMPT = """
You are a safety and relevance classifier for a live technical job interview.
Classify the user's message into EXACTLY ONE category:

- OFF_TOPIC: Asking for recipes, cooking, sports scores, weather, trivia, or general non-interview requests.
- INAPPROPRIATE: Abusive, profane, hateful, or offensive remarks.
- NORMAL: Technical explanations, work experience, clarifying questions, "I don't know", or normal responses.

Output ONLY the exact category name (NORMAL, OFF_TOPIC, or INAPPROPRIATE).
"""


OFF_TOPIC_TRIGGERS = {
    "recipe", "cook", "cooking", "weather", "sports", "score", "game", "pizza",
    "burger", "movie", "song", "joke", "president", "politics",
    "crypto", "bitcoin", "lottery"
}

INAPPROPRIATE_TRIGGERS = {
    "fuck", "shit", "bitch", "bastard", "idiot", "stupid", "hate", "shut up"
}


def check_message_relevance(latest_message: str, recent_turns: list | None = None) -> str:
    """
    Classify latest candidate turn into NORMAL, OFF_TOPIC, or INAPPROPRIATE.
    Uses fast local heuristic checks first to save LLM API quota on standard interview turns.
    """
    if not latest_message or not latest_message.strip():
        return "NORMAL"

    text_lower = latest_message.lower().strip()

    has_off_topic = any(trigger in text_lower for trigger in OFF_TOPIC_TRIGGERS)
    has_inappropriate = any(trigger in text_lower for trigger in INAPPROPRIATE_TRIGGERS)

    # Fast-path: Normal interview responses skip extra LLM API call
    if not has_off_topic and not has_inappropriate:
        return "NORMAL"

    model_name = os.getenv("MODEL_NAME") or "gemini/gemini-2.0-flash"

    context_snippet = ""
    if recent_turns:
        last_turn = recent_turns[-1]
        context_snippet = f"\nPrior interviewer question: \"{last_turn[1]}\"\n"

    user_prompt = f"{context_snippet}Candidate latest message: \"{latest_message.strip()}\"\nClassification:"

    messages = [
        {"role": "system", "content": MODERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        raw_content = response.choices[0].message.content or ""
        classification = raw_content.strip().upper()

        if "INAPPROPRIATE" in classification:
            return "INAPPROPRIATE"
        elif "OFF_TOPIC" in classification or "OFFTOPIC" in classification:
            return "OFF_TOPIC"
        else:
            return "NORMAL"
    except Exception as e:
        logger.error("Error running moderation check: %s", e)
        # Fail safe to NORMAL on API errors to avoid blocking eligible candidates
        return "NORMAL"
