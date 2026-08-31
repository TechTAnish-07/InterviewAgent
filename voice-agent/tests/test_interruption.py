import asyncio
import os
import pytest
from unittest.mock import MagicMock
from livekit.agents import AgentSession, TurnHandlingOptions, InterruptionOptions, llm
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins import silero, groq
from memory import ConversationMemory
from agent import InterviewAgent


@pytest.mark.asyncio
async def test_agent_session_turn_handling_config(monkeypatch):
    """
    Test that AgentSession instantiates with TurnHandlingOptions and InterruptionOptions.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy")

    turn_handling = TurnHandlingOptions(
        interruption=InterruptionOptions(
            enabled=True,
            mode="adaptive",
            min_duration=0.8,
            min_words=2,
        )
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3"),
        turn_handling=turn_handling,
    )

    interruption_opts = session.options.turn_handling.get("interruption", {})
    assert interruption_opts.get("enabled") is True
    assert interruption_opts.get("min_duration") == 0.8
    assert interruption_opts.get("min_words") == 2
    assert interruption_opts.get("mode") == "adaptive"


@pytest.mark.asyncio
async def test_speak_and_log_uninterrupted():
    """
    Test that when a speech handle completes without interruption, full reply is added to memory.
    """
    memory = ConversationMemory(window_size=5, session_id="test-uninterrupted")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    h = SpeechHandle.create(allow_interruptions=True)
    mock_session.say.return_value = h

    agent = InterviewAgent(
        session_id="test-uninterrupted",
        context={"candidateName": "Alice", "jobRole": "Engineer"},
        system_prompt="Test prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity

    full_reply = "What experience do you have with distributed database transactions?"
    agent._speak_and_log(full_reply, candidate_text="Tell me about databases")

    msg = llm.ChatMessage(role="assistant", content=[full_reply], interrupted=False)
    h._item_added([msg])
    h._mark_done()
    await asyncio.sleep(0.01)

    assert len(memory.recent_turns) == 1
    assert memory.recent_turns[0] == ("Tell me about databases", full_reply)


@pytest.mark.asyncio
async def test_speak_and_log_interrupted():
    """
    Test that when a speech handle is interrupted mid-sentence, memory retains the full generated reply committed at generation time.
    """
    memory = ConversationMemory(window_size=5, session_id="test-interrupted")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    h = SpeechHandle.create(allow_interruptions=True)
    mock_session.say.return_value = h

    agent = InterviewAgent(
        session_id="test-interrupted",
        context={"candidateName": "Bob", "jobRole": "Backend Developer"},
        system_prompt="Test prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity

    full_reply = "Could you explain how Kafka handles partition rebalancing when a broker dies?"
    truncated_spoken = "Could you explain how Kafka"

    agent._speak_and_log(full_reply, candidate_text="I know streaming technologies.")

    # Simulate SDK adding truncated text item upon interruption
    msg = llm.ChatMessage(role="assistant", content=[truncated_spoken], interrupted=True)
    h._item_added([msg])
    h.interrupt()
    h._mark_done()
    await asyncio.sleep(0.01)

    assert len(memory.recent_turns) == 1
    # Check that memory recorded the full generated reply committed at generation time
    assert memory.recent_turns[0] == ("I know streaming technologies.", full_reply)


@pytest.mark.asyncio
async def test_streaming_speak_and_log_memory():
    """
    Test that _speak_and_log handles an AsyncIterable text stream and records text into memory upon generation stream completion.
    """
    memory = ConversationMemory(window_size=5, session_id="test-streaming-memory")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    h = SpeechHandle.create(allow_interruptions=True)
    async def mock_say(stream_or_text):
        if hasattr(stream_or_text, "__aiter__"):
            async for _ in stream_or_text:
                pass
        return h
    mock_session.say.side_effect = lambda s: (asyncio.create_task(mock_say(s)) if hasattr(s, "__aiter__") else None) or h

    agent = InterviewAgent(
        session_id="test-streaming-memory",
        context={"candidateName": "Charlie", "jobRole": "DevOps Engineer"},
        system_prompt="Test prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity

    full_text_container = ["Could you explain Kubernetes pod scheduling strategy?"]

    async def mock_text_stream():
        yield "Could you explain "
        yield "Kubernetes pod "
        yield "scheduling strategy?"

    agent._speak_and_log(
        mock_text_stream(),
        candidate_text="I work with K8s.",
        full_generated_text_container=full_text_container,
    )

    msg = llm.ChatMessage(role="assistant", content=[full_text_container[0]], interrupted=False)
    h._item_added([msg])
    h._mark_done()
    await asyncio.sleep(0.05)

    assert len(memory.recent_turns) == 1
    assert memory.recent_turns[0] == ("I work with K8s.", full_text_container[0])
