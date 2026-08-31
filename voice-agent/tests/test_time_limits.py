import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from memory import ConversationMemory
from agent import InterviewAgent, SOFT_TIME_LIMIT_SECONDS, HARD_TIME_LIMIT_SECONDS, MAX_TURN_COUNT
from prompts import WRAP_UP_INSTRUCTION


@pytest.mark.asyncio
async def test_max_turn_count_limit():
    memory = ConversationMemory(window_size=5, session_id="test-max-turns")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    agent = InterviewAgent(
        session_id="test-max-turns",
        context={"candidateName": "Test Candidate", "jobRole": "Software Engineer"},
        system_prompt="Base prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity
    agent.end_interview_flow = AsyncMock()

    # Set turn count right at the limit
    agent._turn_count = MAX_TURN_COUNT - 1

    turn_ctx = MagicMock()
    await agent._process_user_turn(transcript="Here is my answer", turn_ctx=turn_ctx)

    assert agent._turn_count == MAX_TURN_COUNT
    agent.end_interview_flow.assert_called_once_with("max_turns_reached")


@pytest.mark.asyncio
async def test_hard_time_limit():
    memory = ConversationMemory(window_size=5, session_id="test-hard-time")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    agent = InterviewAgent(
        session_id="test-hard-time",
        context={"candidateName": "Test Candidate", "jobRole": "Software Engineer"},
        system_prompt="Base prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity
    agent.end_interview_flow = AsyncMock()

    # Simulate start time being 1900 seconds ago (> HARD_TIME_LIMIT_SECONDS=1800)
    agent._interview_start_time = time.time() - (HARD_TIME_LIMIT_SECONDS + 100)

    turn_ctx = MagicMock()
    await agent._process_user_turn(transcript="Let's continue discussing", turn_ctx=turn_ctx)

    agent.end_interview_flow.assert_called_once_with("time_limit_reached")


@pytest.mark.asyncio
async def test_soft_time_limit_prompt_injection():
    memory = ConversationMemory(window_size=5, session_id="test-soft-time")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    agent = InterviewAgent(
        session_id="test-soft-time",
        context={"candidateName": "Test Candidate", "jobRole": "Software Engineer"},
        system_prompt="Base prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity

    # Simulate elapsed time between SOFT (1500s) and HARD (1800s)
    agent._interview_start_time = time.time() - (SOFT_TIME_LIMIT_SECONDS + 10)
    assert agent._soft_warning_sent is False

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(delta=MagicMock(content="We are running low on time.", tool_calls=None))]

    async def mock_stream():
        yield mock_resp

    with patch("litellm.acompletion", return_value=mock_stream()) as mock_llm:
        turn_ctx = MagicMock()
        await agent._process_user_turn(transcript="I have a question about databases", turn_ctx=turn_ctx)

        assert agent._soft_warning_sent is True
        call_kwargs = mock_llm.call_args[1]
        sent_messages = call_kwargs["messages"]
        system_msg = sent_messages[0]["content"]

        assert "[TIME LIMIT APPROACHING" in system_msg
        assert "approaching the time limit for this interview" in system_msg


@pytest.mark.asyncio
async def test_dynamic_duration_custom_minutes():
    """Test dynamic duration configured via context metadata (e.g. 15 minutes)."""
    memory = ConversationMemory(window_size=5, session_id="test-custom-duration")
    mock_room = MagicMock()

    agent = InterviewAgent(
        session_id="test-custom-duration",
        context={"candidateName": "Test", "jobRole": "Engineer", "durationMinutes": 15},
        system_prompt="Base prompt",
        memory=memory,
        room=mock_room,
    )

    assert agent._max_time_seconds == 15 * 60  # 900s
    assert agent._warning_time_seconds == 10 * 60  # 600s (5 mins before 900s)


@pytest.mark.asyncio
async def test_processing_turn_flag_resets_safely():
    """Test that _is_processing_turn is cleanly reset even if an exception occurs."""
    memory = ConversationMemory(window_size=5, session_id="test-flag-reset")
    mock_room = MagicMock()
    mock_session = MagicMock()
    mock_activity = MagicMock()
    mock_activity.session = mock_session

    agent = InterviewAgent(
        session_id="test-flag-reset",
        context={"candidateName": "Test", "jobRole": "Engineer"},
        system_prompt="Base prompt",
        memory=memory,
        room=mock_room,
    )
    agent._activity = mock_activity

    turn_ctx = MagicMock()
    with patch("litellm.acompletion", side_effect=Exception("LLM error")):
        await agent._process_user_turn(transcript="Hello", turn_ctx=turn_ctx)

    assert agent._is_processing_turn is False

