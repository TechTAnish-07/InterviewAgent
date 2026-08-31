import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from memory import ConversationMemory, DEFAULT_WINDOW_SIZE, SUMMARIZATION_MODEL
from prompts import SUMMARIZE_INSTRUCTION


def test_memory_default_window_size():
    mem = ConversationMemory(session_id="test_session")
    assert mem.window_size == 10
    assert mem.summarization_model == SUMMARIZATION_MODEL


def test_memory_10_turn_window_and_batch_folding():
    mem = ConversationMemory(window_size=10, batch_fold_size=3, max_recent_tokens=3000)

    # Add 10 turn pairs (should not trigger any folds yet)
    for i in range(1, 11):
        mem.add_turn(f"User question {i}", f"Agent answer {i}")

    assert len(mem.recent_turns) == 10
    assert len(mem.pending_fold_buffer) == 0

    # Add 11th turn pair: 1st evicted pair enters pending_fold_buffer
    mem.add_turn("User question 11", "Agent answer 11")
    assert len(mem.recent_turns) == 10
    assert len(mem.pending_fold_buffer) == 1
    assert mem.pending_fold_buffer[0] == ("User question 1", "Agent answer 1")

    # Add 12th and 13th turn pairs: pending_fold_buffer reaches 3 pairs
    mem.add_turn("User question 12", "Agent answer 12")
    mem.add_turn("User question 13", "Agent answer 13")
    assert len(mem.recent_turns) == 10
    assert len(mem.pending_fold_buffer) == 3


@pytest.mark.asyncio
async def test_memory_groq_background_batched_fold():
    mem = ConversationMemory(window_size=10, batch_fold_size=3, max_recent_tokens=3000)

    mock_llm_response = AsyncMock()
    mock_llm_response.choices = [
        AsyncMock(message=AsyncMock(content="Candidate discussed Python memory management and database indexing."))
    ]

    with patch("memory.litellm.acompletion", return_value=mock_llm_response) as mock_acompletion:
        # Add 13 turns to trigger a 3-pair batch fold
        for i in range(1, 14):
            mem.add_turn(f"Candidate question {i}", f"Interviewer reply {i}")

        # Give async background task time to run
        await asyncio.sleep(0.1)

        # Verify summarization model was called
        assert mock_acompletion.called
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == mem.summarization_model


        # Verify summary was appended to summary_bullets and rolling_summary
        assert len(mem.summary_bullets) == 1
        assert "Candidate discussed Python memory management" in mem.rolling_summary


@pytest.mark.asyncio
async def test_token_safety_net_early_fold():
    # Set low token threshold (50 tokens) to test token safety net
    mem = ConversationMemory(window_size=10, batch_fold_size=3, max_recent_tokens=50)

    # Add long turn pairs that exceed 50 tokens
    long_cand = "I have extensive experience building scalable microservices in Python, FastAPI, Docker, and Kubernetes for high throughput applications." * 2
    long_agent = "That sounds impressive. How do you handle distributed tracing, database connection pooling, and message queues in your architecture?" * 2

    mem.add_turn(long_cand, long_agent)
    mem.add_turn(long_cand, long_agent)

    # Token safety net should trigger eviction into pending_fold_buffer even before 10 turns
    assert len(mem.pending_fold_buffer) > 0


@pytest.mark.asyncio
async def test_flush_pending_folds():
    mem = ConversationMemory(window_size=10, batch_fold_size=3)
    mem.pending_fold_buffer.append(("Turn candidate 1", "Turn agent 1"))

    mock_llm_response = AsyncMock()
    mock_llm_response.choices = [
        AsyncMock(message=AsyncMock(content="Flushed summary bullet."))
    ]

    with patch("memory.litellm.acompletion", return_value=mock_llm_response):
        await mem.flush_pending_folds()

    assert len(mem.pending_fold_buffer) == 0
    assert "Flushed summary bullet." in mem.rolling_summary


def test_build_context_messages_structure():
    mem = ConversationMemory(window_size=10)
    mem.rolling_summary = "- Prior Q&A summary point 1"
    mem.pending_fold_buffer = [("Old user", "Old assistant")]
    mem.recent_turns = [("Hi", "Hello"), ("How are you?", "I am fine.")]

    # Default build_context_messages returns all dialogue turns (pending_fold_buffer + recent_turns) without system message
    messages = mem.build_context_messages()
    assert len(messages) == 6
    assert messages[0] == {"role": "user", "content": "Old user"}
    assert messages[1] == {"role": "assistant", "content": "Old assistant"}
    assert messages[2] == {"role": "user", "content": "Hi"}
    assert messages[3] == {"role": "assistant", "content": "Hello"}
    assert messages[4] == {"role": "user", "content": "How are you?"}
    assert messages[5] == {"role": "assistant", "content": "I am fine."}

    # With include_summary_system_msg=True, system message is prepended
    messages_with_sys = mem.build_context_messages(include_summary_system_msg=True)
    assert len(messages_with_sys) == 7
    assert messages_with_sys[0] == {"role": "system", "content": "Prior Conversation Summary:\n- Prior Q&A summary point 1"}


@pytest.mark.asyncio
async def test_failed_fold_retains_turns_in_buffer():
    mem = ConversationMemory(window_size=10, batch_fold_size=3)
    mem.pending_fold_buffer.append(("Turn 1", "Reply 1"))

    with patch("memory.litellm.acompletion", side_effect=RuntimeError("Groq timeout")):
        await mem.trigger_background_fold(reason="test_fail")

    # Turns must remain in pending_fold_buffer upon failure so they are never lost from context
    assert len(mem.pending_fold_buffer) == 1
    assert mem.pending_fold_buffer[0] == ("Turn 1", "Reply 1")
