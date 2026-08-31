import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tools import show_coding_question, get_current_code, run_code_check


@pytest.mark.asyncio
async def test_show_coding_question():
    mock_room = MagicMock()
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_data = AsyncMock()

    context = {
        "session_id": "test-session-123",
        "room": mock_room,
        "current_question": None,
    }

    question = "Write a function to reverse a linked list."
    res = await show_coding_question(question_text=question, context=context)

    assert "Write a function to reverse a linked list." in res
    assert context["current_question"] == question
    mock_room.local_participant.publish_data.assert_called_once()
    call_args = mock_room.local_participant.publish_data.call_args
    sent_payload = json.loads(call_args[0][0])
    assert sent_payload["type"] == "coding_question"
    assert sent_payload["questionText"] == question


@pytest.mark.asyncio
async def test_get_current_code_empty():
    context = {
        "latest_code": "",
        "last_run_result": None,
    }
    res = await get_current_code(context=context)
    assert "has not written any code" in res
    assert "has not clicked Run or executed their code yet" in res


@pytest.mark.asyncio
async def test_get_current_code_with_run_result():
    context = {
        "latest_code": "def solve(n):\n    return n * 2",
        "last_run_result": {
            "code": "def solve(n):\n    return n * 2",
            "stdout": "42\n",
            "stderr": "",
            "status": "Accepted",
        },
    }
    res = await get_current_code(context=context)
    assert "def solve(n):" in res
    assert "Status: Accepted" in res
    assert "42" in res


@pytest.mark.asyncio
async def test_run_code_check_success():
    context = {
        "session_id": "session-456",
        "latest_code": "print('hello world')",
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "status": "Accepted",
        "stdout": "hello world\n",
        "stderr": "",
    })

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_resp
        res = await run_code_check(language="python", stdin="5", context=context)

        assert "Execution Result for python" in res
        assert "hello world" in res
        assert "Status: Accepted" in res


@pytest.mark.asyncio
async def test_run_code_check_no_code():
    context = {
        "session_id": "session-789",
        "latest_code": "",
    }
    res = await run_code_check(language="python", context=context)
    assert "Error: Cannot execute code" in res


def test_question_resolution_and_triggers():
    from questions import ALL_QUESTIONS
    assert "E1" in ALL_QUESTIONS
    assert "Two Sum" in ALL_QUESTIONS["E1"]["title"]

    # Test candidate missing-question triggers
    sample_candidate_phrases = [
        "i don't see the question",
        "there is no question on my screen",
        "nothing on screen",
        "screen is empty",
        "where is it",
        "cannot see it",
        "not showing",
        "cant see the problem",
        "show me the question",
    ]
    QUESTION_QUERY_TRIGGERS = (
        "where is the question", "where's the question", "where is the problem", "where's the problem",
        "where is it", "where's it", "where question", "where problem",
        "can't find the question", "cannot find the question", "can't find the problem",
        "can't see the question", "cannot see the question", "can't see the problem",
        "don't see the question", "do not see the question", "dont see the question",
        "can't see it", "cannot see it", "don't see it", "cant see it", "cant see",
        "can't see", "cannot see", "can not see", "not able to see", "unable to see",
        "not seeing", "not seeing any", "see nothing", "don't see anything", "dont see anything",
        "no question", "no problem", "there is no question", "there's no question",
        "nothing on screen", "nothing on my screen", "nothing on the screen",
        "screen is empty", "screen is blank", "blank screen", "empty screen",
        "not showing", "not visible", "isn't showing", "is not showing", "isn't visible", "is not visible",
        "share the question", "show the question", "show me the question", "display the question",
        "send the question", "what is the question", "what's the question", "what is the problem",
        "what's the problem", "put it on my screen", "put the question on my screen",
        "repeat the question", "i see no question", "cannot find it", "can't find it"
    )

    for phrase in sample_candidate_phrases:
        matched = any(trig in phrase.lower() for trig in QUESTION_QUERY_TRIGGERS)
        assert matched, f"Phrase '{phrase}' should match QUESTION_QUERY_TRIGGERS"

