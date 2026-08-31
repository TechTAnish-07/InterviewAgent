import asyncio
import json
import os
import pytest
import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "ollama/qwen3.5:9b")

TOOLS_LIST = [
    {
        "type": "function",
        "function": {
            "name": "end_interview",
            "description": "Wrap up and conclude the live interview when enough ground has been covered or the candidate requests to finish.",
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
            "description": "Repeat or recall the interviewer's most recent spoken response. Use when candidate asks to repeat.",
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
            "description": "Display a coding question on the candidate's screen in their code editor panel.",
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
            "description": "Retrieve the latest code currently written by the candidate in their editor along with execution results.",
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
            "description": "Execute the candidate's code on the backend with optional stdin to test an edge case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language of the code (e.g. python, javascript)",
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


RESOLVED_MODEL = MODEL_NAME


@pytest.mark.asyncio
async def test_ollama_basic_generation():
    """Test 1: Basic local text generation with Ollama and LiteLLM."""
    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=[{"role": "user", "content": "Respond with 'READY' if you can read this."}],
        temperature=0.1,
        max_tokens=500,
    )
    content = resp.choices[0].message.content or ""
    print(f"\n[Test 1 Basic Generation Output]: {content}")
    assert len(content.strip()) > 0


@pytest.mark.asyncio
async def test_ollama_streaming():
    """Test 2: Streaming output with Ollama and LiteLLM."""
    from agent import _stream_llm_response

    resp_stream = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5 separated by spaces."}],
        temperature=0.1,
        max_tokens=500,
        stream=True,
    )
    full_text_container = []
    tool_calls_container = []

    chunks = []
    async for chunk in _stream_llm_response(resp_stream, full_text_container, tool_calls_container):
        chunks.append(chunk)

    print(f"\n[Test 2 Streamed Chunks ({len(chunks)})]: {''.join(chunks)}")
    print(f"[Test 2 Full Text Container]: {full_text_container}")
    assert len(full_text_container) > 0
    assert len(full_text_container[0]) > 0



@pytest.mark.asyncio
async def test_ollama_tool_calling_show_coding():
    """Test 3a: Tool calling for show_coding_question."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical interviewer. When transitioning to coding, invoke the show_coding_question tool with the problem statement.",
        },
        {
            "role": "user",
            "content": "I am ready for the coding exercise now. Please show me the coding question.",
        },
    ]

    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        tools=TOOLS_LIST,
    )

    choice = resp.choices[0].message
    print(f"\n[Test 3a Choice message]: {choice}")
    tool_calls = getattr(choice, "tool_calls", None)
    assert tool_calls is not None, "Expected model to return tool_calls"
    assert len(tool_calls) > 0
    func_name = tool_calls[0].function.name
    print(f"[Test 3a Invoked Function]: {func_name}")
    assert func_name == "show_coding_question"


@pytest.mark.asyncio
async def test_ollama_tool_calling_end_interview():
    """Test 3b: Tool calling for end_interview."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical interviewer. When the interview is finished or the candidate requests to wrap up, invoke the end_interview tool.",
        },
        {
            "role": "user",
            "content": "Thank you, I'd like to end the interview now.",
        },
    ]

    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        tools=TOOLS_LIST,
    )

    choice = resp.choices[0].message
    print(f"\n[Test 3b Choice message]: {choice}")
    tool_calls = getattr(choice, "tool_calls", None)
    assert tool_calls is not None, "Expected model to return tool_calls"
    func_name = tool_calls[0].function.name
    print(f"[Test 3b Invoked Function]: {func_name}")
    assert func_name == "end_interview"


@pytest.mark.asyncio
async def test_ollama_tool_calling_get_current_code():
    """Test 3c: Tool calling for get_current_code."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical interviewer. When the candidate says they are done with their code solution, invoke get_current_code to inspect their code.",
        },
        {
            "role": "user",
            "content": "I am done writing my solution in the code editor. That's my final code.",
        },
    ]

    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        tools=TOOLS_LIST,
    )

    choice = resp.choices[0].message
    print(f"\n[Test 3c Choice message]: {choice}")
    tool_calls = getattr(choice, "tool_calls", None)
    assert tool_calls is not None, "Expected model to return tool_calls"
    func_name = tool_calls[0].function.name
    print(f"[Test 3c Invoked Function]: {func_name}")
    assert func_name in ("get_current_code", "run_code_check")


@pytest.mark.asyncio
async def test_ollama_memory_summarization():
    """Test 4: Conversation memory summarization with Ollama."""
    from memory import ConversationMemory

    mem = ConversationMemory(
        window_size=2,
        batch_fold_size=2,
        summarization_model=MODEL_NAME,
    )

    mem.add_turn("My name is Alice and I have 5 years experience with Python and FastAPI.", "Great, tell me about an async project.")
    mem.add_turn("I built a real-time analytics streaming engine with Redis and FastAPI.", "How did you handle race conditions?")
    mem.add_turn("We used distributed Redis locks with TTL.", "Very good. Let's move to database indexing.")

    # Trigger memory fold
    await mem.trigger_background_fold(reason="test_fold")
    # Allow background task to complete
    await asyncio.sleep(0.5)
    await mem.flush_pending_folds()
    print(f"\n[Test 4 Rolling Summary]:\n{mem.rolling_summary}")
    assert len(mem.summary_bullets) > 0 or len(mem.rolling_summary) > 0


@pytest.mark.asyncio
async def test_ollama_moderation():
    """Test 5: Moderation check with Ollama fallback."""
    from moderation import check_message_relevance

    res_off_topic = await check_message_relevance("Can you give me a pizza recipe with mushrooms?", model_name=MODEL_NAME)
    print(f"\n[Test 5 Moderation Off-Topic]: {res_off_topic}")
    assert res_off_topic in ("OFF_TOPIC", "NORMAL")

    res_normal = await check_message_relevance("I implemented binary search in Python.", model_name=MODEL_NAME)
    print(f"[Test 5 Moderation Normal]: {res_normal}")
    assert res_normal == "NORMAL"


@pytest.mark.asyncio
async def test_ollama_tool_calling_repeat():
    """Test 6: Tool calling for repeat_last_response."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical interviewer. If the candidate asks you to repeat or say that again, invoke the repeat_last_response tool.",
        },
        {
            "role": "user",
            "content": "Sorry, I didn't catch that. Could you please repeat what you just said?",
        },
    ]

    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        tools=TOOLS_LIST,
    )

    choice = resp.choices[0].message
    print(f"\n[Test 6 Choice message]: {choice}")
    tool_calls = getattr(choice, "tool_calls", None)
    assert tool_calls is not None, "Expected model to return tool_calls"
    func_name = tool_calls[0].function.name
    print(f"[Test 6 Invoked Function]: {func_name}")
    assert func_name == "repeat_last_response"


@pytest.mark.asyncio
async def test_ollama_tool_calling_run_code():
    """Test 7: Tool calling for run_code_check."""
    messages = [
        {
            "role": "system",
            "content": "You are a technical interviewer. When you want to run or test the candidate's code on the backend with specific input, invoke the run_code_check tool.",
        },
        {
            "role": "user",
            "content": "I have written my solution. Can you test my code with stdin='5' in python?",
        },
    ]

    resp = await litellm.acompletion(
        model=RESOLVED_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        tools=TOOLS_LIST,
    )

    choice = resp.choices[0].message
    print(f"\n[Test 7 Choice message]: {choice}")
    tool_calls = getattr(choice, "tool_calls", None)
    assert tool_calls is not None, "Expected model to return tool_calls"
    func_name = tool_calls[0].function.name
    print(f"[Test 7 Invoked Function]: {func_name}")
    assert func_name in ("run_code_check", "get_current_code")


@pytest.mark.asyncio
async def test_ollama_feedback_generation():
    """Test 8: Feedback report generation with Ollama."""
    from prompts import FEEDBACK_GENERATION_PROMPT

    prompt = FEEDBACK_GENERATION_PROMPT.format(
        candidate_name="Alice",
        job_role="Senior Python Engineer",
        resume_text="5 years Python, FastAPI, Docker",
        full_conversation_context="Interviewer: Explain GIL.\nCandidate: The GIL is a mutex that protects Python objects from concurrent thread access.",
    )

    resp = litellm.completion(
        model=RESOLVED_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert technical interviewer. Generate concise structured feedback."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    content = resp.choices[0].message.content or ""
    print(f"\n[Test 8 Feedback Generation Report Snippet]:\n{content[:200]}...")
    assert len(content.strip()) > 50
    assert "Overview" in content or "Strengths" in content


