import json
import logging
import os
from fastapi import APIRouter, HTTPException, Request
from livekit import api

logger = logging.getLogger("dispatch-controller")

router = APIRouter(tags=["Dispatch"])


@router.post("/dispatch-agent")
async def dispatch_agent(request: Request):
    """
    Dispatch the 'interview-agent' LiveKit worker to a specific room.
    Reads request body directly to prevent 422 validation errors.
    """
    try:
        body_bytes = await request.body()
        if not body_bytes:
            raise ValueError("Empty request body")
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception as e:
        logger.warning("Empty or invalid dispatch payload: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid or empty JSON body: {e}")

    target_room = data.get("room") or data.get("room_name") or data.get("roomName")
    target_session_id = data.get("session_id") if data.get("session_id") is not None else data.get("sessionId")

    if not target_room or target_session_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Both 'room' and 'session_id' are required. Received payload: {data}"
        )

    livekit_url = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")

    # Normalize URL: livekit.api needs http:// not ws://
    http_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")

    meta_dict = {
        "sessionId": target_session_id,
        "candidateName": data.get("candidate_name") or data.get("candidateName") or "Candidate",
        "jobTitle": data.get("job_title") or data.get("jobTitle") or data.get("jobRole") or "Software Engineer",
        "summary": data.get("summary") or "",
        "skills": data.get("skills") or "[]",
        "resumeText": data.get("resume_text") or data.get("resumeText") or "",
    }
    metadata_str = json.dumps(meta_dict)

    try:
        lk = api.LiveKitAPI(http_url, api_key, api_secret)
        dispatch = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="interview-agent",
                room=str(target_room),
                metadata=metadata_str,
            )
        )
        await lk.aclose()
        logger.info(
            "Dispatched interview-agent to room %s (session=%s), dispatch_id=%s",
            target_room, target_session_id, dispatch.id
        )
        return {"dispatch_id": dispatch.id, "room": target_room, "session_id": target_session_id}
    except Exception as e:
        logger.error("Failed to dispatch agent to room %s: %s", target_room, e)
        raise HTTPException(status_code=502, detail=f"Agent dispatch failed: {e}")
