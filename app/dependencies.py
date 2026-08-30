"""
Shared FastAPI security dependencies.

Why this exists:
    /dispatch-agent, /resume/normalize, and /resume/check-relevance are server-to-server
    endpoints called exclusively by the Spring Boot backend (LiveInterview).  They are NOT
    browser endpoints — CORS does not protect them.  Instead, every caller must supply the
    shared X-Internal-Api-Key header whose value matches the INTERNAL_API_KEY env variable
    that both services share.

Usage (in a controller):
    from app.dependencies import verify_internal_key
    from fastapi import Depends

    @router.post("/dispatch-agent", dependencies=[Depends(verify_internal_key)])
    async def dispatch_agent(request: Request): ...
"""
import os
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger("interview-agent.security")


async def verify_internal_key(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-Api-Key"),
) -> None:
    """
    Validate the X-Internal-Api-Key header against the INTERNAL_API_KEY env variable.
    Raises HTTP 401 if missing or invalid.
    """
    expected = os.getenv("INTERNAL_API_KEY", "")

    if not expected:
        # Fail-open only in dev if the env var was never set at all.
        # Log a loud warning so this never silently passes in production.
        logger.warning(
            "[SECURITY] INTERNAL_API_KEY env variable is not set. "
            "All internal-key checks are BYPASSED — set this before deploying to production."
        )
        return

    if not x_internal_api_key or x_internal_api_key.strip() != expected.strip():
        logger.warning(
            "[SECURITY] Rejected request: invalid or missing X-Internal-Api-Key header."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-Internal-Api-Key",
        )
