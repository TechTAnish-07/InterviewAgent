from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
@router.head("/")
async def root():
    return {"status": "ok", "service": "interview-agent-api"}


@router.get("/health")
@router.head("/health")
async def health_check():
    return {"status": "ok"}

