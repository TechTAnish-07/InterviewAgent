import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.controllers import health_controller, resume_controller, dispatch_controller

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # LiveKit Agent worker is started by start.sh (or as a separate Render worker service).
    # Do NOT spawn it here — doing so would create a duplicate worker process,
    # doubling memory usage and opening a second internal HTTP port.
    logger.info("FastAPI startup complete. LiveKit worker managed externally.")
    yield
    logger.info("FastAPI shutdown.")



app = FastAPI(
    title="Resume Normalization Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health_controller.router)
app.include_router(resume_controller.router)
app.include_router(dispatch_controller.router)
