import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.controllers import health_controller, resume_controller, dispatch_controller

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: spawn LiveKit Agent worker process (agent.py dev)
    agent_process = None
    try:
        logger.info("Starting LiveKit Voice Agent worker process (agent.py dev)...")
        agent_process = subprocess.Popen([sys.executable, "agent.py", "dev"])
    except Exception as e:
        logger.error("Failed to start LiveKit Agent worker process: %s", e)

    yield

    # Shutdown: terminate agent process when FastAPI shuts down
    if agent_process:
        logger.info("Terminating LiveKit Voice Agent worker process...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except Exception:
            agent_process.kill()


app = FastAPI(
    title="Resume Normalization Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health_controller.router)
app.include_router(resume_controller.router)
app.include_router(dispatch_controller.router)
