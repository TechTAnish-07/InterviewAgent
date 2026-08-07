from fastapi import FastAPI
from app.controllers import health_controller, resume_controller

app = FastAPI(
    title="Resume Normalization Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_controller.router)
app.include_router(resume_controller.router)
