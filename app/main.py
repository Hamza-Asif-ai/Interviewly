"""FastAPI application entrypoint with Gradio app mounted at the root."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import practice, prep
from .ui.gradio_app import build_gradio_app, APP_TITLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Ensure the database schema exists before accepting requests."""
    logger.info("Initializing database at %s", settings.database_url)
    init_db()
    yield


app = FastAPI(
    title="Interviewly API",
    description="AI Interview Buddy & Performance Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prep.router)
app.include_router(practice.router)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "app": "interviewly"}


gradio_app = build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)