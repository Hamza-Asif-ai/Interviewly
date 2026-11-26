"""FastAPI application entrypoint with Gradio app mounted at the root."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import init_db
from .routers import practice, prep
from .ui.gradio_app import build_gradio_app, APP_TITLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_request_windows: dict[str, deque[float]] = defaultdict(deque)
_analysis_slots = asyncio.Semaphore(max(1, settings.max_concurrent_analyses))
_EXPENSIVE_PATHS = ("/prep/generate", "/practice/upload")


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

cors_origins = settings.cors_origins or ["http://localhost:7860"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def guard_requests(request: Request, call_next):
    """Bound per-IP request rate and expensive-analysis concurrency."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _request_windows[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return JSONResponse(
            status_code=429, content={"detail": "Too many requests. Try again shortly."}
        )
    window.append(now)

    if not request.url.path.startswith(_EXPENSIVE_PATHS):
        return await call_next(request)
    try:
        await asyncio.wait_for(_analysis_slots.acquire(), timeout=0.2)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=429, content={"detail": "Server busy. Please retry shortly."}
        )
    try:
        return await call_next(request)
    finally:
        _analysis_slots.release()


app.include_router(prep.router)
app.include_router(practice.router)


@app.get("/health")
def health() -> dict:
    """Health check endpoint (kept minimal and fast for uptime monitors)."""
    return {"status": "ok"}


gradio_app = build_gradio_app()
app = gr.mount_gradio_app(
    app, gradio_app, path="/", max_file_size=settings.max_upload_bytes
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)