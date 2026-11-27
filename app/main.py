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
from .limits import analysis_slots, client_ip, is_expensive_request
from .retention import cleanup_all, retention_loop
from .routers import practice, prep
from .ui.gradio_app import build_gradio_app, APP_TITLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_request_windows: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the database and start periodic privacy cleanup."""
    logger.info("Initializing database at %s", settings.database_url)
    init_db()
    await asyncio.to_thread(cleanup_all)
    task = asyncio.create_task(retention_loop())
    app.state.retention_task = task
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


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
    """Rate-limit expensive actions only, and cap concurrent analyses."""
    if not is_expensive_request(request):
        return await call_next(request)

    now = time.monotonic()
    ip = client_ip(request)
    window = _request_windows[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return JSONResponse(
            status_code=429, content={"detail": "Too many requests. Try again shortly."}
        )
    window.append(now)

    acquired = await asyncio.to_thread(analysis_slots.try_acquire)
    if not acquired:
        return JSONResponse(
            status_code=429, content={"detail": "Server busy. Please retry shortly."}
        )
    try:
        return await call_next(request)
    finally:
        analysis_slots.release()


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