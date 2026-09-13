"""Entrypoint: launch the Interviewly FastAPI + Gradio server."""

from __future__ import annotations

import logging
import os

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))


def main() -> None:
    """Run the Interviewly server on 0.0.0.0:7860."""
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()