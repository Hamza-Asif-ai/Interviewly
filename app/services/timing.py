"""Strict wall-clock logging and hard timeouts for backend pipeline steps."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, TypeVar

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="interviewly-step"
)


class StepTimeoutError(TimeoutError):
    """A processing step exceeded its hard wall-clock budget."""


def step_budget(limit: float | None = None) -> float:
    """Return a positive per-step budget, never above the global cap."""
    requested = settings.pipeline_step_timeout if limit is None else limit
    return max(0.001, min(float(requested), float(settings.pipeline_step_timeout)))


@contextmanager
def log_step(name: str) -> Iterator[None]:
    """Log the start and exact duration of a pipeline step."""
    started = time.time()
    logger.info("Starting %s...", name)
    try:
        yield
    finally:
        logger.info("%s finished in %.2fs", name, time.time() - started)


def run_with_timeout(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = None,
    label: str = "step",
    **kwargs: Any,
) -> T:
    """Run blocking work off the caller thread and raise TimeoutError on overrun."""
    budget = step_budget(timeout)
    started = time.time()
    logger.info("Starting %s (hard timeout %.1fs)...", label, budget)
    future = _EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=budget)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        logger.error("%s exceeded %.1fs hard timeout; aborting step.", label, budget)
        raise StepTimeoutError(
            f"{label} exceeded the {budget:.0f}s hard timeout."
        ) from exc
    except Exception:
        logger.exception("%s failed after %.2fs", label, time.time() - started)
        raise
    finally:
        logger.info("%s finished in %.2fs", label, time.time() - started)


async def await_with_timeout(
    awaitable: Awaitable[T],
    *,
    timeout: float | None = None,
    label: str = "step",
) -> T:
    """Await work on the event loop with the same hard timeout guarantee."""
    budget = step_budget(timeout)
    started = time.time()
    logger.info("Starting %s (hard timeout %.1fs)...", label, budget)
    try:
        return await asyncio.wait_for(awaitable, timeout=budget)
    except asyncio.TimeoutError as exc:
        logger.error("%s exceeded %.1fs hard timeout; aborting step.", label, budget)
        raise StepTimeoutError(
            f"{label} exceeded the {budget:.0f}s hard timeout."
        ) from exc
    finally:
        logger.info("%s finished in %.2fs", label, time.time() - started)
