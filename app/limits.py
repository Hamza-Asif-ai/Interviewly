"""Shared request-limit helpers for the API and the Gradio event handlers."""

from __future__ import annotations

import ipaddress
import threading

from fastapi import Request

from .config import settings


class AnalysisSlots:
    """A non-blocking concurrency cap usable from both async and sync code."""

    def __init__(self, capacity: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, capacity))

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()


analysis_slots = AnalysisSlots(settings.max_concurrent_analyses)


def client_ip(request: Request) -> str:
    """Resolve the client IP, trusting forwarded headers only from known proxies.

    Render terminates TLS at its own proxy, so the peer address is the proxy's.
    Only then is ``X-Forwarded-For`` consulted, walking right-to-left and
    skipping trusted hops so a client cannot spoof its own rate-limit bucket.
    """
    peer = request.client.host if request.client else "unknown"
    trusted = set(settings.trusted_proxy_ips)
    forwarded = request.headers.get("x-forwarded-for", "")
    if peer in trusted and forwarded:
        for candidate in reversed(forwarded.split(",")):
            candidate = candidate.strip()
            if not candidate or candidate in trusted:
                continue
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            return candidate
    return peer


def is_expensive_request(request: Request) -> bool:
    """Whether a request triggers prep/practice work rather than UI plumbing."""
    if request.method != "POST":
        return False
    path = request.url.path.rstrip("/")
    if path in ("/prep/generate", "/practice/upload"):
        return True
    return (
        path.startswith("/gradio_api/run/")
        or path == "/gradio_api/queue/join"
        or path.startswith("/run/")
        or path == "/queue/join"
    )
