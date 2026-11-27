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

_PROXY_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "fc00::/7",
        "fe80::/10",
        "::1/128",
    )
)


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _is_proxy_peer(peer: str) -> bool:
    """Whether the direct peer is a local/reverse proxy rather than a client."""
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(address in network for network in _PROXY_NETWORKS)


def client_ip(request: Request) -> str:
    """Resolve the client IP without letting clients pick their own bucket.

    Render's edge is Cloudflare: it sets ``CF-Connecting-IP`` and rejects client
    requests that supply that header, so it is safe when the direct peer is the
    proxy (private/loopback address, or an explicitly configured trusted proxy).
    ``X-Forwarded-For`` is append-only and client-forgeable on Render, so it is
    only consulted for explicitly configured proxies.
    """
    peer = request.client.host if request.client else "unknown"
    trusted = set(settings.trusted_proxy_ips)
    explicit_proxy = peer in trusted
    if explicit_proxy or _is_proxy_peer(peer):
        cf_ip = request.headers.get("cf-connecting-ip", "").strip()
        if _is_ip(cf_ip):
            return cf_ip
        if explicit_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            for candidate in reversed(forwarded.split(",")):
                candidate = candidate.strip()
                if candidate and candidate not in trusted and _is_ip(candidate):
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
        or path.startswith("/gradio_api/call/")
        or path == "/gradio_api/queue/join"
        or path.startswith("/run/")
        or path == "/queue/join"
    )
