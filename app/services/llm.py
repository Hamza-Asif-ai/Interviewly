"""Shared OpenAI-compatible LLM helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI, OpenAI

from ..config import settings

logger = logging.getLogger(__name__)

_CACHED_CLIENT: OpenAI | None = None
_CACHED_ASYNC_CLIENT: AsyncOpenAI | None = None

MODEL_FALLBACKS: List[str] = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    # Verified to serve content on new accounts if every 1.5/2.x is retired.
    "gemini-flash-latest",
]

_WORKING_MODEL: str | None = None


def _normalize_model(model: str) -> str:
    """Strip any 'models/' prefix (e.g. 'models/gemini-2.0-flash')."""
    prefix = "models/"
    if model.lower().startswith(prefix):
        return model[len(prefix):]
    return model


def _model_candidates(preferred: str | None) -> List[str]:
    """Return [preferred, (working), *fallbacks], de-duplicated, prefix-free.

    A model that already succeeded in this process is tried first so a
    retired model name only costs one failed call per run, not per request.
    """
    candidates: List[str] = []
    for raw in (_WORKING_MODEL, preferred or settings.llm_model):
        if raw:
            normalized = _normalize_model(raw)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
    for fallback in MODEL_FALLBACKS:
        normalized = _normalize_model(fallback)
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates or [_normalize_model(settings.llm_model)]


def _remember_working_model(candidate: str, primary: str) -> None:
    """Cache a successfully used fallback model for subsequent calls."""
    global _WORKING_MODEL
    if candidate != primary:
        _WORKING_MODEL = candidate
        logger.warning(
            "LLM primary model %r failed; using fallback %r.",
            primary,
            candidate,
        )


def get_client() -> OpenAI:
    """Return a cached synchronous OpenAI-compatible client."""
    global _CACHED_CLIENT
    if _CACHED_CLIENT is None:
        _CACHED_CLIENT = OpenAI(
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_base_url,
        )
    return _CACHED_CLIENT


def get_async_client() -> AsyncOpenAI:
    """Return a cached asynchronous OpenAI-compatible client."""
    global _CACHED_ASYNC_CLIENT
    if _CACHED_ASYNC_CLIENT is None:
        _CACHED_ASYNC_CLIENT = AsyncOpenAI(
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_base_url,
        )
    return _CACHED_ASYNC_CLIENT


def chat_completion(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float | None = None,
) -> str:
    """Run a chat completion synchronously and return the message content.

    Useful for short synchronous calls. Prefer the async variant in
    long-running request handlers.
    """
    client = get_client()
    candidates = _model_candidates(model or settings.llm_model)
    primary = candidates[0]
    last_error: Exception | None = None
    last_empty: str | None = None
    per_request_timeout = timeout if timeout is not None else settings.llm_timeout
    for candidate in candidates:
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=per_request_timeout,
            )
            content = response.choices[0].message.content or ""
            if content:
                _remember_working_model(candidate, primary)
                return content
            if last_empty is None:
                last_empty = content
            logger.warning("LLM model %r returned empty content; trying next.", candidate)
        except Exception as exc:  # noqa: BLE001 - try next fallback model
            last_error = exc
            logger.warning("LLM chat completion with model %r failed: %s", candidate, exc)
    if last_empty is not None:
        return last_empty
    raise last_error  # type: ignore[misc]


async def achat_completion(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float | None = None,
) -> str:
    """Run a chat completion asynchronously and return the message content."""
    client = get_async_client()
    candidates = _model_candidates(model or settings.llm_model)
    primary = candidates[0]
    last_error: Exception | None = None
    last_empty: str | None = None
    per_request_timeout = timeout if timeout is not None else settings.llm_timeout
    for candidate in candidates:
        try:
            response = await client.chat.completions.create(
                model=candidate,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=per_request_timeout,
            )
            content = response.choices[0].message.content or ""
            if content:
                _remember_working_model(candidate, primary)
                return content
            if last_empty is None:
                last_empty = content
            logger.warning("LLM model %r returned empty content; trying next.", candidate)
        except Exception as exc:  # noqa: BLE001 - try next fallback model
            last_error = exc
            logger.warning("LLM chat completion with model %r failed: %s", candidate, exc)
    if last_empty is not None:
        return last_empty
    raise last_error  # type: ignore[misc]


def parse_json_response(content: str) -> Any:
    """Parse a strict-JSON response from the model, tolerating markdown fences."""
    text = (content or "").strip()
    if not text:
        raise ValueError("LLM returned empty content; expected a JSON object.")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning(
            "Could not parse JSON (len=%d). Tail ends with '}': %s. Snippet=%r",
            len(text),
            text.endswith("}"),
            text[:400],
        )
        raise ValueError(
            f"LLM did not return valid JSON (len={len(text)}). "
            f"Original error: {exc}"
        ) from exc


async def extract_faithful_json(content: str) -> Any:
    """Parse JSON returned by the model, retrying a strictly-scoped re-prompt."""
    if not content or not content.strip():
        raise ValueError("LLM returned empty content; expected a JSON object.")
    try:
        return parse_json_response(content)
    except ValueError as original:
        logger.warning(
            "LLM returned invalid JSON (len=%d); requesting repairs. Snippet=%r",
            len(content),
            content[:500],
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You output only valid JSON: a single, complete JSON object and "
                    "nothing else. No code fences, no explanation, no text before or after."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The JSON below is invalid, truncated, or contains extra text. Fix it "
                    "and return ONLY the corrected, complete JSON object.\n"
                    f"{content}"
                ),
            },
        ]
        repaired = await achat_completion(
            messages, temperature=0.0, max_tokens=4096
        )
        if not repaired or not repaired.strip():
            raise ValueError(
                "LLM returned empty content during JSON repair."
            ) from original
        logger.info(
            "JSON repair returned %d chars, ends with '}': %s",
            len(repaired),
            repaired.rstrip().endswith("}"),
        )
        try:
            return parse_json_response(repaired)
        except ValueError as exc:
            raise ValueError(
                f"LLM failed to return valid JSON after repair. "
                f"Original error: {original}; Repair error: {exc}"
            ) from exc