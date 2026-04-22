"""Lightweight JSONL event sink for async SKD runtime visualization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import time
from typing import Any, Iterator


_ASYNC_SKD_EVENT_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("async_skd_event_context", default={})


def _event_log_path() -> str | None:
    path = os.getenv("VERL_ASYNC_SKD_EVENT_LOG")
    return path if path else None


def is_async_skd_event_enabled() -> bool:
    return _event_log_path() is not None


def get_async_skd_event_context() -> dict[str, Any]:
    return dict(_ASYNC_SKD_EVENT_CONTEXT.get())


@contextmanager
def async_skd_event_context(**fields: Any) -> Iterator[None]:
    previous = get_async_skd_event_context()
    merged = {**previous, **{key: value for key, value in fields.items() if value is not None}}
    token = _ASYNC_SKD_EVENT_CONTEXT.set(merged)
    try:
        yield
    finally:
        _ASYNC_SKD_EVENT_CONTEXT.reset(token)


def emit_async_skd_event(event: str, **fields: Any) -> None:
    path = _event_log_path()
    if path is None:
        return

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "event": event,
        **get_async_skd_event_context(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
