"""Tests for async SKD runtime event emission."""

from __future__ import annotations

import json

from verl.experimental.async_skd.events import (
    async_skd_event_context,
    emit_async_skd_event,
    get_async_skd_event_context,
    is_async_skd_event_enabled,
)


def test_emit_async_skd_event_writes_jsonl_when_path_is_configured(tmp_path, monkeypatch):
    event_path = tmp_path / "async_skd_events.jsonl"
    monkeypatch.setenv("VERL_ASYNC_SKD_EVENT_LOG", str(event_path))

    emit_async_skd_event("sample_launch", sample_id="s0", replica_id="replica-0", active=3)

    [line] = event_path.read_text().splitlines()
    event = json.loads(line)
    assert event["event"] == "sample_launch"
    assert event["sample_id"] == "s0"
    assert event["replica_id"] == "replica-0"
    assert event["active"] == 3
    assert "ts" in event
    assert "pid" in event


def test_emit_async_skd_event_is_noop_without_path(tmp_path, monkeypatch):
    monkeypatch.delenv("VERL_ASYNC_SKD_EVENT_LOG", raising=False)

    emit_async_skd_event("sample_launch", sample_id="s0")

    assert not list(tmp_path.iterdir())
    assert not is_async_skd_event_enabled()


def test_async_skd_event_context_is_merged_into_events(tmp_path, monkeypatch):
    event_path = tmp_path / "async_skd_events.jsonl"
    monkeypatch.setenv("VERL_ASYNC_SKD_EVENT_LOG", str(event_path))

    with async_skd_event_context(sample_id="s1", scheduler_worker_idx=2, source_type="lookahead"):
        assert get_async_skd_event_context()["sample_id"] == "s1"
        emit_async_skd_event("replica_request_start", role="student", replica_id="server-0")

    [line] = event_path.read_text().splitlines()
    event = json.loads(line)
    assert event["event"] == "replica_request_start"
    assert event["sample_id"] == "s1"
    assert event["scheduler_worker_idx"] == 2
    assert event["source_type"] == "lookahead"
    assert event["role"] == "student"
    assert event["replica_id"] == "server-0"
