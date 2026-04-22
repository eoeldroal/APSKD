"""Tests for replica-level async SKD request events."""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.experimental.async_skd.events import async_skd_event_context
from verl.workers.rollout.replica import TokenOutput


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLoadBalancer:
    def __init__(self):
        self.acquire_server = _RemoteMethod(self._acquire_server)
        self.release_server = _RemoteMethod(self._release_server)
        self.released: list[str] = []

    async def _acquire_server(self, request_id: str) -> str:
        del request_id
        return "student-replica-0"

    def _release_server(self, server_id: str) -> None:
        self.released.append(server_id)


class _FakeServer:
    def __init__(self):
        self.generate = _RemoteMethod(self._generate)

    async def _generate(self, **kwargs) -> TokenOutput:
        del kwargs
        return TokenOutput(token_ids=[1, 2, 3], log_probs=None, stop_reason="length", extra_fields={})


@pytest.mark.asyncio
async def test_async_llm_server_manager_emits_replica_request_events(tmp_path, monkeypatch):
    event_path = tmp_path / "async_skd_events.jsonl"
    monkeypatch.setenv("VERL_ASYNC_SKD_EVENT_LOG", str(event_path))
    load_balancer = _FakeLoadBalancer()
    manager = AsyncLLMServerManager(
        config=OmegaConf.create({}),
        servers=[("student-replica-0", _FakeServer())],
        load_balancer_handle=load_balancer,
    )

    with async_skd_event_context(
        sample_id="sample-0",
        scheduler_worker_idx=1,
        source_type="base_current",
        barrier_role="current",
    ):
        output = await manager.generate(
            request_id="request-0",
            prompt_ids=[10, 11, 12],
            sampling_params={"max_tokens": 1},
        )

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["replica_request_start", "replica_request_finish"]
    assert events[0]["sample_id"] == "sample-0"
    assert events[0]["scheduler_worker_idx"] == 1
    assert events[0]["replica_id"] == "student-replica-0"
    assert events[0]["role"] == "student"
    assert events[0]["request_kind"] == "student_generate"
    assert "request_instance_id" in events[0]
    assert events[0]["prompt_len"] == 3
    assert events[1]["request_instance_id"] == events[0]["request_instance_id"]
    assert events[1]["request_kind"] == "student_generate"
    assert events[1]["status"] == "ok"
    assert events[1]["output_tokens"] == 3
    assert output.extra_fields["rollout_server_id"] == "student-replica-0"
    assert load_balancer.released == ["student-replica-0"]
