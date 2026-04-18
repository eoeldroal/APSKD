from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLoadBalancer:
    def __init__(self):
        self.acquire_server = _RemoteMethod(self._acquire_server)
        self.release_server = _RemoteMethod(self._release_server)
        self.released = []

    async def _acquire_server(self, request_id: str):
        assert request_id == "sticky-request"
        return "server-a"

    def _release_server(self, server_id: str):
        self.released.append(server_id)


class _FakeServer:
    def __init__(self):
        self.generate = _RemoteMethod(self._generate)

    async def _generate(self, **kwargs):
        assert kwargs["prompt_ids"] == [1, 2, 3]
        return SimpleNamespace(extra_fields={"existing": "value"})


@pytest.mark.asyncio
async def test_async_llm_server_manager_records_rollout_server_id():
    load_balancer = _FakeLoadBalancer()
    server = _FakeServer()
    manager = AsyncLLMServerManager(
        config={},
        servers=[("server-a", server)],
        load_balancer_handle=load_balancer,
    )

    output = await manager.generate(
        "sticky-request",
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 1},
    )

    assert output.extra_fields["existing"] == "value"
    assert output.extra_fields["rollout_server_id"] == "server-a"
    assert load_balancer.released == ["server-a"]
