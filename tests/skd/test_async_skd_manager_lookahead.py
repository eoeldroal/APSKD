"""Unit tests for AsyncSkdAgentLoopManager lookahead scheduling."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.manager import AsyncSkdAgentLoopManager
from verl.experimental.async_skd.state import AsyncSkdSample, SkdCommittedUnit, SkdPartialState
from verl.protocol import DataProto


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLookaheadWorker:
    def __init__(
        self,
        *,
        name: str,
        base_delays: dict[int, float],
        lookahead_results: dict[str, list[AsyncSkdSample]],
        calls: list[tuple[str, str, Any]],
    ):
        self._name = name
        self._base_delays = base_delays
        self._lookahead_results = lookahead_results
        self._calls = calls
        self.generate_sequence_single = _RemoteMethod(self._generate_sequence_single)
        self.generate_skd_until_boundary = _RemoteMethod(self._generate_skd_until_boundary)

    async def _generate_sequence_single(self, sample: DataProto) -> DataProto:
        input_pos = int(sample.non_tensor_batch["input_pos"][0])
        self._calls.append((self._name, "base", input_pos))
        await asyncio.sleep(self._base_delays.get(input_pos, 0.0))
        return _make_output(input_pos)

    async def _generate_skd_until_boundary(
        self,
        batch: DataProto | None = None,
        *,
        partial_state: SkdPartialState | None = None,
        sample_id: str,
        logical_step: int,
        source_type: str,
    ) -> AsyncSkdSample:
        del logical_step, source_type
        if batch is not None:
            input_pos = int(batch.non_tensor_batch["input_pos"][0])
            self._calls.append((self._name, "lookahead", input_pos))
        else:
            assert partial_state is not None
            self._calls.append((self._name, "resume", sample_id))
        await asyncio.sleep(0)
        return self._lookahead_results[sample_id].pop(0)


def _make_prompts(batch_size: int) -> DataProto:
    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)},
        non_tensors={
            "input_pos": np.array(list(range(batch_size)), dtype=object),
            "preferred_worker": np.array([f"sample-{i}" for i in range(batch_size)], dtype=object),
        },
        meta_info={"global_steps": 3, "validate": False},
    )


def _make_source_sample(input_pos: int) -> DataProto:
    return DataProto.from_dict(
        non_tensors={
            "input_pos": np.array([input_pos], dtype=object),
            "preferred_worker": np.array([f"lookahead-{input_pos}"], dtype=object),
        },
        meta_info={"global_steps": 4, "validate": False},
    )


def _make_output(input_pos: int) -> DataProto:
    prompt_len = 2
    response_len = 3
    seq_len = prompt_len + response_len
    prompts = torch.tensor([[100 + input_pos, 200 + input_pos]], dtype=torch.long)
    responses = torch.tensor([[input_pos, input_pos + 10, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={
            "input_pos": np.array([input_pos], dtype=object),
            "payload": np.array([f"out-{input_pos}"], dtype=object),
        },
        meta_info={
            "metrics": [
                {
                    "generate_sequences": float(input_pos + 1),
                    "tool_calls": float(input_pos % 2),
                    "num_preempted": -1,
                }
            ]
        },
    )


def _make_completed_sample(sample_id: str, input_pos: int, logical_step: int = 4) -> AsyncSkdSample:
    return AsyncSkdSample.from_completed(
        sample_id=sample_id,
        logical_step=logical_step,
        source_type="lookahead",
        batch=_make_output(input_pos),
    )


def _make_partial(sample_id: str, logical_step: int = 4) -> SkdPartialState:
    return SkdPartialState(
        sample_id=sample_id,
        logical_step=logical_step,
        source_type="lookahead",
        agent_state="generating",
        last_committed_unit=SkdCommittedUnit.ASSISTANT_GEN_CHUNK.value,
        request_id=f"req-{sample_id}",
        response_ids=[1],
        response_mask=[1],
        rollout_birth_version=3,
        rollout_min_version=3,
        rollout_max_version=3,
        committed_gen_chunks=1,
        committed_env_units=0,
        committed_prefix_tokens=1,
        extra_fields={
            "teacher_ids_list": [[1, 0, 0, 0]],
            "teacher_logprobs_list": [[-1.0, 0.0, 0.0, 0.0]],
        },
    )


def _make_partial_sample(sample_id: str, logical_step: int = 4) -> AsyncSkdSample:
    return AsyncSkdSample.from_partial(partial_state=_make_partial(sample_id, logical_step=logical_step))


def _make_manager(
    *,
    prefetch_limit: int,
    source_items: list[tuple[str, DataProto]],
    lookahead_results: dict[str, list[AsyncSkdSample]],
    base_delays: dict[int, float] | None = None,
    rollout_n: int = 1,
):
    calls: list[tuple[str, str, Any]] = []
    manager = AsyncSkdAgentLoopManager.__new__(AsyncSkdAgentLoopManager)
    manager.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": rollout_n,
                    "agent": {
                        "async_skd_mode": "lookahead",
                        "async_skd_prefetch_limit": prefetch_limit,
                    },
                }
            }
        }
    )
    manager.rollout_config = OmegaConf.create({"n": rollout_n})
    manager.stream_teacher_with_rollout = False
    manager.agent_loop_workers = [
        _FakeLookaheadWorker(
            name="worker-0",
            base_delays=base_delays or {},
            lookahead_results=lookahead_results,
            calls=calls,
        ),
        _FakeLookaheadWorker(
            name="worker-1",
            base_delays=base_delays or {},
            lookahead_results=lookahead_results,
            calls=calls,
        ),
    ]
    manager._test_source_items = list(source_items)
    manager._test_carryover_partials = []

    def next_lookahead_sample(logical_step: int):
        del logical_step
        if not manager._test_source_items:
            return None
        return manager._test_source_items.pop(0)

    manager._next_lookahead_sample = next_lookahead_sample
    return manager, calls


@pytest.mark.asyncio
async def test_lookahead_manager_promotes_completed_samples_after_base_outputs():
    manager, calls = _make_manager(
        prefetch_limit=2,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
        ],
        lookahead_results={
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
        },
    )

    output = await manager.generate_sequences(_make_prompts(4))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3, 100, 101]
    assert output.non_tensor_batch["payload"].tolist() == [
        "out-0",
        "out-1",
        "out-2",
        "out-3",
        "out-100",
        "out-101",
    ]
    assert manager._async_skd_carryover_partials == []
    assert [call[1:] for call in calls].count(("lookahead", 100)) == 1
    assert [call[1:] for call in calls].count(("lookahead", 101)) == 1


@pytest.mark.asyncio
async def test_lookahead_manager_carries_partial_and_excludes_it_from_train_batch():
    manager, _ = _make_manager(
        prefetch_limit=2,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
        ],
        lookahead_results={
            "lookahead-100": [_make_partial_sample("lookahead-100")],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
        },
    )

    output = await manager.generate_sequences(_make_prompts(4))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3, 101]
    assert [partial.sample_id for partial in manager._async_skd_carryover_partials] == ["lookahead-100"]
    assert manager._next_fresh_quota(96) == 95


@pytest.mark.asyncio
async def test_lookahead_manager_does_not_continue_partial_after_base_barrier():
    manager, calls = _make_manager(
        prefetch_limit=1,
        source_items=[("lookahead-100", _make_source_sample(100))],
        lookahead_results={"lookahead-100": [_make_partial_sample("lookahead-100")]},
    )

    output = await manager.generate_sequences(_make_prompts(2))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert [partial.sample_id for partial in manager._async_skd_carryover_partials] == ["lookahead-100"]
    assert [call for call in calls if call[1] == "resume"] == []


@pytest.mark.asyncio
async def test_lookahead_manager_can_continue_partial_before_base_barrier_without_refilling_budget():
    manager, calls = _make_manager(
        prefetch_limit=1,
        source_items=[("lookahead-100", _make_source_sample(100))],
        lookahead_results={
            "lookahead-100": [
                _make_partial_sample("lookahead-100"),
                _make_completed_sample("lookahead-100", 100),
            ]
        },
        base_delays={1: 0.05},
    )

    output = await manager.generate_sequences(_make_prompts(2))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 100]
    assert manager._async_skd_carryover_partials == []
    assert [call[1] for call in calls].count("lookahead") == 1
    assert [call[1] for call in calls].count("resume") == 1
    assert manager._test_source_items == []


def test_lookahead_manager_next_fresh_quota_ignores_promoted_count():
    manager, _ = _make_manager(
        prefetch_limit=0,
        source_items=[],
        lookahead_results={},
    )
    manager._async_skd_carryover_partials = [_make_partial("a"), _make_partial("b")]

    assert manager._next_fresh_quota(96) == 94


@pytest.mark.asyncio
async def test_lookahead_manager_rejects_rollout_n_greater_than_one():
    manager, _ = _make_manager(
        prefetch_limit=1,
        source_items=[],
        lookahead_results={},
        rollout_n=2,
    )

    with pytest.raises(ValueError, match="rollout.n == 1"):
        await manager.generate_sequences(_make_prompts(1))
