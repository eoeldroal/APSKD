"""Unit tests for AsyncSkdAgentLoopManager lookahead scheduling."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.manager import AsyncSkdAgentLoopManager
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
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
        self.generate_skd_from_partial_to_completion = _RemoteMethod(self._generate_skd_from_partial_to_completion)

    async def _generate_sequence_single(self, sample: DataProto) -> DataProto:
        input_pos = int(sample.non_tensor_batch["input_pos"][0])
        self._calls.append((self._name, "base", input_pos))
        await asyncio.sleep(self._base_delays.get(input_pos, 0.0))
        output = _make_output(input_pos)
        output.meta_info["metrics"][0]["rollout_server_id"] = self._name
        return output

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
        sample = self._lookahead_results[sample_id].pop(0)
        if sample.kind == "completed":
            sample.require_completed().meta_info["metrics"][0]["rollout_server_id"] = self._name
        return sample

    async def _generate_skd_from_partial_to_completion(self, partial_state: SkdPartialState) -> AsyncSkdSample:
        self._calls.append((self._name, "carryover", partial_state.sample_id))
        await asyncio.sleep(0)
        sample = self._lookahead_results[partial_state.sample_id].pop(0)
        if sample.kind == "completed":
            sample.require_completed().meta_info["metrics"][0]["rollout_server_id"] = self._name
        return sample


class _FakeLookaheadSource:
    def __init__(self, source_items: list[tuple[str, DataProto]]):
        self.source_items = list(source_items)
        self.reserved_steps: list[int] = []
        self.promoted_samples: list[AsyncSkdSample] = []
        self.carryover_partials: list[SkdPartialState] = []

    def reserve_lookahead(self, logical_step: int) -> tuple[str, DataProto] | None:
        self.reserved_steps.append(logical_step)
        if not self.source_items:
            return None
        return self.source_items.pop(0)

    def record_promoted(self, samples: list[AsyncSkdSample]) -> None:
        self.promoted_samples.extend(samples)

    def record_carryover(self, partials: list[SkdPartialState]) -> None:
        self.carryover_partials.extend(partials)

    def next_fresh_quota(self, base_batch_size: int) -> int:
        return max(0, base_batch_size - len(self.carryover_partials))


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


def _make_completed_sample(
    sample_id: str,
    input_pos: int,
    logical_step: int = 4,
    source_type: str = "lookahead",
) -> AsyncSkdSample:
    return AsyncSkdSample.from_completed(
        sample_id=sample_id,
        logical_step=logical_step,
        source_type=source_type,
        batch=_make_output(input_pos),
    )


def _make_partial(sample_id: str, logical_step: int = 4, committed_gen_chunks: int = 1) -> SkdPartialState:
    return SkdPartialState(
        sample_id=sample_id,
        logical_step=logical_step,
        source_type="lookahead",
        agent_state="generating",
        request_id=f"req-{sample_id}",
        response_ids=[1],
        response_mask=[1],
        rollout_birth_version=3,
        rollout_min_version=3,
        rollout_max_version=3,
        committed_gen_chunks=committed_gen_chunks,
        committed_env_units=0,
        committed_prefix_tokens=1,
        extra_fields={
            "teacher_ids_list": [[1, 0, 0, 0]],
            "teacher_logprobs_list": [[-1.0, 0.0, 0.0, 0.0]],
        },
    )


def _make_partial_sample(
    sample_id: str,
    logical_step: int = 4,
    committed_gen_chunks: int = 1,
) -> AsyncSkdSample:
    return AsyncSkdSample.from_partial(
        partial_state=_make_partial(
            sample_id,
            logical_step=logical_step,
            committed_gen_chunks=committed_gen_chunks,
        )
    )


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
    source = _FakeLookaheadSource(source_items)
    manager.set_async_skd_data_source(source)
    return manager, calls, source


@pytest.mark.asyncio
async def test_manager_generates_carryover_and_fresh_current_work_in_stable_order():
    manager, calls, _ = _make_manager(
        prefetch_limit=0,
        source_items=[],
        lookahead_results={
            "carry-200": [_make_completed_sample("carry-200", 200, source_type="resumed_current")],
            "carry-201": [_make_completed_sample("carry-201", 201, source_type="resumed_current")],
        },
    )

    output = await manager.generate_sequences_with_carryover(
        fresh_prompts=_make_prompts(2),
        carryover_partials=[_make_partial("carry-200"), _make_partial("carry-201")],
    )

    assert output.non_tensor_batch["input_pos"].tolist() == [200, 201, 0, 1]
    assert output.non_tensor_batch["payload"].tolist() == ["out-200", "out-201", "out-0", "out-1"]
    assert [call[1:] for call in calls].count(("carryover", "carry-200")) == 1
    assert [call[1:] for call in calls].count(("carryover", "carry-201")) == 1
    assert [call[1:] for call in calls].count(("base", 0)) == 1
    assert [call[1:] for call in calls].count(("base", 1)) == 1


@pytest.mark.asyncio
async def test_manager_generates_only_carryover_current_work_without_fresh_prompts():
    manager, calls, _ = _make_manager(
        prefetch_limit=0,
        source_items=[],
        lookahead_results={
            "carry-200": [_make_completed_sample("carry-200", 200, source_type="resumed_current")],
        },
    )

    output = await manager.generate_sequences_with_carryover(
        fresh_prompts=None,
        carryover_partials=[_make_partial("carry-200")],
    )

    assert output.non_tensor_batch["input_pos"].tolist() == [200]
    assert [call[1:] for call in calls] == [("carryover", "carry-200")]


@pytest.mark.asyncio
async def test_manager_generate_sequences_with_carryover_rejects_rollout_n_greater_than_one():
    manager, _, _ = _make_manager(
        prefetch_limit=0,
        source_items=[],
        lookahead_results={},
        rollout_n=2,
    )

    with pytest.raises(ValueError, match="rollout.n == 1"):
        await manager.generate_sequences_with_carryover(
            fresh_prompts=_make_prompts(1),
            carryover_partials=[],
        )


@pytest.mark.asyncio
async def test_carryover_path_admits_lookahead_and_appends_promoted_after_current_work():
    manager, calls, source = _make_manager(
        prefetch_limit=2,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
        ],
        lookahead_results={
            "carry-200": [_make_completed_sample("carry-200", 200, source_type="resumed_current")],
            "carry-201": [_make_completed_sample("carry-201", 201, source_type="resumed_current")],
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
        },
        base_delays={0: 0.05, 1: 0.05},
    )

    output = await manager.generate_sequences_with_carryover(
        fresh_prompts=_make_prompts(2),
        carryover_partials=[_make_partial("carry-200"), _make_partial("carry-201")],
    )

    assert output.non_tensor_batch["input_pos"].tolist() == [200, 201, 0, 1, 100, 101]
    assert [sample.sample_id for sample in source.promoted_samples] == ["lookahead-100", "lookahead-101"]
    assert source.carryover_partials == []
    assert [call[1] for call in calls].count("lookahead") == 2
    assert output.meta_info["timing"]["async_skd/lookahead_started_count"] == 2


@pytest.mark.asyncio
async def test_carryover_path_drain_stops_lookahead_refill():
    manager, calls, source = _make_manager(
        prefetch_limit=8,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
            ("lookahead-102", _make_source_sample(102)),
            ("lookahead-103", _make_source_sample(103)),
        ],
        lookahead_results={
            "carry-200": [_make_completed_sample("carry-200", 200, source_type="resumed_current")],
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
            "lookahead-102": [_make_completed_sample("lookahead-102", 102)],
            "lookahead-103": [_make_completed_sample("lookahead-103", 103)],
        },
    )

    await manager.generate_sequences_with_carryover(
        fresh_prompts=None,
        carryover_partials=[_make_partial("carry-200")],
    )

    assert [call[1] for call in calls].count("lookahead") == 0
    assert len(source.source_items) == 4


@pytest.mark.asyncio
async def test_lookahead_manager_promotes_completed_samples_after_base_outputs():
    manager, calls, source = _make_manager(
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
    assert [sample.sample_id for sample in source.promoted_samples] == ["lookahead-100", "lookahead-101"]
    assert [call[1:] for call in calls].count(("lookahead", 100)) == 1
    assert [call[1:] for call in calls].count(("lookahead", 101)) == 1


@pytest.mark.asyncio
async def test_lookahead_manager_carries_partial_and_excludes_it_from_train_batch():
    manager, _, source = _make_manager(
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
    assert [sample.sample_id for sample in source.promoted_samples] == ["lookahead-101"]
    assert [partial.sample_id for partial in source.carryover_partials] == ["lookahead-100"]
    assert manager._next_fresh_quota(96) == 95


@pytest.mark.asyncio
async def test_lookahead_manager_does_not_continue_partial_after_base_barrier():
    manager, calls, source = _make_manager(
        prefetch_limit=1,
        source_items=[("lookahead-100", _make_source_sample(100))],
        lookahead_results={"lookahead-100": [_make_partial_sample("lookahead-100")]},
    )

    output = await manager.generate_sequences(_make_prompts(2))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert [partial.sample_id for partial in manager._async_skd_carryover_partials] == ["lookahead-100"]
    assert [partial.sample_id for partial in source.carryover_partials] == ["lookahead-100"]
    assert [call for call in calls if call[1] == "resume"] == []


@pytest.mark.asyncio
async def test_lookahead_manager_can_continue_partial_before_base_barrier_without_refilling_budget():
    manager, calls, source = _make_manager(
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
    assert source.source_items == []


@pytest.mark.asyncio
async def test_lookahead_manager_carries_partial_at_default_old_gen_chunk_cap():
    manager, calls, source = _make_manager(
        prefetch_limit=1,
        source_items=[("lookahead-100", _make_source_sample(100))],
        lookahead_results={
            "lookahead-100": [
                _make_partial_sample("lookahead-100", committed_gen_chunks=16),
                _make_completed_sample("lookahead-100", 100),
            ]
        },
        base_delays={1: 0.05},
    )

    output = await manager.generate_sequences(_make_prompts(2))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert [partial.sample_id for partial in manager._async_skd_carryover_partials] == ["lookahead-100"]
    assert [partial.sample_id for partial in source.carryover_partials] == ["lookahead-100"]
    assert [call for call in calls if call[1] == "resume"] == []


@pytest.mark.asyncio
async def test_lookahead_manager_records_source_promoted_and_carryover_samples():
    manager, _, source = _make_manager(
        prefetch_limit=2,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
        ],
        lookahead_results={
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_partial_sample("lookahead-101")],
        },
    )

    output = await manager.generate_sequences(_make_prompts(4))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3, 100]
    assert source.reserved_steps == [4, 4]
    assert [sample.sample_id for sample in source.promoted_samples] == ["lookahead-100"]
    assert [partial.sample_id for partial in source.carryover_partials] == ["lookahead-101"]
    assert [sample.sample_id for sample in manager._async_skd_last_promoted_samples] == ["lookahead-100"]
    assert [partial.sample_id for partial in manager._async_skd_carryover_partials] == ["lookahead-101"]
    assert manager._next_fresh_quota(96) == 95


def test_lookahead_manager_next_fresh_quota_ignores_promoted_count():
    manager, _, source = _make_manager(
        prefetch_limit=0,
        source_items=[],
        lookahead_results={},
    )
    source.record_carryover([_make_partial("a"), _make_partial("b")])

    assert manager._next_fresh_quota(96) == 94


@pytest.mark.asyncio
async def test_lookahead_refills_the_worker_that_frees_a_slot_first():
    manager, calls, source = _make_manager(
        prefetch_limit=4,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
            ("lookahead-102", _make_source_sample(102)),
            ("lookahead-103", _make_source_sample(103)),
        ],
        lookahead_results={
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
            "lookahead-102": [_make_completed_sample("lookahead-102", 102)],
            "lookahead-103": [_make_completed_sample("lookahead-103", 103)],
        },
        base_delays={2: 0.05, 3: 0.05},
    )

    output = await manager.generate_sequences(_make_prompts(4))

    lookahead_calls = [call for call in calls if call[1] == "lookahead"]
    worker0_lookahead = [call for call in lookahead_calls if call[0] == "worker-0"]
    worker1_lookahead = [call for call in lookahead_calls if call[0] == "worker-1"]

    assert len(worker0_lookahead) > len(worker1_lookahead)
    assert [sample.sample_id for sample in source.promoted_samples] == [
        "lookahead-100",
        "lookahead-101",
        "lookahead-102",
        "lookahead-103",
    ]
    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3, 100, 101, 102, 103]


@pytest.mark.asyncio
async def test_lookahead_refill_does_not_exceed_worker_capacity():
    manager, calls, _ = _make_manager(
        prefetch_limit=4,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
            ("lookahead-102", _make_source_sample(102)),
            ("lookahead-103", _make_source_sample(103)),
        ],
        lookahead_results={
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
            "lookahead-102": [_make_completed_sample("lookahead-102", 102)],
            "lookahead-103": [_make_completed_sample("lookahead-103", 103)],
        },
        base_delays={2: 0.05, 3: 0.05},
    )

    output = await manager.generate_sequences(_make_prompts(4))

    timing = output.meta_info["timing"]
    assert timing["async_skd/worker_capacity"] == 2
    assert timing["async_skd/worker_active_max"] <= 2
    assert timing["async_skd/lookahead_started_count"] == 4
    assert [call[1] for call in calls].count("lookahead") == 4


@pytest.mark.asyncio
async def test_lookahead_refill_stops_after_base_barrier_drain():
    manager, calls, source = _make_manager(
        prefetch_limit=8,
        source_items=[
            ("lookahead-100", _make_source_sample(100)),
            ("lookahead-101", _make_source_sample(101)),
            ("lookahead-102", _make_source_sample(102)),
            ("lookahead-103", _make_source_sample(103)),
            ("lookahead-104", _make_source_sample(104)),
            ("lookahead-105", _make_source_sample(105)),
        ],
        lookahead_results={
            "lookahead-100": [_make_completed_sample("lookahead-100", 100)],
            "lookahead-101": [_make_completed_sample("lookahead-101", 101)],
            "lookahead-102": [_make_completed_sample("lookahead-102", 102)],
            "lookahead-103": [_make_completed_sample("lookahead-103", 103)],
            "lookahead-104": [_make_completed_sample("lookahead-104", 104)],
            "lookahead-105": [_make_completed_sample("lookahead-105", 105)],
        },
    )

    await manager.generate_sequences(_make_prompts(2))

    lookahead_calls = [call for call in calls if call[1] == "lookahead"]
    assert len(lookahead_calls) <= 2
    assert len(source.source_items) >= 4


@pytest.mark.asyncio
async def test_lookahead_reports_worker_slot_and_server_distribution_metrics():
    manager, _, _ = _make_manager(
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
    timing = output.meta_info["timing"]

    assert timing["async_skd/worker_capacity"] == 2
    assert timing["async_skd/lookahead_started_count"] == 2
    assert "async_skd/worker_0_completed_count" in timing
    assert "async_skd/worker_1_completed_count" in timing


@pytest.mark.asyncio
async def test_lookahead_manager_rejects_rollout_n_greater_than_one():
    manager, _, _ = _make_manager(
        prefetch_limit=1,
        source_items=[],
        lookahead_results={},
        rollout_n=2,
    )

    with pytest.raises(ValueError, match="rollout.n == 1"):
        await manager.generate_sequences(_make_prompts(1))
