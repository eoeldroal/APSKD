"""Unit tests for AsyncSkdAgentLoopWorker boundary execution."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.protocol import DataProto
from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _FakeTokenizer


LOSS_TOP_K = 4


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def make_single_batch() -> DataProto:
    raw_prompt = [{"role": "user", "content": "hi"}]
    return DataProto.from_dict(
        non_tensors={
            "raw_prompt": _object_array([raw_prompt]),
            "index": np.array([7], dtype=object),
            "agent_name": np.array(["skd_agent"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
        },
        meta_info={"global_steps": 12, "validate": False},
    )


def make_partial() -> SkdPartialState:
    raw_prompt = [{"role": "user", "content": "hi"}]
    return SkdPartialState(
        sample_id="partial-sample",
        logical_step=12,
        source_type="lookahead",
        agent_state="generating",
        request_id="req-partial",
        messages=raw_prompt,
        prompt_ids=[1, 2, 3, 10],
        teacher_prompt_ids=[1, 2, 3, 10],
        response_ids=[10],
        response_mask=[1],
        assistant_turns=0,
        user_turns=0,
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=7,
        committed_gen_chunks=1,
        committed_env_units=0,
        committed_prefix_tokens=1,
        extra_fields={
            "teacher_prompt_ids": [1, 2, 3, 10],
            "teacher_ids_list": [[10, 0, 0, 0]],
            "teacher_logprobs_list": [[-1.0] * LOSS_TOP_K],
            "skd_pending_turn_response_ids": [10],
            "skd_committed_gen_chunks": 1,
            "skd_committed_env_units": 0,
            "skd_committed_prefix_tokens": 1,
            "rollout_birth_version": 7,
            "rollout_min_version": 7,
            "rollout_max_version": 7,
            "raw_prompt": raw_prompt,
            "async_skd_input_non_tensor_batch": {
                "raw_prompt": _object_array([raw_prompt]),
                "index": np.array([7], dtype=object),
                "agent_name": np.array(["skd_agent"], dtype=object),
                "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
            },
        },
    )


class _DummyWorker(AsyncSkdAgentLoopWorker):
    reward_loop_worker_handles = None
    distillation_enabled = False
    stream_teacher_with_rollout = False
    processor = None

    def __init__(self, loop_result: AgentLoopOutput | SkdPartialState):
        self.rollout_config = OmegaConf.create(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "calculate_log_probs": False,
                "prompt_length": 4,
                "response_length": 4,
                "val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                "agent": {"default_agent_loop": "skd_agent"},
            }
        )
        self.tokenizer = _FakeTokenizer()
        self.loop = SkdAgentLoop.__new__(SkdAgentLoop)
        self.loop.calls = []

        async def fake_run_until_exportable_boundary(
            loop_self,
            sampling_params: dict[str, Any],
            *,
            sample_id: str,
            logical_step: int,
            source_type: str,
            partial_state: SkdPartialState | None = None,
            **kwargs: Any,
        ):
            loop_self.calls.append(
                {
                    "sampling_params": sampling_params,
                    "sample_id": sample_id,
                    "logical_step": logical_step,
                    "source_type": source_type,
                    "partial_state": partial_state,
                    "kwargs": kwargs,
                }
            )
            return loop_result

        self.loop.run_until_exportable_boundary = MethodType(fake_run_until_exportable_boundary, self.loop)

        async def fake_run_from_partial_to_completion(
            loop_self,
            sampling_params: dict[str, Any],
            *,
            partial_state: SkdPartialState,
        ):
            loop_self.calls.append(
                {
                    "sampling_params": sampling_params,
                    "partial_state": partial_state,
                    "completion": True,
                }
            )
            return loop_result

        self.loop.run_from_partial_to_completion = MethodType(fake_run_from_partial_to_completion, self.loop)

    def _get_or_create_agent_loop(self, agent_name: str):
        assert agent_name == "skd_agent"
        return self.loop


@pytest.mark.asyncio
async def test_generate_skd_until_boundary_wraps_partial_result_from_fresh_batch():
    partial = make_partial()
    worker = _DummyWorker(partial)

    result = await worker.generate_skd_until_boundary(
        make_single_batch(),
        sample_id="fresh-sample",
        logical_step=12,
        source_type="lookahead",
    )

    assert isinstance(result, AsyncSkdSample)
    assert result.kind == "partial"
    assert result.require_partial() is partial
    assert partial.extra_fields["async_skd_input_non_tensor_batch"]["index"].tolist() == [7]
    assert worker.loop.calls[0]["sample_id"] == "fresh-sample"
    assert worker.loop.calls[0]["logical_step"] == 12
    assert worker.loop.calls[0]["source_type"] == "lookahead"
    assert worker.loop.calls[0]["partial_state"] is None
    assert worker.loop.calls[0]["kwargs"]["raw_prompt"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_generate_skd_until_boundary_wraps_completed_result_from_partial_state():
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 20],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": []},
    )
    partial = make_partial()
    worker = _DummyWorker(output)

    result = await worker.generate_skd_until_boundary(
        None,
        partial_state=partial,
        sample_id="partial-sample",
        logical_step=12,
        source_type="resumed_current",
    )

    assert result.kind == "completed"
    batch = result.require_completed()
    assert len(batch) == 1
    assert batch.non_tensor_batch["index"].tolist() == [7]
    assert batch.non_tensor_batch["reward_model"][0] == {"ground_truth": "42"}
    assert batch.batch["responses"].shape == (1, 4)
    assert worker.loop.calls[0]["partial_state"] is partial


@pytest.mark.asyncio
async def test_generate_skd_from_partial_to_completion_returns_completed_resumed_sample():
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 20],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": []},
    )
    partial = make_partial()
    worker = _DummyWorker(output)

    result = await worker.generate_skd_from_partial_to_completion(partial)

    assert result.kind == "completed"
    assert result.sample_id == "partial-sample"
    assert result.logical_step == 12
    assert result.source_type == "resumed_current"
    batch = result.require_completed()
    assert batch.non_tensor_batch["index"].tolist() == [7]
    assert batch.non_tensor_batch["reward_model"][0] == {"ground_truth": "42"}
    assert batch.batch["responses"].shape == (1, 4)
    assert worker.loop.calls[0]["partial_state"] is partial
    assert worker.loop.calls[0]["completion"] is True


@pytest.mark.asyncio
async def test_generate_skd_until_boundary_rejects_invalid_input_pairing():
    worker = _DummyWorker(make_partial())

    with pytest.raises(ValueError, match="exactly one of batch or partial_state"):
        await worker.generate_skd_until_boundary(
            make_single_batch(),
            partial_state=make_partial(),
            sample_id="bad",
            logical_step=1,
            source_type="lookahead",
        )

    with pytest.raises(ValueError, match="exactly one of batch or partial_state"):
        await worker.generate_skd_until_boundary(
            None,
            sample_id="bad",
            logical_step=1,
            source_type="lookahead",
        )
