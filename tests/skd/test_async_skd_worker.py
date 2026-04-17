"""Unit tests for AsyncSkdAgentLoopWorker primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, _InternalAgentLoopOutput
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.protocol import DataProto
from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _to_internal


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


@pytest.mark.asyncio
async def test_async_skd_worker_generate_sequence_single_runs_exactly_one_sample_on_cpu():
    class _DummyWorker(AsyncSkdAgentLoopWorker):
        reward_loop_worker_handles = None
        distillation_enabled = False
        stream_teacher_with_rollout = False

        def __init__(self):
            self.rollout_config = OmegaConf.create(
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "calculate_log_probs": True,
                    "val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                    "agent": {"default_agent_loop": "default_agent"},
                }
            )
            self.calls = []

        async def _run_agent_loop(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            **kwargs,
        ) -> _InternalAgentLoopOutput:
            self.calls.append(
                {
                    "sampling_params": sampling_params,
                    "trajectory": trajectory,
                    "agent_name": agent_name,
                    "trace": trace,
                    "kwargs": kwargs,
                }
            )
            return _to_internal(
                output_prompt_ids=[101, 102],
                output_response_ids=[11, 12, 13],
                output_response_mask=[1, 1, 1],
                metrics=AgentLoopMetrics(),
                extra_fields={"custom_extra": "single"},
                num_turns=1,
                prompt_len=4,
                response_len=5,
            )

    raw_prompt = [{"role": "user", "content": "hi"}]
    batch = DataProto.from_dict(
        non_tensors={
            "raw_prompt": _object_array([raw_prompt]),
            "index": np.array([7], dtype=object),
        },
        meta_info={"global_steps": 12, "validate": True},
    )

    worker = _DummyWorker()
    output = await worker.generate_sequence_single(batch)

    assert len(worker.calls) == 1
    call = worker.calls[0]
    assert call["sampling_params"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": True,
    }
    assert call["trajectory"] == {"step": 12, "sample_index": 7, "rollout_n": 0, "validate": True}
    assert call["agent_name"] == "default_agent"
    assert call["kwargs"]["raw_prompt"] == raw_prompt

    assert len(output) == 1
    assert output.batch["prompts"].shape == (1, 4)
    assert output.batch["responses"].shape == (1, 5)
    assert output.non_tensor_batch["agent_name"].tolist() == ["default_agent"]
    assert output.non_tensor_batch["raw_prompt"][0] == raw_prompt
    assert output.non_tensor_batch["index"].tolist() == [7]
    assert output.non_tensor_batch["custom_extra"].tolist() == ["single"]
    assert output.meta_info["metrics"] == [AgentLoopMetrics().model_dump()]


@pytest.mark.asyncio
async def test_async_skd_worker_generate_sequence_single_rejects_multi_sample_batch_on_cpu():
    class _DummyWorker:
        generate_sequence_single = AsyncSkdAgentLoopWorker.generate_sequence_single

    batch = DataProto.from_dict(
        non_tensors={"index": np.array([0, 1], dtype=object)},
        meta_info={"global_steps": 12, "validate": False},
    )

    with pytest.raises(ValueError, match="exactly one sample"):
        await _DummyWorker().generate_sequence_single(batch)
