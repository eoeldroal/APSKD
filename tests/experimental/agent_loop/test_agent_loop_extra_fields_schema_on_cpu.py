# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data
        # Return a short, deterministic "generation" for testing.
        return TokenOutput(token_ids=prompt_ids[-1:] + [11, 12, 13], log_probs=[0.0, 0.0, 0.0, 0.0])

    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        # Minimal tokenization: return a small prompt.
        return [101, 102]

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [0] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [0] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _to_internal(
    *,
    output_prompt_ids: list[int],
    output_response_ids: list[int],
    output_response_mask: list[int],
    metrics: AgentLoopMetrics,
    extra_fields: dict[str, Any],
    num_turns: int,
    prompt_len: int,
    response_len: int,
) -> _InternalAgentLoopOutput:
    prompt_ids = _pad_1d(output_prompt_ids, length=prompt_len, pad_id=0)
    response_ids = _pad_1d(output_response_ids, length=response_len, pad_id=0)
    response_mask = _pad_1d(output_response_mask, length=response_len, pad_id=0)

    seq_len = prompt_len + response_len
    attention_mask = _pad_1d([1] * len(output_prompt_ids), length=prompt_len, pad_id=0) + _pad_1d(
        [1] * len(output_response_ids),
        length=response_len,
        pad_id=0,
    )
    input_ids = prompt_ids + response_ids
    position_ids = list(range(seq_len))

    def t(x: list[int]) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    return _InternalAgentLoopOutput(
        prompt_ids=t(prompt_ids),
        response_ids=t(response_ids),
        response_mask=t(response_mask),
        attention_mask=t(attention_mask),
        input_ids=t(input_ids),
        position_ids=t(position_ids),
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=None,
        num_turns=num_turns,
        metrics=metrics,
        extra_fields=extra_fields,
    )


@pytest.mark.asyncio
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu():
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    processor = None

    trainer_config = DictConfigWrap(config)
    data_config = DictConfigWrap(config.data)

    single_turn = SingleTurnAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        data_config=data_config,
    )

    raw_prompt = [{"role": "user", "content": "hi"}]
    sampling_params: dict[str, Any] = {}

    out = await single_turn.run(sampling_params=sampling_params, raw_prompt=raw_prompt)

    # Agent loop outputs should always contain these fields with consistent types.
    assert out.extra_fields["turn_scores"] == []
    assert out.extra_fields["tool_rewards"] == []

    internal_a = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields=out.extra_fields,
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    # Mimic two "worker chunks" and concatenate as in training.
    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False, "stream_teacher_with_rollout": False},
    )()
    merged = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a],
        input_non_tensor_batch={
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["single_turn_agent"], dtype=object),
        },
    )

    # Stable schema: present regardless of which loop produced a sample.
    stable_keys = (
        "turn_scores",
        "tool_rewards",
        "min_global_steps",
        "max_global_steps",
        "extras",
    )
    for key in stable_keys:
        assert key in merged.non_tensor_batch, f"missing key in merged batch: {key}"
        assert merged.non_tensor_batch[key].shape == (1,), (
            f"invalid shape for {key}: {merged.non_tensor_batch[key].shape}"
        )

    # And the list-typed fields are actually lists (not missing / scalar).
    assert merged.non_tensor_batch["turn_scores"][0] == []
    assert merged.non_tensor_batch["tool_rewards"][0] == []


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = False
        stream_teacher_with_rollout = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.reward_loop_worker_handles = None

    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        internal = await AgentLoopWorker._agent_loop_postprocess(
            _DummyWorker(),
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    expected = torch.tensor(routed_experts.copy()).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


def test_skd_teacher_reconstruction_matches_standard_left_shift_contract():
    class _DummyWorker:
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        stream_teacher_with_rollout = False

    output = AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": [[111, 112], [221, 222]],
            "teacher_logprobs_list": [[-1.0, -2.0], [-3.0, -4.0]],
        },
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            _DummyWorker(),
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    teacher_ids = output.extra_fields["teacher_ids"]
    teacher_logprobs = output.extra_fields["teacher_logprobs"]

    assert teacher_ids.tolist() == [
        [0, 0],
        [0, 0],
        [111, 112],
        [221, 222],
        [0, 0],
    ]
    assert teacher_logprobs.tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-1.0, -2.0],
        [-3.0, -4.0],
        [0.0, 0.0],
    ]

    prompt_len = len(output.prompt_ids)
    response_len = len(output.response_ids)
    response_slice = teacher_ids[prompt_len - 1 : prompt_len + response_len - 1]
    response_logprob_slice = teacher_logprobs[prompt_len - 1 : prompt_len + response_len - 1]
    assert response_slice.tolist() == [[111, 112], [221, 222]]
    assert response_logprob_slice.tolist() == [[-1.0, -2.0], [-3.0, -4.0]]


def test_skd_teacher_reconstruction_preserves_dummy_rows_for_tool_spans():
    class _DummyWorker:
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        stream_teacher_with_rollout = False

    output = AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12, 90, 91, 13],
        response_mask=[1, 1, 0, 0, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": [[111, 112], [221, 222], [0, 0], [0, 0], [331, 332]],
            "teacher_logprobs_list": [[-1.0, -2.0], [-3.0, -4.0], [0.0, 0.0], [0.0, 0.0], [-5.0, -6.0]],
        },
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            _DummyWorker(),
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    prompt_len = len(output.prompt_ids)
    response_len = len(output.response_ids)
    response_slice = output.extra_fields["teacher_ids"][prompt_len - 1 : prompt_len + response_len - 1]
    response_logprob_slice = output.extra_fields["teacher_logprobs"][prompt_len - 1 : prompt_len + response_len - 1]
    assert response_slice.tolist() == [[111, 112], [221, 222], [0, 0], [0, 0], [331, 332]]
    assert response_logprob_slice.tolist() == [[-1.0, -2.0], [-3.0, -4.0], [0.0, 0.0], [0.0, 0.0], [-5.0, -6.0]]


def test_skd_processing_tools_appends_dummy_teacher_rows(monkeypatch):
    async def _fake_processing(self, agent_data):
        del self
        agent_data.prompt_ids += [71, 72]
        agent_data.response_mask += [0, 0]
        return AgentState.GENERATING

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", _fake_processing)

    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    loop.loss_top_k = 3
    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="req-tools",
        tools_kwargs={},
    )
    agent_data.response_mask = [1]
    agent_data.prompt_ids = [41]
    agent_data.extra_fields["teacher_prompt_ids"] = [91]
    agent_data.extra_fields["teacher_ids_list"] = [[11, 12, 13]]
    agent_data.extra_fields["teacher_logprobs_list"] = [[-1.0, -2.0, -3.0]]

    next_state = asyncio.run(SkdAgentLoop._handle_processing_tools_state(loop, agent_data))

    assert next_state == AgentState.GENERATING
    assert agent_data.response_mask == [1, 0, 0]
    assert agent_data.prompt_ids == [41, 71, 72]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [91, 71, 72]
    assert agent_data.extra_fields["teacher_ids_list"] == [[11, 12, 13], [0, 0, 0], [0, 0, 0]]
    assert agent_data.extra_fields["teacher_logprobs_list"] == [
        [-1.0, -2.0, -3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_skd_interaction_appends_dummy_teacher_rows(monkeypatch):
    async def _fake_interacting(self, agent_data):
        del self
        agent_data.prompt_ids += [81, 82, 83]
        agent_data.response_mask += [0, 0, 0]
        return AgentState.TERMINATED

    monkeypatch.setattr(ToolAgentLoop, "_handle_interacting_state", _fake_interacting)

    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    loop.loss_top_k = 2
    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="req-interaction",
        tools_kwargs={},
    )
    agent_data.response_mask = [1, 1]
    agent_data.prompt_ids = [51, 52]
    agent_data.extra_fields["teacher_prompt_ids"] = [151, 152]
    agent_data.extra_fields["teacher_ids_list"] = [[10, 11], [20, 21]]
    agent_data.extra_fields["teacher_logprobs_list"] = [[-1.0, -1.1], [-2.0, -2.1]]

    next_state = asyncio.run(SkdAgentLoop._handle_interacting_state(loop, agent_data))

    assert next_state == AgentState.TERMINATED
    assert agent_data.response_mask == [1, 1, 0, 0, 0]
    assert agent_data.prompt_ids == [51, 52, 81, 82, 83]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [151, 152, 81, 82, 83]
    assert agent_data.extra_fields["teacher_ids_list"] == [[10, 11], [20, 21], [0, 0], [0, 0], [0, 0]]
    assert agent_data.extra_fields["teacher_logprobs_list"] == [
        [-1.0, -1.1],
        [-2.0, -2.1],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_skd_pending_initializes_teacher_prompt_stream():
    captured_system_contents = []

    async def _fake_apply_chat_template(messages, tools=None, images=None, videos=None):
        del tools, images, videos
        first_message = messages[0] if messages else {}
        if first_message.get("role") == "system":
            captured_system_contents.append(first_message.get("content"))
        return [10 * len(captured_system_contents)]

    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    loop.tool_schemas = []
    loop.teacher_system_prompt = "Teacher-only note."
    loop.apply_chat_template = _fake_apply_chat_template

    agent_data = AgentData(
        messages=[
            {"role": "system", "content": "Student system."},
            {"role": "user", "content": "Question"},
        ],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="req-pending",
        tools_kwargs={},
    )

    next_state = asyncio.run(SkdAgentLoop._handle_pending_state(loop, agent_data, {}))

    assert next_state == AgentState.GENERATING
    assert agent_data.prompt_ids == [10]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [20]
    assert captured_system_contents == [
        "Student system.",
        "Student system.\n\nTeacher-only note.",
    ]


def test_skd_generate_uses_teacher_prompt_stream():
    class _FakeServerManager:
        async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
            del request_id, prompt_ids, sampling_params, image_data, video_data
            return TokenOutput(token_ids=[42], num_preempted=0, extra_fields={})

    class _FakeTeacherManager:
        def __init__(self):
            self.sequence_ids = None
            self.logprob_start_len = None

        async def compute_teacher_logprobs_single(
            self, sequence_ids, request_id=None, logprob_start_len=0, multi_modal_data=None
        ):
            del request_id, multi_modal_data
            self.sequence_ids = sequence_ids
            self.logprob_start_len = logprob_start_len
            return torch.tensor([[42, 102, 103]], dtype=torch.int32), torch.tensor([[-1.0, -2.0, -3.0]])

    class _FakeToolParser:
        async def extract_tool_calls(self, response_ids, tools):
            del response_ids, tools
            return None, []

    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    loop.teacher_server_manager = _FakeTeacherManager()
    loop.server_manager = _FakeServerManager()
    loop.tokenizer = type("_Tok", (), {"eos_token_id": 999})()
    loop.response_length = 8
    loop.skd_chunk_size = 4
    loop.skd_verify_top_k = 3
    loop.max_chunks_per_sample = 1
    loop.loss_top_k = 3
    loop.tools = {}
    loop.tool_parser = _FakeToolParser()
    loop.interaction_config_file = None
    loop.max_assistant_turns = None
    loop.max_user_turns = None

    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="req-generate",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [10, 11, 12, 13]

    next_state = asyncio.run(SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False))

    assert next_state == AgentState.TERMINATED
    assert loop.teacher_server_manager.sequence_ids == [10, 11, 12, 13, 42]
    assert loop.teacher_server_manager.logprob_start_len == 3
    assert agent_data.prompt_ids == [1, 2, 3, 42]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [10, 11, 12, 13, 42]
    assert agent_data.extra_fields["teacher_ids_list"] == [[42, 102, 103]]
    assert agent_data.extra_fields["teacher_logprobs_list"] == [[-1.0, -2.0, -3.0]]
