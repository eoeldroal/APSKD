"""Worker primitives for bounded asynchronous SKD rollout."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    RolloutTraceConfig,
    _monkey_patch_log_timing,
    get_trajectory_info,
    monkey_patch_timing_begin,
)
from verl.experimental.async_skd.events import async_skd_event_context
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.protocol import DataProto

if TYPE_CHECKING:
    from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop


_ASYNC_SKD_INPUT_NON_TENSOR_BATCH = "async_skd_input_non_tensor_batch"


class AsyncSkdAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that owns async-SKD-specific execution primitives."""

    @staticmethod
    def _object_array(value: Any) -> np.ndarray:
        array = np.empty(1, dtype=object)
        array[0] = value
        return array

    def _build_sampling_params(self, *, validate: bool) -> dict[str, Any]:
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        return sampling_params

    def _ensure_agent_name(self, batch: DataProto) -> None:
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = self.rollout_config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop], dtype=object)

    def _single_kwargs(self, batch: DataProto) -> dict[str, Any]:
        return {key: value[0] for key, value in batch.non_tensor_batch.items()}

    def _single_input_non_tensor_batch(self, batch: DataProto) -> dict[str, np.ndarray]:
        return {key: value[:1].copy() for key, value in batch.non_tensor_batch.items()}

    def _input_non_tensor_from_partial(self, partial_state: SkdPartialState) -> dict[str, np.ndarray]:
        saved = partial_state.extra_fields.get(_ASYNC_SKD_INPUT_NON_TENSOR_BATCH)
        if saved is None:
            raw_prompt = partial_state.extra_fields.get("raw_prompt", partial_state.messages)
            return {
                "raw_prompt": self._object_array(raw_prompt),
                "agent_name": np.array(["skd_agent"], dtype=object),
            }
        return {key: np.asarray(value, dtype=object) for key, value in saved.items()}

    def _strip_internal_async_skd_extra_fields(self, output: AgentLoopOutput) -> None:
        output.extra_fields.pop(_ASYNC_SKD_INPUT_NON_TENSOR_BATCH, None)

    async def generate_sequence_single(
        self,
        batch: DataProto,
        *,
        async_skd_context: dict[str, Any] | None = None,
    ) -> DataProto:
        """Generate one sequence from agent loop without changing the batched API contract.

        This method is intentionally kept out of the base ``AgentLoopWorker``.
        Async SKD schedulers use it as a sample-level execution primitive while
        existing trainer paths keep calling the original batched method.
        """
        if len(batch) != 1:
            raise ValueError(f"generate_sequence_single expects exactly one sample, got batch size {len(batch)}.")

        batch_timer = time.perf_counter()
        validate = batch.meta_info.get("validate", False)
        sampling_params = self._build_sampling_params(validate=validate)

        # by default, we assume it's a single turn agent
        self._ensure_agent_name(batch)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(1)

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        trace_this_sample = max_samples_per_worker is None or max_samples_per_worker >= 1

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        run_timer = monkey_patch_timing_begin(capture_gpu=True)
        kwargs = {k: v[0] for k, v in batch.non_tensor_batch.items()}
        with async_skd_event_context(**(async_skd_context or {})):
            internal_output = await self._run_agent_loop(
                sampling_params, trajectory_info[0], trace=trace_this_sample, **kwargs
            )
        _monkey_patch_log_timing(
            "AsyncSkdAgentLoopWorker.generate_sequence_single.run",
            run_timer,
            validate=validate,
        )

        postprocess_timer = monkey_patch_timing_begin(capture_gpu=False)
        output = self._postprocess(
            [internal_output],
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=validate,
        )
        _monkey_patch_log_timing(
            "AsyncSkdAgentLoopWorker.generate_sequence_single.postprocess",
            postprocess_timer,
            validate=validate,
        )
        _monkey_patch_log_timing(
            "AsyncSkdAgentLoopWorker.generate_sequence_single.total",
            batch_timer,
            validate=validate,
        )
        return output

    async def generate_skd_until_boundary(
        self,
        batch: DataProto | None = None,
        *,
        partial_state: SkdPartialState | None = None,
        sample_id: str,
        logical_step: int,
        source_type: str,
        agent_name: str = "skd_agent",
        async_skd_context: dict[str, Any] | None = None,
    ) -> AsyncSkdSample:
        """Run an SKD sample until completion or the next exportable boundary."""
        if (batch is None) == (partial_state is None):
            raise ValueError("generate_skd_until_boundary expects exactly one of batch or partial_state")

        if batch is not None:
            if len(batch) != 1:
                raise ValueError(f"generate_skd_until_boundary expects single-sample batch, got {len(batch)}")
            self._ensure_agent_name(batch)
            kwargs = self._single_kwargs(batch)
            agent_name = kwargs.pop("agent_name")
            validate = batch.meta_info.get("validate", False)
            input_non_tensor_batch = self._single_input_non_tensor_batch(batch)
        else:
            assert partial_state is not None
            kwargs = {}
            validate = False
            input_non_tensor_batch = self._input_non_tensor_from_partial(partial_state)

        sampling_params = self._build_sampling_params(validate=validate)
        agent_loop = self._get_or_create_agent_loop(agent_name)
        from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop

        if not isinstance(agent_loop, SkdAgentLoop):
            raise TypeError(
                "generate_skd_until_boundary requires skd_agent loop, "
                f"got {type(agent_loop).__name__} for agent_name={agent_name!r}"
            )

        with async_skd_event_context(**(async_skd_context or {})):
            result = await agent_loop.run_until_exportable_boundary(
                sampling_params,
                sample_id=sample_id,
                logical_step=logical_step,
                source_type=source_type,
                partial_state=partial_state,
                **kwargs,
            )

        if isinstance(result, SkdPartialState):
            if batch is not None:
                result.extra_fields[_ASYNC_SKD_INPUT_NON_TENSOR_BATCH] = deepcopy(input_non_tensor_batch)
            return AsyncSkdSample.from_partial(partial_state=result)

        if not isinstance(result, AgentLoopOutput):
            raise TypeError(f"Unexpected SKD boundary result type: {type(result).__name__}")

        self._strip_internal_async_skd_extra_fields(result)
        postprocess_kwargs = self._single_kwargs(DataProto.from_dict(non_tensors=input_non_tensor_batch))
        internal_output = await self._agent_loop_postprocess(result, validate, **postprocess_kwargs)
        completed_batch = self._postprocess(
            [internal_output],
            input_non_tensor_batch=input_non_tensor_batch,
            validate=validate,
        )
        return AsyncSkdSample.from_completed(
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            batch=completed_batch,
        )

    async def generate_skd_from_partial_to_completion(
        self,
        partial_state: SkdPartialState,
        *,
        source_type: str = "resumed_current",
        agent_name: str = "skd_agent",
        async_skd_context: dict[str, Any] | None = None,
    ) -> AsyncSkdSample:
        """Resume an SKD partial as current-step work and run it to completion."""
        print(
            "[ASYNC_SKD] resume "
            f"sample_id={partial_state.sample_id} start_chunks={partial_state.committed_gen_chunks} "
            f"start_resp_len={len(partial_state.response_mask)} "
            f"start_prefix_tokens={partial_state.committed_prefix_tokens}",
            flush=True,
        )
        input_non_tensor_batch = self._input_non_tensor_from_partial(partial_state)
        sampling_params = self._build_sampling_params(validate=False)
        agent_loop = self._get_or_create_agent_loop(agent_name)
        from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop

        if not isinstance(agent_loop, SkdAgentLoop):
            raise TypeError(
                "generate_skd_from_partial_to_completion requires skd_agent loop, "
                f"got {type(agent_loop).__name__} for agent_name={agent_name!r}"
            )

        with async_skd_event_context(**(async_skd_context or {})):
            result = await agent_loop.run_from_partial_to_completion(
                sampling_params,
                partial_state=partial_state,
            )
        self._strip_internal_async_skd_extra_fields(result)
        postprocess_kwargs = self._single_kwargs(DataProto.from_dict(non_tensors=input_non_tensor_batch))
        internal_output = await self._agent_loop_postprocess(result, False, **postprocess_kwargs)
        completed_batch = self._postprocess(
            [internal_output],
            input_non_tensor_batch=input_non_tensor_batch,
            validate=False,
        )
        return AsyncSkdSample.from_completed(
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=source_type,
            batch=completed_batch,
        )
