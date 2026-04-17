"""Worker primitives for bounded asynchronous SKD rollout."""

from __future__ import annotations

import time

import numpy as np

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker,
    RolloutTraceConfig,
    _monkey_patch_log_timing,
    get_trajectory_info,
    monkey_patch_timing_begin,
)
from verl.protocol import DataProto


class AsyncSkdAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that owns async-SKD-specific execution primitives."""

    async def generate_sequence_single(self, batch: DataProto) -> DataProto:
        """Generate one sequence from agent loop without changing the batched API contract.

        This method is intentionally kept out of the base ``AgentLoopWorker``.
        Async SKD schedulers use it as a sample-level execution primitive while
        existing trainer paths keep calling the original batched method.
        """
        if len(batch) != 1:
            raise ValueError(f"generate_sequence_single expects exactly one sample, got batch size {len(batch)}.")

        batch_timer = time.perf_counter()
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop], dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(1)

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        trace_this_sample = max_samples_per_worker is None or max_samples_per_worker >= 1

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        run_timer = monkey_patch_timing_begin(capture_gpu=True)
        kwargs = {k: v[0] for k, v in batch.non_tensor_batch.items()}
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
