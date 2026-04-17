"""Agent-loop manager variants for bounded asynchronous SKD."""

from __future__ import annotations

import asyncio
from typing import Any

from omegaconf import OmegaConf
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await


class AsyncSkdAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with an optional sample-level async execution path.

    The public contract stays identical to ``AgentLoopManager.generate_sequences``:
    callers pass one ``DataProto`` batch and receive one ``DataProto`` batch.
    ``sample_async`` only changes the internal scheduling granularity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(AsyncSkdAgentLoopWorker)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        mode = self._async_skd_mode()
        if mode in {"sync", "disabled", "none"}:
            return await super().generate_sequences(prompts)
        if mode != "sample_async":
            raise ValueError(f"Unsupported async SKD rollout mode: {mode!r}")

        rollout_n = self._rollout_n()
        if rollout_n != 1:
            raise ValueError(f"Async SKD sample_async currently requires rollout.n == 1, got {rollout_n}")

        if self.stream_teacher_with_rollout:
            await self.teacher_model_manager.wake_up()
        try:
            outputs = await self._generate_sequences_sample_async(prompts)
        finally:
            if self.stream_teacher_with_rollout:
                await self.teacher_model_manager.sleep()

        output = DataProto.concat(outputs)
        metrics = [single.meta_info.pop("metrics") for single in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _async_skd_mode(self) -> str:
        mode = OmegaConf.select(self.config, "actor_rollout_ref.rollout.agent.async_skd_mode", default=None)
        if mode is None:
            mode = OmegaConf.select(self.config, "distillation.async_skd.mode", default="sync")
        return str(mode)

    def _rollout_n(self) -> int:
        value: Any = OmegaConf.select(self.config, "actor_rollout_ref.rollout.n", default=None)
        if value is None:
            value = getattr(self.rollout_config, "n", 1)
        return int(value)

    async def _generate_sequences_sample_async(self, prompts: DataProto) -> list[DataProto]:
        """Run all base samples concurrently and collect by FIRST_COMPLETED.

        Returned outputs are ordered by the original input position, not by
        completion order.  This preserves downstream assumptions about uid,
        index, reward metadata, and repeated rollout layout.
        """
        if len(prompts) == 0:
            return []
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        outputs: list[DataProto | None] = [None] * len(prompts)
        active: dict[asyncio.Task, tuple[int, Any]] = {}

        def worker_for_pos(pos: int) -> Any:
            worker_idx = min(pos * len(self.agent_loop_workers) // len(prompts), len(self.agent_loop_workers) - 1)
            return self.agent_loop_workers[worker_idx]

        def launch(pos: int) -> None:
            worker = worker_for_pos(pos)
            sample = prompts[pos : pos + 1]
            task = asyncio.ensure_future(worker.generate_sequence_single.remote(sample))
            active[task] = (pos, worker)

        # Submit the whole base batch immediately. Ray async actors can execute
        # many async methods concurrently, so this preserves the same request
        # concurrency as AgentLoopWorker.generate_sequences(...), while exposing
        # per-sample completion events to the manager.
        for pos in range(len(prompts)):
            launch(pos)

        while active:
            done, _ = await asyncio.wait(active.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pos, _ = active.pop(task)
                outputs[pos] = await task

        return [output for output in outputs if output is not None]
