"""Agent-loop manager variants for bounded asynchronous SKD."""

from __future__ import annotations

import asyncio
from typing import Any

from omegaconf import OmegaConf
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
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
        self._async_skd_data_source = None

    def set_async_skd_data_source(self, source: Any | None) -> None:
        self._async_skd_data_source = source

    def _get_async_skd_data_source(self) -> Any | None:
        return getattr(self, "_async_skd_data_source", None)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        mode = self._async_skd_mode()
        if mode in {"sync", "disabled", "none"}:
            return await super().generate_sequences(prompts)
        if mode not in {"sample_async", "lookahead"}:
            raise ValueError(f"Unsupported async SKD rollout mode: {mode!r}")

        rollout_n = self._rollout_n()
        if rollout_n != 1:
            raise ValueError(f"Async SKD {mode} currently requires rollout.n == 1, got {rollout_n}")

        if self.stream_teacher_with_rollout:
            await self.teacher_model_manager.wake_up()
        try:
            if mode == "lookahead":
                outputs = await self._generate_sequences_lookahead(prompts)
            else:
                outputs = await self._generate_sequences_sample_async(prompts)
        finally:
            if self.stream_teacher_with_rollout:
                await self.teacher_model_manager.sleep()

        return self._finalize_outputs(outputs)

    def _finalize_outputs(self, outputs: list[DataProto]) -> DataProto:
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

    def _lookahead_prefetch_limit(self, batch_size: int) -> int:
        value = OmegaConf.select(
            self.config,
            "actor_rollout_ref.rollout.agent.async_skd_prefetch_limit",
            default=0,
        )
        return max(0, min(int(value), batch_size))

    def _next_lookahead_sample(self, logical_step: int) -> tuple[str, DataProto] | None:
        source = self._get_async_skd_data_source()
        if source is None:
            return None
        return source.reserve_lookahead(logical_step)

    def _can_continue_lookahead_partial(self, partial_state: SkdPartialState) -> bool:
        del partial_state
        return True

    def _next_fresh_quota(self, base_batch_size: int) -> int:
        source = self._get_async_skd_data_source()
        if source is not None:
            return int(source.next_fresh_quota(base_batch_size))
        carryover_count = len(getattr(self, "_async_skd_carryover_partials", []))
        return max(0, base_batch_size - carryover_count)

    @auto_await
    async def generate_sequences_with_carryover(
        self,
        *,
        fresh_prompts: DataProto | None,
        carryover_partials: list[SkdPartialState],
    ) -> DataProto:
        rollout_n = self._rollout_n()
        if rollout_n != 1:
            raise ValueError(f"Async SKD carryover currently requires rollout.n == 1, got {rollout_n}")

        if fresh_prompts is None and not carryover_partials:
            raise ValueError("generate_sequences_with_carryover requires fresh_prompts or carryover_partials")

        if fresh_prompts is not None and len(fresh_prompts) == 0:
            fresh_prompts = None

        if self.stream_teacher_with_rollout:
            await self.teacher_model_manager.wake_up()
        try:
            outputs = await self._generate_sequences_with_carryover(fresh_prompts, carryover_partials)
        finally:
            if self.stream_teacher_with_rollout:
                await self.teacher_model_manager.sleep()

        return self._finalize_outputs(outputs)

    async def _generate_sequences_with_carryover(
        self,
        fresh_prompts: DataProto | None,
        carryover_partials: list[SkdPartialState],
    ) -> list[DataProto]:
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        fresh_count = len(fresh_prompts) if fresh_prompts is not None else 0
        output_count = len(carryover_partials) + fresh_count
        outputs: list[DataProto | None] = [None] * output_count
        active: dict[asyncio.Task, int] = {}

        def worker_for_order(order: int) -> Any:
            return self.agent_loop_workers[order % len(self.agent_loop_workers)]

        for pos, partial in enumerate(carryover_partials):
            worker = worker_for_order(pos)
            task = asyncio.ensure_future(worker.generate_skd_from_partial_to_completion.remote(partial))
            active[task] = pos

        if fresh_prompts is not None:
            offset = len(carryover_partials)
            for pos in range(fresh_count):
                worker = worker_for_order(offset + pos)
                sample = fresh_prompts[pos : pos + 1]
                task = asyncio.ensure_future(worker.generate_sequence_single.remote(sample))
                active[task] = offset + pos

        while active:
            done, _ = await asyncio.wait(active.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                order = active.pop(task)
                result = await task
                if isinstance(result, AsyncSkdSample):
                    result.validate()
                    outputs[order] = result.require_completed()
                else:
                    outputs[order] = result

        return [output for output in outputs if output is not None]

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

    async def _generate_sequences_lookahead(self, prompts: DataProto) -> list[DataProto]:
        """Run base samples while opportunistically filling freed slots with bounded lookahead."""
        if len(prompts) == 0:
            return []
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        base_completed: list[DataProto | None] = [None] * len(prompts)
        promoted_lookahead: list[tuple[int, AsyncSkdSample]] = []
        carryover_partials: list[tuple[int, SkdPartialState]] = []
        base_active: dict[asyncio.Task, int] = {}
        lookahead_active: dict[asyncio.Task, int] = {}
        prefetch_limit = self._lookahead_prefetch_limit(len(prompts))
        lookahead_started_count = 0
        lookahead_launch_count = 0
        drain_requested = False
        logical_step = int(prompts.meta_info.get("global_steps", 0)) + 1

        def worker_for_pos(pos: int) -> Any:
            worker_idx = min(pos * len(self.agent_loop_workers) // len(prompts), len(self.agent_loop_workers) - 1)
            return self.agent_loop_workers[worker_idx]

        def worker_for_lookahead() -> Any:
            nonlocal lookahead_launch_count
            worker = self.agent_loop_workers[lookahead_launch_count % len(self.agent_loop_workers)]
            lookahead_launch_count += 1
            return worker

        def launch_base(pos: int) -> None:
            worker = worker_for_pos(pos)
            sample = prompts[pos : pos + 1]
            task = asyncio.ensure_future(worker.generate_sequence_single.remote(sample))
            base_active[task] = pos

        def launch_lookahead_batch(sample_id: str, sample: DataProto, admission_order: int) -> None:
            worker = worker_for_lookahead()
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    sample,
                    sample_id=sample_id,
                    logical_step=logical_step,
                    source_type="lookahead",
                )
            )
            lookahead_active[task] = admission_order

        def launch_lookahead_partial(partial_state: SkdPartialState, admission_order: int) -> None:
            worker = worker_for_lookahead()
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    None,
                    partial_state=partial_state,
                    sample_id=partial_state.sample_id,
                    logical_step=partial_state.logical_step,
                    source_type=partial_state.source_type,
                )
            )
            lookahead_active[task] = admission_order

        def try_admit_lookahead() -> None:
            nonlocal lookahead_started_count
            if drain_requested or not base_active or lookahead_started_count >= prefetch_limit:
                return
            next_item = self._next_lookahead_sample(logical_step)
            if next_item is None:
                return
            sample_id, sample = next_item
            admission_order = lookahead_started_count
            lookahead_started_count += 1
            launch_lookahead_batch(sample_id, sample, admission_order)

        for pos in range(len(prompts)):
            launch_base(pos)

        while base_active or lookahead_active:
            done, _ = await asyncio.wait(
                set(base_active.keys()) | set(lookahead_active.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                if task in base_active:
                    pos = base_active.pop(task)
                    base_completed[pos] = await task
                    if not base_active:
                        drain_requested = True
                    try_admit_lookahead()

            for task in done:
                if task in lookahead_active:
                    admission_order = lookahead_active.pop(task)
                    sample: AsyncSkdSample = await task
                    sample.validate()
                    if sample.kind == "completed":
                        promoted_lookahead.append((admission_order, sample))
                        continue

                    partial = sample.require_partial()
                    if (
                        not drain_requested
                        and bool(base_active)
                        and self._can_continue_lookahead_partial(partial)
                    ):
                        launch_lookahead_partial(partial, admission_order)
                    else:
                        carryover_partials.append((admission_order, partial))

            if not drain_requested:
                try_admit_lookahead()

        self._async_skd_last_promoted_samples = [
            sample for _, sample in sorted(promoted_lookahead, key=lambda item: item[0])
        ]
        self._async_skd_carryover_partials = [
            partial for _, partial in sorted(carryover_partials, key=lambda item: item[0])
        ]
        source = self._get_async_skd_data_source()
        if source is not None:
            source.record_promoted(self._async_skd_last_promoted_samples)
            source.record_carryover(self._async_skd_carryover_partials)

        promoted_outputs = [sample.require_completed() for sample in self._async_skd_last_promoted_samples]
        return [output for output in base_completed if output is not None] + promoted_outputs
