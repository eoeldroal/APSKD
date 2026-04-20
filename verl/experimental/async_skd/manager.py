"""Agent-loop manager variants for bounded asynchronous SKD."""

from __future__ import annotations

import asyncio
import math
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
        extra_timing = getattr(self, "_async_skd_last_worker_slot_metrics", None)
        if extra_timing:
            timing.update(extra_timing)
            self._async_skd_last_worker_slot_metrics = None
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        extra_metrics = getattr(self, "_async_skd_last_step_metrics", None)
        if extra_metrics:
            output.meta_info["async_skd_metrics"] = extra_metrics
            self._async_skd_last_step_metrics = None
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

    def _lookahead_max_old_gen_chunks(self) -> int:
        value = OmegaConf.select(
            self.config,
            "actor_rollout_ref.rollout.agent.async_skd_max_old_gen_chunks",
            default=16,
        )
        return max(0, int(value))

    def _next_lookahead_sample(self, logical_step: int) -> tuple[str, DataProto] | None:
        source = self._get_async_skd_data_source()
        if source is None:
            return None
        return source.reserve_lookahead(logical_step)

    def _can_continue_lookahead_partial(self, partial_state: SkdPartialState) -> bool:
        return partial_state.committed_gen_chunks < self._lookahead_max_old_gen_chunks()

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
        fresh_count = len(fresh_prompts) if fresh_prompts is not None else 0
        current_items: list[tuple[str, int, Any]] = []

        for pos, partial in enumerate(carryover_partials):
            current_items.append(("carryover", pos, partial))

        if fresh_prompts is not None:
            offset = len(carryover_partials)
            for pos in range(fresh_count):
                current_items.append(("fresh", offset + pos, fresh_prompts[pos : pos + 1]))

        logical_step = 0
        if fresh_prompts is not None:
            logical_step = int(fresh_prompts.meta_info.get("global_steps", 0)) + 1
        elif carryover_partials:
            logical_step = max(partial.logical_step for partial in carryover_partials)
        prefetch_limit = self._lookahead_prefetch_limit(len(current_items))
        return await self._generate_current_work_with_lookahead(
            current_items,
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
        )

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
        current_items = [("fresh", pos, prompts[pos : pos + 1]) for pos in range(len(prompts))]
        logical_step = int(prompts.meta_info.get("global_steps", 0)) + 1
        prefetch_limit = self._lookahead_prefetch_limit(len(current_items))
        return await self._generate_current_work_with_lookahead(
            current_items,
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
        )

    async def _generate_current_work_with_lookahead(
        self,
        current_items: list[tuple[str, int, Any]],
        *,
        logical_step: int,
        prefetch_limit: int,
    ) -> list[DataProto]:
        """Run current work and use freed worker slots for bounded lookahead."""
        if not current_items:
            return []
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        current_count = len(current_items)
        current_completed: list[DataProto | None] = [None] * current_count
        promoted_lookahead: list[tuple[int, AsyncSkdSample]] = []
        carryover_partials: list[tuple[int, SkdPartialState]] = []
        current_active: dict[asyncio.Task, tuple[int, int]] = {}
        lookahead_active: dict[asyncio.Task, tuple[int, int]] = {}
        lookahead_started_count = 0
        drain_requested = False
        num_workers = len(self.agent_loop_workers)
        worker_capacity = max(1, math.ceil(current_count / num_workers))
        worker_active_counts = [0 for _ in range(num_workers)]
        worker_completed_counts = [0 for _ in range(num_workers)]
        worker_active_max = 0
        lookahead_continued_partial_count = 0

        def worker_idx_for_order(order: int) -> int:
            return min(order * num_workers // current_count, num_workers - 1)

        def worker_for_idx(worker_idx: int) -> Any:
            return self.agent_loop_workers[worker_idx]

        def note_launch(worker_idx: int) -> None:
            nonlocal worker_active_max
            worker_active_counts[worker_idx] += 1
            worker_active_max = max(worker_active_max, max(worker_active_counts))

        def note_finish(worker_idx: int) -> None:
            worker_active_counts[worker_idx] -= 1
            if worker_active_counts[worker_idx] < 0:
                raise RuntimeError(f"worker_active_counts[{worker_idx}] became negative")

        def launch_current(kind: str, order: int, payload: Any) -> None:
            worker_idx = worker_idx_for_order(order)
            worker = worker_for_idx(worker_idx)
            if kind == "fresh":
                task = asyncio.ensure_future(worker.generate_sequence_single.remote(payload))
            elif kind == "carryover":
                task = asyncio.ensure_future(worker.generate_skd_from_partial_to_completion.remote(payload))
            else:
                raise ValueError(f"Unsupported async SKD current work kind: {kind!r}")
            current_active[task] = (order, worker_idx)
            note_launch(worker_idx)

        def launch_lookahead_batch(sample_id: str, sample: DataProto, admission_order: int, worker_idx: int) -> None:
            worker = worker_for_idx(worker_idx)
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    sample,
                    sample_id=sample_id,
                    logical_step=logical_step,
                    source_type="lookahead",
                )
            )
            lookahead_active[task] = (admission_order, worker_idx)
            note_launch(worker_idx)
            print(
                "[ASYNC_SKD] prefetch_start "
                f"sample_id={sample_id} admission_order={admission_order} "
                f"worker={worker_idx} active_on_worker={worker_active_counts[worker_idx]} "
                f"worker_capacity={worker_capacity}",
                flush=True,
            )

        def launch_lookahead_partial(partial_state: SkdPartialState, admission_order: int, worker_idx: int) -> None:
            worker = worker_for_idx(worker_idx)
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    None,
                    partial_state=partial_state,
                    sample_id=partial_state.sample_id,
                    logical_step=partial_state.logical_step,
                    source_type=partial_state.source_type,
                )
            )
            lookahead_active[task] = (admission_order, worker_idx)
            note_launch(worker_idx)

        def try_admit_lookahead(worker_idx: int) -> None:
            nonlocal lookahead_started_count
            if drain_requested or not current_active or lookahead_started_count >= prefetch_limit:
                return
            if worker_active_counts[worker_idx] >= worker_capacity:
                return
            next_item = self._next_lookahead_sample(logical_step)
            if next_item is None:
                return
            sample_id, sample = next_item
            admission_order = lookahead_started_count
            lookahead_started_count += 1
            launch_lookahead_batch(sample_id, sample, admission_order, worker_idx)

        for kind, order, payload in current_items:
            launch_current(kind, order, payload)

        while current_active or lookahead_active:
            done, _ = await asyncio.wait(
                set(current_active.keys()) | set(lookahead_active.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                if task in current_active:
                    order, worker_idx = current_active.pop(task)
                    note_finish(worker_idx)
                    result = await task
                    if isinstance(result, AsyncSkdSample):
                        result.validate()
                        current_completed[order] = result.require_completed()
                    else:
                        current_completed[order] = result
                    worker_completed_counts[worker_idx] += 1
                    if not current_active and not drain_requested:
                        print(
                            "[ASYNC_SKD] drain_start "
                            f"completed_current={current_count} lookahead_active={len(lookahead_active)} "
                            f"started={lookahead_started_count} promoted={len(promoted_lookahead)} "
                            f"carryover_next={len(carryover_partials)}",
                            flush=True,
                        )
                        drain_requested = True
                    try_admit_lookahead(worker_idx)

            for task in done:
                if task in lookahead_active:
                    admission_order, worker_idx = lookahead_active.pop(task)
                    note_finish(worker_idx)
                    sample: AsyncSkdSample = await task
                    sample.validate()
                    if sample.kind == "completed":
                        promoted_lookahead.append((admission_order, sample))
                        worker_completed_counts[worker_idx] += 1
                        if not drain_requested:
                            try_admit_lookahead(worker_idx)
                        continue

                    partial = sample.require_partial()
                    can_continue = self._can_continue_lookahead_partial(partial)
                    if not drain_requested and bool(current_active) and can_continue:
                        lookahead_continued_partial_count += 1
                        launch_lookahead_partial(partial, admission_order, worker_idx)
                    else:
                        if drain_requested:
                            carryover_reason = "drain"
                        elif not current_active:
                            carryover_reason = "no_current"
                        elif not can_continue:
                            carryover_reason = "stale_cap"
                        else:
                            carryover_reason = "unknown"
                        print(
                            "[ASYNC_SKD] carryover "
                            f"sample_id={partial.sample_id} reason={carryover_reason} "
                            f"chunks={partial.committed_gen_chunks} "
                            f"resp_len={len(partial.response_mask)} prefix_tokens={partial.committed_prefix_tokens} "
                            f"worker={worker_idx}",
                            flush=True,
                        )
                        carryover_partials.append((admission_order, partial))

        self._async_skd_last_worker_slot_metrics = {
            "async_skd/worker_capacity": worker_capacity,
            "async_skd/worker_active_max": worker_active_max,
            "async_skd/lookahead_started_count": lookahead_started_count,
        }
        for idx, count in enumerate(worker_completed_counts):
            self._async_skd_last_worker_slot_metrics[f"async_skd/worker_{idx}_completed_count"] = count
        self._async_skd_last_step_metrics = {
            "async_skd/lookahead_prefetch_limit": prefetch_limit,
            "async_skd/lookahead_started_count": lookahead_started_count,
            "async_skd/lookahead_promoted_count": len(promoted_lookahead),
            "async_skd/lookahead_carryover_count": len(carryover_partials),
            "async_skd/lookahead_continued_partial_count": lookahead_continued_partial_count,
            "async_skd/worker_capacity": worker_capacity,
            "async_skd/worker_active_max": worker_active_max,
        }
        print(
            "[ASYNC_SKD] rollout "
            f"prefetch_limit={prefetch_limit} started={lookahead_started_count} "
            f"promoted={len(promoted_lookahead)} carryover_next={len(carryover_partials)} "
            f"continued_partial={lookahead_continued_partial_count} "
            f"worker_capacity={worker_capacity} worker_active_max={worker_active_max}",
            flush=True,
        )
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
        return [output for output in current_completed if output is not None] + promoted_outputs
