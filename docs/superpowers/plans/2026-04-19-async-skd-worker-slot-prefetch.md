# Async SKD Worker-Slot Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace round-robin lookahead admission with worker-slot refill scheduling, then add server-replica observability to decide whether preferred-server routing is needed later.

**Architecture:** The manager schedules one sample per Ray async worker call and tracks active work per `agent_loop_worker` rather than constructing tensor batches. Lookahead is admitted when a worker slot becomes free, subject to global prefetch limits, drain state, and stale-prefix constraints. Server replica ids are logged as observability metadata only; this plan does not add preferred-server routing.

**Tech Stack:** Python, asyncio, Ray async actors, verl `DataProto`, `AsyncSkdAgentLoopManager`, `AsyncSkdAgentLoopWorker`, SGLang server adapter, pytest-asyncio.

---

## Preconditions

- The safe-point label removal refactor is already applied.
- `SkdCommittedUnit`, `RESUMABLE_COMMITTED_UNITS`, `last_committed_unit`, and `skd_last_committed_unit` are not correctness gates.
- `SkdPartialState` still carries `committed_gen_chunks`, `committed_env_units`, and `committed_prefix_tokens`.
- `AsyncSkdAgentLoopManager._generate_sequences_lookahead()` still has sample-level task scheduling.
- `AsyncSkdAgentLoopWorker.generate_sequence_single()` and `generate_skd_until_boundary()` exist.

This plan does not implement:

- preferred SGLang server routing
- GPU-number based routing
- queue actors
- persistent rollout/trainer split
- new dataclasses for task payloads

## File Structure

- Modify `verl/experimental/async_skd/manager.py`.
  - Owns worker-slot accounting and lookahead admission policy.
  - Should not know CUDA ids.
  - Should track `worker_idx` and active task counts.

- Modify `verl/experimental/agent_loop/agent_loop.py`.
  - Adds server-replica observability by recording acquired `server_id` into returned output `extra_fields`.
  - Does not change load-balancer routing.

- Modify `tests/skd/test_async_skd_manager_lookahead.py`.
  - Adds worker-slot refill tests.
  - Existing fake workers already record `(worker_name, call_kind, input_pos_or_sample_id)`.

- Add or modify `tests/experimental/agent_loop/test_agent_loop_server_observability.py`.
  - Tests that `AsyncLLMServerManager.generate()` annotates `rollout_server_id`.

- Modify `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md`.
  - Keeps high-level design consistent after implementation.

- Modify `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md`.
  - Marks worker-slot refill as implemented and documents server observability.

## Task 1: Add Failing Tests For Worker-Slot Refill

**Files:**
- Modify: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Add a test where the fast worker receives refill lookahead**

Append this test near the existing lookahead manager tests:

```python
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
        # With 4 base samples and 2 workers, worker-0 gets base 0,1 and
        # worker-1 gets base 2,3. Delaying worker-1 makes worker-0 free slots first.
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
```

- [ ] **Step 2: Add a test that worker capacity is not exceeded**

Append:

```python
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
```

- [ ] **Step 3: Add a test that drain stops refill**

Append:

```python
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

    # Base batch size 2 with 2 workers gives capacity 1 per worker. Once both
    # base samples finish, drain_requested must prevent unbounded chaining.
    lookahead_calls = [call for call in calls if call[1] == "lookahead"]
    assert len(lookahead_calls) <= 2
    assert len(source.source_items) >= 4
```

- [ ] **Step 4: Run these tests and verify failure**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/skd/test_async_skd_manager_lookahead.py::test_lookahead_refills_the_worker_that_frees_a_slot_first \
  tests/skd/test_async_skd_manager_lookahead.py::test_lookahead_refill_does_not_exceed_worker_capacity \
  tests/skd/test_async_skd_manager_lookahead.py::test_lookahead_refill_stops_after_base_barrier_drain
```

Expected:

```text
FAIL
```

The first test should fail because current lookahead assignment is round-robin. The second should fail because `async_skd/worker_capacity` metrics do not exist. If all tests pass before implementation, inspect the test assumptions before continuing.

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/skd/test_async_skd_manager_lookahead.py
git commit -m "test: specify async skd worker slot refill"
```

## Task 2: Implement Worker-Slot Refill In Lookahead Mode

**Files:**
- Modify: `verl/experimental/async_skd/manager.py`
- Test: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Import `math`**

At the top of `manager.py`, change imports:

```python
import asyncio
import math
from typing import Any
```

- [ ] **Step 2: Change active task bookkeeping**

Inside `_generate_sequences_lookahead()`, replace:

```python
base_active: dict[asyncio.Task, int] = {}
lookahead_active: dict[asyncio.Task, int] = {}
```

with:

```python
base_active: dict[asyncio.Task, tuple[int, int]] = {}
lookahead_active: dict[asyncio.Task, tuple[int, int]] = {}
```

Then add after `logical_step`:

```python
num_workers = len(self.agent_loop_workers)
worker_capacity = max(1, math.ceil(len(prompts) / num_workers))
worker_active_counts = [0 for _ in range(num_workers)]
worker_active_max = 0
```

- [ ] **Step 3: Replace worker lookup helpers**

Replace `worker_for_pos()` and `worker_for_lookahead()` with:

```python
def worker_idx_for_pos(pos: int) -> int:
    return min(pos * num_workers // len(prompts), num_workers - 1)

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
```

- [ ] **Step 4: Launch base tasks with worker index**

Replace `launch_base()` with:

```python
def launch_base(pos: int) -> None:
    worker_idx = worker_idx_for_pos(pos)
    worker = worker_for_idx(worker_idx)
    sample = prompts[pos : pos + 1]
    task = asyncio.ensure_future(worker.generate_sequence_single.remote(sample))
    base_active[task] = (pos, worker_idx)
    note_launch(worker_idx)
```

- [ ] **Step 5: Launch lookahead on a chosen worker**

Replace `launch_lookahead_batch()` with:

```python
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
```

Replace `launch_lookahead_partial()` with:

```python
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
```

- [ ] **Step 6: Make admission worker-aware**

Replace `try_admit_lookahead()` with:

```python
def try_admit_lookahead(worker_idx: int) -> None:
    nonlocal lookahead_started_count
    if drain_requested or not base_active or lookahead_started_count >= prefetch_limit:
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
```

- [ ] **Step 7: Update completion handling**

In the `for task in done` block for base tasks, replace:

```python
pos = base_active.pop(task)
base_completed[pos] = await task
if not base_active:
    drain_requested = True
try_admit_lookahead()
```

with:

```python
pos, worker_idx = base_active.pop(task)
note_finish(worker_idx)
base_completed[pos] = await task
if not base_active:
    drain_requested = True
try_admit_lookahead(worker_idx)
```

In the `for task in done` block for lookahead tasks, replace:

```python
admission_order = lookahead_active.pop(task)
sample: AsyncSkdSample = await task
```

with:

```python
admission_order, worker_idx = lookahead_active.pop(task)
note_finish(worker_idx)
sample: AsyncSkdSample = await task
```

Replace partial continuation:

```python
launch_lookahead_partial(partial, admission_order)
```

with:

```python
launch_lookahead_partial(partial, admission_order, worker_idx)
```

After handling the lookahead result, before leaving the loop body, add:

```python
if sample.kind == "completed" and not drain_requested:
    try_admit_lookahead(worker_idx)
```

This keeps a worker filled after lookahead completions too, but still respects the global prefetch limit and drain state.

- [ ] **Step 8: Remove global round-robin refill call**

Remove:

```python
if not drain_requested:
    try_admit_lookahead()
```

at the end of the loop. Refill is now triggered by the worker that completed work.

- [ ] **Step 9: Add metrics to output timing**

Before returning outputs, add:

```python
self._async_skd_last_worker_slot_metrics = {
    "async_skd/worker_capacity": worker_capacity,
    "async_skd/worker_active_max": worker_active_max,
    "async_skd/lookahead_started_count": lookahead_started_count,
}
```

Then modify `_finalize_outputs()` so it merges manager-local async SKD metrics when present:

```python
extra_timing = getattr(self, "_async_skd_last_worker_slot_metrics", None)
if extra_timing:
    timing.update(extra_timing)
    self._async_skd_last_worker_slot_metrics = None
```

Place this after:

```python
timing = self._performance_metrics(metrics, output)
```

- [ ] **Step 10: Run focused tests**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_async_skd_manager_lookahead.py
```

Expected:

```text
all tests in tests/skd/test_async_skd_manager_lookahead.py pass
```

- [ ] **Step 11: Commit implementation**

```bash
git add verl/experimental/async_skd/manager.py tests/skd/test_async_skd_manager_lookahead.py
git commit -m "feat: add async skd worker slot refill"
```

## Task 3: Add Server-Replica Observability Without Routing Changes

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Create: `tests/experimental/agent_loop/test_agent_loop_server_observability.py`

- [ ] **Step 1: Add a failing unit test**

Create `tests/experimental/agent_loop/test_agent_loop_server_observability.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLoadBalancer:
    def __init__(self):
        self.acquire_server = _RemoteMethod(self._acquire_server)
        self.release_server = _RemoteMethod(self._release_server)
        self.released = []

    async def _acquire_server(self, request_id: str):
        assert request_id == "sticky-request"
        return "server-a"

    def _release_server(self, server_id: str):
        self.released.append(server_id)


class _FakeServer:
    def __init__(self):
        self.generate = _RemoteMethod(self._generate)

    async def _generate(self, **kwargs):
        assert kwargs["prompt_ids"] == [1, 2, 3]
        return SimpleNamespace(extra_fields={"existing": "value"})


@pytest.mark.asyncio
async def test_async_llm_server_manager_records_rollout_server_id():
    load_balancer = _FakeLoadBalancer()
    server = _FakeServer()
    manager = AsyncLLMServerManager(
        config={},
        servers=[("server-a", server)],
        load_balancer_handle=load_balancer,
    )

    output = await manager.generate(
        "sticky-request",
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 1},
    )

    assert output.extra_fields["existing"] == "value"
    assert output.extra_fields["rollout_server_id"] == "server-a"
    assert load_balancer.released == ["server-a"]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/experimental/agent_loop/test_agent_loop_server_observability.py::test_async_llm_server_manager_records_rollout_server_id
```

Expected:

```text
FAIL with KeyError: 'rollout_server_id'
```

- [ ] **Step 3: Implement metadata annotation**

In `AsyncLLMServerManager.generate()` in `verl/experimental/agent_loop/agent_loop.py`, after the server call returns and before `return output`, add:

```python
            if hasattr(output, "extra_fields") and output.extra_fields is not None:
                output.extra_fields["rollout_server_id"] = server_id
```

Do not change load-balancer acquire/release semantics.

- [ ] **Step 4: Run the observability test**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/experimental/agent_loop/test_agent_loop_server_observability.py
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Run related SKD manager tests**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_async_skd_manager_lookahead.py tests/skd/test_async_skd_worker.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 6: Commit**

```bash
git add verl/experimental/agent_loop/agent_loop.py tests/experimental/agent_loop/test_agent_loop_server_observability.py
git commit -m "feat: record rollout server id in agent loop outputs"
```

## Task 4: Add Worker/Server Distribution Metrics To Async SKD Manager

**Files:**
- Modify: `verl/experimental/async_skd/manager.py`
- Modify: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Add fake worker server ids**

In `_FakeLookaheadWorker._generate_sequence_single()`, modify returned output:

```python
output = _make_output(input_pos)
output.meta_info["metrics"][0]["rollout_server_id"] = self._name
return output
```

In `_FakeLookaheadWorker._generate_skd_until_boundary()`, when returning a completed sample, set server id on the batch:

```python
sample = self._lookahead_results[sample_id].pop(0)
if sample.kind == "completed":
    sample.require_completed().meta_info["metrics"][0]["rollout_server_id"] = self._name
return sample
```

- [ ] **Step 2: Add a failing metric test**

Append:

```python
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
```

- [ ] **Step 3: Run the failing metric test**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/skd/test_async_skd_manager_lookahead.py::test_lookahead_reports_worker_slot_and_server_distribution_metrics
```

Expected:

```text
FAIL
```

- [ ] **Step 4: Implement worker completion counters**

In `_generate_sequences_lookahead()`, initialize:

```python
worker_completed_counts = [0 for _ in range(num_workers)]
```

When a base task completes, after `base_completed[pos] = await task`, add:

```python
worker_completed_counts[worker_idx] += 1
```

When a lookahead task completes and `sample.kind == "completed"`, add:

```python
worker_completed_counts[worker_idx] += 1
```

When setting `_async_skd_last_worker_slot_metrics`, include:

```python
for idx, count in enumerate(worker_completed_counts):
    self._async_skd_last_worker_slot_metrics[f"async_skd/worker_{idx}_completed_count"] = count
```

- [ ] **Step 5: Run metric tests**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_async_skd_manager_lookahead.py
```

Expected:

```text
all tests in tests/skd/test_async_skd_manager_lookahead.py pass
```

- [ ] **Step 6: Commit**

```bash
git add verl/experimental/async_skd/manager.py tests/skd/test_async_skd_manager_lookahead.py
git commit -m "feat: add async skd worker slot metrics"
```

## Task 5: Update Documentation

**Files:**
- Modify: `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md`
- Modify: `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md`

- [ ] **Step 1: Verify current docs references**

Run:

```bash
rg -n "Worker-Slot|Server-Replica|Preferred Server|round-robin|worker_active|rollout_server_id|worker_capacity" \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/*.md
```

Expected:

```text
Matches in design.md and implementation_plan.md for worker-slot refill and server-replica observability.
```

- [ ] **Step 2: Update design wording**

In `design.md`, under `Worker-Slot Refill Policy`, ensure this final text exists:

```text
The first implementation is worker-slot aware. It does not require exact CUDA device ids and does not force preferred SGLang server routing. Server replica ids are observed through output metadata so worker-level scheduling can be compared against actual SGLang server distribution.
```

- [ ] **Step 3: Update implementation plan patch status**

In `implementation_plan.md`, ensure `Patch 12: Worker-Slot Refill Lookahead` states:

```text
This patch is implemented before preferred-server routing. Preferred-server routing is allowed only after server-replica metrics show worker-level refill is insufficient.
```

- [ ] **Step 4: Run markdown checks**

Run:

```bash
git diff --check -- \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md
```

Expected:

```text
No output.
```

- [ ] **Step 5: Commit**

```bash
git add \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md
git commit -m "docs: describe async skd worker slot refill implementation"
```

## Task 6: Final Focused Verification

**Files:**
- No source changes.

- [ ] **Step 1: Run focused async SKD tests**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/skd/test_async_skd_manager.py \
  tests/skd/test_async_skd_manager_lookahead.py \
  tests/skd/test_async_skd_worker.py \
  tests/skd/test_async_skd_worker_boundary.py \
  tests/experimental/agent_loop/test_agent_loop_server_observability.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run broader SKD smoke tests**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q \
  tests/skd/test_skd_logic.py \
  tests/skd/test_async_skd_data_source.py \
  tests/skd/test_async_skd_state.py \
  tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Search for accidental preferred-server routing**

Run:

```bash
rg -n "preferred_server|preferred_server_id|acquire_specific|force_server" \
  verl/experimental/agent_loop \
  verl/experimental/async_skd \
  tests/skd \
  tests/experimental/agent_loop
```

Expected:

```text
No matches, unless preferred-server routing has been intentionally moved into a separate later branch.
```

- [ ] **Step 4: Commit verification note if files changed**

If no files changed, do not commit. If documentation or tests were adjusted during verification, commit only those files:

```bash
git status --short
git add <changed-files>
git commit -m "test: verify async skd worker slot refill"
```

## Self-Review

Spec coverage:

- Worker-slot refill scheduling is covered by Tasks 1 and 2.
- Server-replica observability without routing changes is covered by Task 3.
- Worker/server distribution metrics are covered by Task 4.
- Documentation update is covered by Task 5.
- Verification is covered by Task 6.

Placeholder scan:

- This plan contains concrete task steps, commands, expected results, and code snippets.
- Preferred-server routing is explicitly out of scope and assigned to a later patch only if metrics justify it.

Type consistency:

- Active task bookkeeping uses `tuple[int, int]`, not a new dataclass.
- `worker_idx` is the scheduling key.
- `rollout_server_id` is observability metadata.
- `worker_capacity`, `worker_active_counts`, and `worker_active_max` are manager-local scheduler fields.
