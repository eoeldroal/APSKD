# Async SKD Export Predicate Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove artificial async SKD safe-point labels and make partial export depend on handler-return boundaries, teacher alignment, and Qwen/Hermes tool-boundary state.

**Architecture:** Keep the existing `SkdAgentLoop` outer state machine and handler structure. Do not add new scheduler classes or a large set of helper functions. Remove `SkdCommittedUnit`, `RESUMABLE_COMMITTED_UNITS`, `last_committed_unit`, and `skd_last_committed_unit`; keep the existing stale-prefix counters because they are accounting data, not safe-point labels.

**Tech Stack:** Python, pytest, pytest-asyncio, verl `DataProto`, `SkdAgentLoop`, `AsyncSkdSample`, `SkdPartialState`.

---

## File Structure

Modify:

```text
verl/experimental/async_skd/state.py
verl/experimental/async_skd/__init__.py
verl/experimental/agent_loop/skd_agent_loop.py
tests/skd/test_async_skd_state.py
tests/skd/test_async_skd_worker_boundary.py
tests/skd/test_async_skd_data_source.py
tests/skd/test_async_skd_manager_lookahead.py
tests/skd/test_skd_logic.py
examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md
examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md
```

Do not modify:

```text
verl/experimental/async_skd/manager.py
verl/experimental/async_skd/data_source.py
verl/experimental/async_skd/worker.py
verl/trainer/ppo/ray_trainer.py
```

Those files consume `SkdPartialState` and `AsyncSkdSample`, but they should not need logic changes after the label fields are removed.

## Task 1: Remove Safe-Point Labels From State Envelope

**Files:**
- Modify: `verl/experimental/async_skd/state.py`
- Modify: `verl/experimental/async_skd/__init__.py`
- Test: `tests/skd/test_async_skd_state.py`

- [ ] **Step 1: Write the failing state-envelope test update**

Edit `tests/skd/test_async_skd_state.py`.

Remove this import:

```python
from verl.experimental.async_skd.state import SkdCommittedUnit, SkdPartialState
```

Replace it with:

```python
from verl.experimental.async_skd.state import SkdPartialState
```

Replace `make_partial()` with:

```python
def make_partial() -> SkdPartialState:
    return SkdPartialState(
        sample_id="sample-partial",
        logical_step=5,
        source_type="lookahead_carryover",
        agent_state="generating",
        request_id="req-partial",
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=8,
        committed_gen_chunks=2,
        committed_env_units=1,
        committed_prefix_tokens=128,
        metrics={"generate_sequences": 1.0},
        extra_fields={
            "teacher_ids_list": [[1, 2]],
            "teacher_logprobs_list": [[-1.0, -2.0]],
        },
    )
```

Replace `test_async_skd_sample_rejects_unknown_source_type_and_non_resumable_unit()` with:

```python
def test_async_skd_sample_rejects_unknown_source_type():
    with pytest.raises(ValueError, match="source_type"):
        AsyncSkdSample.from_completed(
            sample_id="bad-source",
            logical_step=0,
            source_type="unknown",
            batch=make_single_batch(),
        )
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_async_skd_state.py
```

Expected:

```text
FAIL with TypeError mentioning missing required positional argument 'last_committed_unit'
```

- [ ] **Step 3: Remove label fields and validation from `state.py`**

In `verl/experimental/async_skd/state.py`, remove:

```python
from enum import Enum
```

Remove the entire `SkdCommittedUnit` class and `RESUMABLE_COMMITTED_UNITS` definition.

In `SkdPartialState`, delete this field:

```python
last_committed_unit: str
```

In `AsyncSkdSample.validate()`, delete this block:

```python
        if self.partial_state.last_committed_unit not in RESUMABLE_COMMITTED_UNITS:
            raise ValueError(
                "Partial AsyncSkdSample last_committed_unit is not resumable: "
                f"{self.partial_state.last_committed_unit!r}"
            )
```

Do not remove these fields:

```python
committed_gen_chunks: int = 0
committed_env_units: int = 0
committed_prefix_tokens: int = 0
```

- [ ] **Step 4: Remove `SkdCommittedUnit` export**

In `verl/experimental/async_skd/__init__.py`, remove:

```python
SkdCommittedUnit
```

from both the import and `__all__`.

- [ ] **Step 5: Verify the state test passes**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_async_skd_state.py
```

Expected:

```text
5 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add verl/experimental/async_skd/state.py verl/experimental/async_skd/__init__.py tests/skd/test_async_skd_state.py
git commit -m "refactor: remove async skd safe point state labels"
```

## Task 2: Replace SkdAgentLoop Label Gate With Real Export Predicate

**Files:**
- Modify: `verl/experimental/agent_loop/skd_agent_loop.py`
- Test: `tests/skd/test_skd_logic.py`

- [ ] **Step 1: Add Qwen/Hermes boundary tests**

In `tests/skd/test_skd_logic.py`, add this tokenizer below `FakeTokenizer`:

```python
OPEN_TOOL = 91001
CLOSE_TOOL = 91002


class FakeHermesTokenizer:
    eos_token_id = EOS

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        text_parts = []
        for token_id in ids:
            if token_id == OPEN_TOOL:
                text_parts.append("<tool_call>")
            elif token_id == CLOSE_TOOL:
                text_parts.append("</tool_call>")
            elif token_id == EOS:
                text_parts.append("<|im_end|>")
            else:
                text_parts.append(str(token_id))
        return "".join(text_parts)
```

Add these tests after `test_skd_generation_can_pause_at_committed_chunk_boundary_and_resume`:

```python
@pytest.mark.asyncio
async def test_skd_export_allows_open_tool_call_prefix_without_eos():
    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert_skd_alignment(agent_data)


@pytest.mark.asyncio
async def test_skd_export_rejects_closed_tool_call_without_eos():
    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11, CLOSE_TOOL],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert not loop._can_export_partial_state(agent_data, next_state)
    assert_skd_alignment(agent_data)
```

Add this tool-close test near `test_skd_boundary_driver_closes_tool_macro_step_before_export`:

```python
@pytest.mark.asyncio
async def test_skd_boundary_driver_closes_eos_tool_call_before_export(monkeypatch):
    async def fake_tool_step(self, agent_data):
        del self
        tool_tokens = [900, 901]
        agent_data.messages.append({"role": "tool", "content": "tool result"})
        agent_data.prompt_ids += tool_tokens
        agent_data.response_mask += [0] * len(tool_tokens)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    class _Parser:
        async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
            del response_ids, tools
            return None, [FakeToolCall(name="lookup", arguments='{"query":"weather"}')]

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step)

    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11, CLOSE_TOOL, EOS],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    loop.tool_parser = _Parser()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await loop._run_until_exportable_boundary(agent_data, AgentState.GENERATING, {})

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [1, 2, 3, OPEN_TOOL, 11, CLOSE_TOOL, EOS, 900, 901]
    assert agent_data.response_mask == [1, 1, 1, 1, 0, 0]
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 1
    assert agent_data.extra_fields["skd_committed_env_units"] == 1
    assert_skd_alignment(agent_data)
    assert_masked_teacher_rows(agent_data)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_skd_logic.py -k "open_tool_call_prefix or closed_tool_call_without_eos or eos_tool_call_before_export"
```

Expected:

```text
FAIL with TypeError mentioning unexpected keyword argument 'stop_after_skd_chunk'
```

- [ ] **Step 3: Rename the stop flag**

In `verl/experimental/agent_loop/skd_agent_loop.py`, change `_handle_generating_state()` parameter:

```python
stop_after_committed_unit: bool = False
```

to:

```python
stop_after_skd_chunk: bool = False
```

Change the fallback error text from:

```python
"stop_after_committed_unit requires SKD teacher verification"
```

to:

```python
"stop_after_skd_chunk requires SKD teacher verification"
```

Change all internal references from `stop_after_committed_unit` to `stop_after_skd_chunk`.

In `_run_until_exportable_boundary()`, change:

```python
stop_after_committed_unit=True,
```

to:

```python
stop_after_skd_chunk=True,
```

- [ ] **Step 4: Remove label writes and label imports from `skd_agent_loop.py`**

In `verl/experimental/agent_loop/skd_agent_loop.py`, replace:

```python
from verl.experimental.async_skd.state import RESUMABLE_COMMITTED_UNITS, SkdPartialState
```

with:

```python
from verl.experimental.async_skd.state import SkdPartialState
```

Delete:

```python
    def _set_skd_last_committed_unit(self, agent_data: AgentData, unit: str) -> None:
        """Record the last fully committed SKD atomic unit."""
        agent_data.extra_fields["skd_last_committed_unit"] = unit

    def _get_skd_last_committed_unit(self, agent_data: AgentData) -> str | None:
        return agent_data.extra_fields.get("skd_last_committed_unit")
```

Delete every call whose function name is `_set_skd_last_committed_unit`.

- [ ] **Step 5: Add the single export predicate helper**

In `verl/experimental/agent_loop/skd_agent_loop.py`, add this method near `_can_export_partial_state()`:

```python
    def _is_qwen_hermes_exportable_assistant_prefix(self, agent_data: AgentData) -> bool:
        """Return whether current assistant prefix may be exported before the next chunk."""
        if not agent_data.response_ids:
            return True

        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        has_closed_tool_block = "</tool_call>" in text
        if not has_closed_tool_block:
            return True

        eos_token_id = self.tokenizer.eos_token_id
        eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        has_eos = any(token_id in eos_ids for token_id in agent_data.response_ids)
        return has_eos
```

No separate `_contains_eos()`, `_teacher_alignment_ok()`, or new atomic-unit wrapper is added in this task.

- [ ] **Step 6: Replace `_can_export_partial_state()`**

In `verl/experimental/agent_loop/skd_agent_loop.py`, replace `_can_export_partial_state()` with:

```python
    def _can_export_partial_state(self, agent_data: AgentData, next_state: AgentState) -> bool:
        """Return whether the current trajectory can be snapshotted for resume."""
        if next_state != AgentState.GENERATING:
            return False

        teacher_ids_list = agent_data.extra_fields.get("teacher_ids_list")
        teacher_logprobs_list = agent_data.extra_fields.get("teacher_logprobs_list")
        if teacher_ids_list is None or teacher_logprobs_list is None:
            return False
        if len(agent_data.response_mask) != len(teacher_ids_list):
            return False
        if len(agent_data.response_mask) != len(teacher_logprobs_list):
            return False
        return self._is_qwen_hermes_exportable_assistant_prefix(agent_data)
```

In `_export_partial_state()`, remove `last_committed_unit` from the error message and from the `SkdPartialState` constructor call.

In `_restore_partial_state()`, remove the `partial_state.last_committed_unit` validation and do not write `agent_data.extra_fields["skd_last_committed_unit"]`.

- [ ] **Step 7: Verify focused SKD logic tests pass**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_skd_logic.py -k "open_tool_call_prefix or closed_tool_call_without_eos or eos_tool_call_before_export"
```

Expected:

```text
3 passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add verl/experimental/agent_loop/skd_agent_loop.py tests/skd/test_skd_logic.py
git commit -m "refactor: use export predicates for skd partial snapshots"
```

## Task 3: Update Existing SKD Tests And Fixtures

**Files:**
- Modify: `tests/skd/test_skd_logic.py`
- Modify: `tests/skd/test_async_skd_worker_boundary.py`
- Modify: `tests/skd/test_async_skd_data_source.py`
- Modify: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Remove label imports and fixture fields**

Remove `SkdCommittedUnit` imports from these files:

```text
tests/skd/test_async_skd_worker_boundary.py
tests/skd/test_async_skd_data_source.py
tests/skd/test_async_skd_manager_lookahead.py
```

In each `SkdPartialState` test fixture, remove any argument with this key:

```python
last_committed_unit
```

In fixture `extra_fields`, remove:

```python
"skd_last_committed_unit"
```

- [ ] **Step 2: Remove label assertions from `tests/skd/test_skd_logic.py`**

Delete assertions matching:

```python
assert agent_data.extra_fields["skd_last_committed_unit"] == "ASSISTANT_GEN_CHUNK"
assert partial.last_committed_unit == "ASSISTANT_GEN_CHUNK"
assert restored_agent_data.extra_fields["skd_last_committed_unit"] == "ASSISTANT_GEN_CHUNK"
assert "skd_last_committed_unit" not in agent_data.extra_fields
```

Do not delete assertions for:

```python
assert_skd_alignment(agent_data)
assert_masked_teacher_rows(agent_data)
assert agent_data.extra_fields["skd_committed_gen_chunks"] == 1
assert agent_data.extra_fields["skd_committed_env_units"] == 1
assert agent_data.extra_fields["skd_committed_prefix_tokens"] == 3
```

- [ ] **Step 3: Rename stop flag in tests**

Replace all test calls:

```python
stop_after_committed_unit=True
```

with:

```python
stop_after_skd_chunk=True
```

- [ ] **Step 4: Run affected tests and verify pass**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_skd_logic.py tests/skd/test_async_skd_worker_boundary.py tests/skd/test_async_skd_data_source.py tests/skd/test_async_skd_manager_lookahead.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/skd/test_skd_logic.py tests/skd/test_async_skd_worker_boundary.py tests/skd/test_async_skd_data_source.py tests/skd/test_async_skd_manager_lookahead.py
git commit -m "test: update async skd tests for export predicates"
```

## Task 4: Remove Residual Safe-Point References From Runtime And Docs

**Files:**
- Modify: `verl/experimental/async_skd/state.py`
- Modify: `verl/experimental/async_skd/__init__.py`
- Modify: `verl/experimental/agent_loop/skd_agent_loop.py`
- Modify: `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md`
- Modify: `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md`

- [ ] **Step 1: Run residual symbol search**

Run:

```bash
rg -n "SkdCommittedUnit|RESUMABLE_COMMITTED_UNITS|last_committed_unit|skd_last_committed_unit|stop_after_committed_unit|ASSISTANT_GEN_CHUNK|ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT|ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT" verl/experimental/async_skd verl/experimental/agent_loop/skd_agent_loop.py tests/skd examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd -g '*.py' -g '*.md'
```

Expected after Tasks 1-3:

```text
Only documentation removal-list references may remain.
No runtime Python file should match.
No test Python file should match.
```

- [ ] **Step 2: Remove runtime and test residual matches**

If any runtime or test Python match remains, delete it or replace it with the export-predicate terminology.

Allowed documentation text:

```text
Remove:
SkdCommittedUnit
RESUMABLE_COMMITTED_UNITS
SkdPartialState.last_committed_unit
AsyncSkdSample validation that checks last_committed_unit
extra_fields["skd_last_committed_unit"]
```

No other documentation reference to these symbols should remain.

- [ ] **Step 3: Run residual search again**

Run:

```bash
rg -n "SkdCommittedUnit|RESUMABLE_COMMITTED_UNITS|last_committed_unit|skd_last_committed_unit|stop_after_committed_unit|ASSISTANT_GEN_CHUNK|ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT|ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT" verl/experimental/async_skd verl/experimental/agent_loop/skd_agent_loop.py tests/skd examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd -g '*.py' -g '*.md'
```

Expected:

```text
No runtime Python matches.
No test Python matches.
Only the explicit documentation removal-list block remains.
```

- [ ] **Step 4: Commit**

Run:

```bash
git add verl/experimental/async_skd/state.py verl/experimental/async_skd/__init__.py verl/experimental/agent_loop/skd_agent_loop.py tests/skd examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md
git commit -m "chore: remove residual skd safe point references"
```

## Task 5: Full Targeted Verification

**Files:**
- Verify only

- [ ] **Step 1: Run targeted async SKD test suite**

Run:

```bash
/home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m pytest -q tests/skd/test_skd_logic.py tests/skd/test_async_skd_state.py tests/skd/test_async_skd_worker.py tests/skd/test_async_skd_worker_boundary.py tests/skd/test_async_skd_data_source.py tests/skd/test_async_skd_manager.py tests/skd/test_async_skd_manager_lookahead.py tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run diff check**

Run:

```bash
git diff --check
```

Expected:

```text
no output
```

- [ ] **Step 3: Confirm no unintended dirty files are staged**

Run:

```bash
git status --short --branch
```

Expected:

```text
Only known unrelated dirty files remain outside this refactor, if they were dirty before:
 M examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_gspo_tool_longctx16k_base.sh
 m recipe
```

- [ ] **Step 4: Commit verification-only documentation if needed**

If Task 5 required documentation-only cleanup, run:

```bash
git add docs/superpowers/plans/2026-04-19-async-skd-export-predicate-refactor.md
git commit -m "docs: add async skd export predicate refactor plan"
```

If the plan was already committed before implementation starts, skip this step.

## Self-Review

Spec coverage:

```text
safe point label removal -> Task 1, Task 2, Task 3, Task 4
chunk + required tool result append as atomic scheduling unit -> Task 2
avoid over-abstraction -> Task 2 keeps one new helper only
existing handler structure preserved -> Task 2
test coverage for Qwen/Hermes export boundary -> Task 2
documentation sync -> Task 4
```

Placeholder scan:

```text
No placeholder markers remain.
```

Type consistency:

```text
stop_after_skd_chunk is the only renamed stop flag.
SkdPartialState has no last_committed_unit.
AsyncSkdSample validates source_type and payload consistency only.
committed_gen_chunks, committed_env_units, committed_prefix_tokens remain unchanged.
```
