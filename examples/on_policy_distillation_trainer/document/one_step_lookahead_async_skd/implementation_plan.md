# Bounded Async SKD Implementation Plan

## 0. Purpose

이 문서는 `skd_agent_loop.py` 기반 tool-aware SKD에 **persistent generator/trainer + bounded one-step lookahead**를 구현하기 위한 순차 계획이다.

목표는 다음이다.

- 기존 SKD semantics를 보존한다.
- Tool result와 teacher row alignment를 깨지 않는다.
- Long trajectory tail에서 생기는 idle GPU를 bounded lookahead로 줄인다.
- Staleness는 `version_lag <= 1`과 prefix budget으로 강하게 제한한다.
- 구현을 함수 단위로 작게 나눠, 각 단계마다 확인 가능한 상태로 닫는다.

비목표는 다음이다.

- 처음부터 trainer loop 전체를 갈아엎는 별도 fully async trainer를 만들지 않는다.
- `k+2`, `k+3` sample까지 무제한 prefetch하지 않는다.
- KD에 exact IS correction을 도입하지 않는다.
- Tool result 중간 interruption이나 live KV cache resume을 구현하지 않는다.
- `rollout.n > 1` group semantics를 첫 구현에서 지원하지 않는다. MVP는 `rollout.n == 1`을 hard constraint로 둔다.

## 1. Fixed Invariants

구현 중 절대 깨면 안 되는 불변식이다.

### 1.1 SKD Alignment

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

Assistant-generated token:

```text
response_mask = 1
teacher row = actual teacher top-k row
```

Tool/user/interact span:

```text
response_mask = 0
teacher row = dummy zero row
```

### 1.2 Staleness

```text
version_lag = train_consume_version - rollout_version <= 1
version_span = rollout_max_version - rollout_min_version <= 1
```

`age > 1` sample은 trainer batch에 들어가면 안 된다.

### 1.3 Lookahead Admission

```text
lookahead_step <= current_step + 1
lookahead_budget은 step 중 refill하지 않음
```

LT가 아무리 길어도 `k+2` 이상 sample을 당겨오지 않는다.

### 1.4 Tool Macro-Step

Tool result는 SKD generation chunk 안에 있지 않다. 하지만 scheduler 기준에서는 다음 전체를 atomic unit으로 본다.

```text
assistant tool-call completion
+ tool parser extraction
+ tool execution
+ tool response serialization
+ prompt_ids append
+ teacher_prompt_ids append
+ response_mask zero-span append
+ dummy teacher rows append
+ alignment assert
```

이 unit이 끝나기 전에는 carryover snapshot을 만들지 않는다.

## 2. Reference Code To Mimic

### 2.1 Keep From `skd_agent_loop.py`

File:

```text
verl/experimental/agent_loop/skd_agent_loop.py
```

Keep:

- `_handle_pending_state()`: student prompt stream과 teacher prompt stream 분리.
- `_handle_generating_state()`: SKD chunk, teacher verification, first-rejection commit.
- `_handle_processing_tools_state()`: tool result span에 dummy teacher rows append.
- `_handle_interacting_state()`: user/interact span에 dummy teacher rows append.
- `_assert_teacher_alignment()`: response mask와 teacher rows 길이 검증.

Do not change the meaning of:

```text
assistant token -> response_mask=1
tool result token -> response_mask=0
```

### 2.2 Mimic From `recipe/gkd`

Files:

```text
recipe/gkd/megatron/ray_trainer.py
recipe/gkd/megatron/megatron_workers.py
```

Mimic:

- actor/trainer와 rollout worker 분리.
- rollout worker를 persistent instance로 유지.
- trainer update 후 rollout weight sync.
- batch future/pipeline idea.

Do not copy directly:

- GKD one-step-off는 batch-level overlap이다.
- 우리의 목표는 intra-step sample-level tail filling이다.

### 2.3 Mimic From `fully_async_policy`

Files:

```text
verl/experimental/fully_async_policy/message_queue.py
verl/experimental/fully_async_policy/fully_async_rollouter.py
verl/experimental/fully_async_policy/fully_async_trainer.py
verl/experimental/fully_async_policy/agent_loop/agent_loop.py
verl/experimental/fully_async_policy/detach_utils.py
```

Mimic:

- active task set.
- pause/backpressure 개념.
- staleness accounting.
- parameter sync after trainer update.
- min/max rollout version metadata.

Do not copy directly:

- 기존 `MessageQueue`는 full이면 oldest sample을 silent drop한다.
- existing `FullyAsyncAgentLoopManager`는 distillation enabled에서 막혀 있다.
- existing partial rollout은 abort/resume 중심이고, SKD는 committed-unit-boundary pause가 필요하다.
- 현재 구현 단계에서는 Ray actor queue를 만들지 않는다. Manager 내부 task set과 local state로 충분하다.

### 2.4 Reuse And Inheritance Map

Use inheritance only at the agent-loop boundary:

```text
AsyncSkdAgentLoopWorker(AgentLoopWorker)
AsyncSkdAgentLoopManager(AgentLoopManager)
```

Do not introduce new parent classes for the trainer, source, or queue in the MVP. Existing verl code already exposes enough hooks:

| Remaining work | Reuse mechanism | New class needed? |
|---|---|---|
| Promoted dynamic batch assembly | `AsyncSkdDataSource` plus `DataProto.concat` | no |
| Carry-over current work scheduling | generalize current `AsyncSkdAgentLoopManager` task loop | no |
| Stale budget enforcement | fill `_can_continue_lookahead_partial()` | no |
| Checkpoint integration | extend `RayPPOTrainer._save_checkpoint()` and `_load_checkpoint()` | no |
| Source-aware metrics | attach values to `DataProto.meta_info` and trainer `metrics` | no |
| Config schema | add fields under distillation config | no |
| Persistent rollout/trainer split | use GKD weight-sync code as reference | later, maybe |

Do not inherit from:

```text
FullyAsyncAgentLoopManager
FullyAsyncRollouter
FullyAsyncTrainer
recipe/gkd/megatron/ray_trainer.py trainer classes
```

Reasons:

- `FullyAsyncAgentLoopManager` rejects distillation-enabled execution.
- `FullyAsyncRollouter` and `FullyAsyncTrainer` are queue-based producer/consumer components. Bounded async SKD currently runs inside one manager-local rollout step.
- `MessageQueue` drops samples when full. Bounded async SKD requires explicit promoted/carry-over/drop accounting.
- GKD performs batch pipeline overlap. Bounded async SKD performs intra-step sample-level tail filling.

Reference code should be copied only as small patterns:

- `asyncio.wait(..., FIRST_COMPLETED)` active-task handling from `fully_async_policy`.
- metric aggregation ideas from `fully_async_policy/detach_utils.py`.
- rollout-only worker and `sync_rollout_weights()` mechanics from `recipe/gkd/megatron`.
- checkpoint save/load location from `RayPPOTrainer`.

## 3. New Files

새 경로를 만든다.

```text
verl/experimental/async_skd/__init__.py
verl/experimental/async_skd/state.py
verl/experimental/async_skd/worker.py
verl/experimental/async_skd/manager.py
```

초기 구현에서는 `manager.py`가 scheduler 역할을 함께 가진다. `queue.py`, 별도 `scheduler.py`, 별도 `ray_trainer.py`는 실제로 producer/consumer 분리가 필요해질 때만 추가한다.

## 4. Phase 1: Last-Committed-Unit Instrumentation

목표: 동작을 바꾸지 않고, SKD agent loop가 마지막으로 완전히 commit한 atomic unit을 기록한다.

대상 파일:

```text
verl/experimental/agent_loop/skd_agent_loop.py
```

### 4.1 Add Helper Functions

추가 함수:

```python
def _set_skd_last_committed_unit(self, agent_data: AgentData, unit: str) -> None:
    ...

def _get_skd_last_committed_unit(self, agent_data: AgentData) -> str | None:
    ...

def _record_rollout_version_from_output(self, agent_data: AgentData, output) -> None:
    ...

def _increment_skd_prefix_stats(self, agent_data: AgentData) -> dict:
    ...
```

저장 위치:

```text
agent_data.extra_fields["skd_last_committed_unit"]
agent_data.extra_fields["rollout_min_version"]
agent_data.extra_fields["rollout_max_version"]
agent_data.extra_fields["skd_committed_gen_chunks"]
agent_data.extra_fields["skd_committed_env_units"]
agent_data.extra_fields["skd_committed_prefix_tokens"]
```

왜 필요한가:

- Scheduler가 pause 가능한 지점을 알아야 한다.
- 나중에 partial carryover가 `version_span <= 1`인지 판단해야 한다.
- Tool macro-step이 닫혔는지 외부 scheduler가 확인할 수 있어야 한다.

### 4.2 Modify `_handle_generating_state()`

수정 위치:

```text
student chunk 생성
teacher verification
first rejection commit
teacher row append
_assert_teacher_alignment()
```

이 직후:

```python
self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK")
```

그리고 committed assistant token 수만큼:

```python
skd_committed_gen_chunks += 1
skd_committed_prefix_tokens += len(new_tokens)
```

단, 이 count는 lookahead/carryover mode에서만 의미가 있다. sync mode에서는 metric으로만 남긴다.

왜 필요한가:

- `ASSISTANT_GEN_CHUNK`는 SKD chunk 단위에서 가장 작은 committed-unit boundary다.
- teacher rows와 response mask가 이미 맞춰진 상태이므로 snapshot 가능하다.

확인 포인트:

- 기존 sync SKD output shape가 변하지 않아야 한다.
- `_assert_teacher_alignment()`가 기존과 동일하게 통과해야 한다.
- `extra_fields`에 새 metric이 들어가도 downstream이 깨지지 않아야 한다.

### 4.3 Modify `_handle_processing_tools_state()`

현재 구조:

```python
prev_prompt_len = len(agent_data.prompt_ids)
prev_response_len = len(agent_data.response_mask)
next_state = await super()._handle_processing_tools_state(agent_data)
self._append_student_prompt_delta_to_teacher_stream(agent_data, prev_prompt_len)
appended_len = len(agent_data.response_mask) - prev_response_len
self._append_dummy_teacher_rows(agent_data, appended_len)
self._assert_teacher_alignment(agent_data)
```

이 직후:

```python
self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT")
```

그리고:

```python
skd_committed_env_units += 1
skd_committed_prefix_tokens += appended_len
```

왜 필요한가:

- Tool result span은 `response_mask=0`과 dummy teacher row가 같이 붙어야 한다.
- Tool execution 중간 상태는 carryover snapshot으로 쓰면 안 된다.

확인 포인트:

- Tool result token 수와 dummy teacher row 수가 항상 같아야 한다.
- `teacher_prompt_ids`에 tool result delta가 반영되어야 한다.

### 4.4 Modify `_handle_interacting_state()`

`_handle_processing_tools_state()`와 같은 방식으로:

```python
self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT")
skd_committed_env_units += 1
skd_committed_prefix_tokens += appended_len
```

왜 필요한가:

- Interaction/user span도 tool span처럼 student-generated token이 아니다.
- KD target은 dummy row여야 한다.

## 5. Phase 2: State Payload and Sample Envelope

목표: 타입 수를 늘리지 않으면서, completed sample과 carry-over sample을 같은 manager-local 인터페이스로 다룰 수 있게 한다.

대상 파일:

```text
verl/experimental/async_skd/state.py
```

### 5.1 Keep Only Boundary Enum

```python
class SkdCommittedUnit(str, Enum):
    ASSISTANT_GEN_CHUNK = "ASSISTANT_GEN_CHUNK"
    ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT = "ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT"
    ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT = "ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT"
```

`SkdCommittedUnit`은 tool macro-step atomicity와 resume 가능 boundary를 표현하므로 유지한다. 반면 source label은 scheduler accounting용 문자열이므로 별도 enum으로 만들지 않는다. `source_type`은 `AsyncSkdSample.validate()`에서 허용 문자열을 검사한다.

허용 source label:

```text
base_current
lookahead
lookahead_promoted
lookahead_carryover
resumed_current
```

### 5.2 Keep `SkdPartialState`

```python
@dataclass
class SkdPartialState:
    sample_id: str
    logical_step: int
    source_type: str
    agent_state: str
    last_committed_unit: str
    request_id: str
    tools_kwargs: dict[str, Any]
    committed_gen_chunks: int
    committed_env_units: int
    committed_prefix_tokens: int
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    teacher_prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    assistant_turns: int
    user_turns: int
    rollout_birth_version: int | None
    rollout_min_version: int | None
    rollout_max_version: int | None
    metrics: dict[str, Any]
    extra_fields: dict[str, Any]
```

왜 필요한가:

- Prefix만 저장하면 SKD resume이 불가능하다.
- Teacher stream과 dummy rows까지 같이 저장해야 한다.
- Tool state와 turn count도 prompt template에 영향을 줄 수 있다.

주의:

- Live tool object나 live interaction object를 그대로 pickle하지 않는다.
- 첫 구현에서는 text-only tool 또는 serializable tool state만 허용한다.
- `SkdPartialState`는 unfinished/carry-over payload만 표현한다. Completed sample wrapper로 재사용하지 않는다.

### 5.3 Add `AsyncSkdSample`

Completed와 partial을 각각 별도 wrapper로 만들지 않는다. Manager-local scheduler는 하나의 envelope만 다룬다.

```python
@dataclass
class AsyncSkdSample:
    sample_id: str
    kind: str  # "completed" | "partial"
    source_type: str
    logical_step: int

    batch: DataProto | None = None
    partial_state: SkdPartialState | None = None

    rollout_birth_version: int | None = None
    rollout_min_version: int | None = None
    rollout_max_version: int | None = None
    train_consume_version: int | None

    committed_gen_chunks: int
    committed_env_units: int
    committed_prefix_tokens: int

    drop_reason: str | None
    metrics: dict[str, Any]

    def validate(self) -> None:
        ...

    def require_completed(self) -> DataProto:
        ...

    def require_partial(self) -> SkdPartialState:
        ...

    @classmethod
    def from_completed(cls, *, sample_id: str, logical_step: int, source_type: str, batch: DataProto) -> "AsyncSkdSample":
        ...

    @classmethod
    def from_partial(cls, *, partial_state: SkdPartialState) -> "AsyncSkdSample":
        ...
```

왜 필요한가:

- Completed sample의 실체는 이미 trainer가 소비 가능한 `DataProto`다.
- Partial sample의 실체는 resume 가능한 `SkdPartialState`다.
- Manager는 둘을 같은 envelope로 다루되, payload 접근은 `require_completed()`와 `require_partial()`로 강제한다.
- Source-aware metrics, staleness metadata, drop reason을 한곳에서 관리한다.

엄격한 payload rule:

```text
kind == "completed":
  batch is not None
  partial_state is None
  len(batch) == 1

kind == "partial":
  batch is None
  partial_state is not None
  partial_state.logical_step == logical_step
```

선제 방지 rule:

- Optional payload 필드에 직접 접근하지 않는다.
- Manager는 active/completed/carryover collection에 넣기 전에 반드시 `sample.validate()`를 호출한다.
- Trainer batch assembly는 반드시 `sample.require_completed()`를 통과한 sample만 사용한다.
- Resume path는 반드시 `sample.require_partial()`를 통과한 sample만 사용한다.
- `source_type`은 문자열로 유지하되, 허용 값 밖이면 `validate()`에서 즉시 실패한다.
- Completed sample metadata는 envelope를 canonical source로 둔다. 같은 값을 `DataProto.non_tensor_batch`에 미러링하더라도 충돌 시 envelope 값을 우선한다.
- Partial sample metadata는 `SkdPartialState`와 envelope에 동시에 존재할 수 있으므로, 생성 시점에 값을 복사하고 `validate()`에서 불일치를 검사한다.
- Scheduler는 manual dataclass construction보다 `from_completed()`와 `from_partial()` constructor를 우선 사용한다. 이렇게 해야 metadata 복사 누락과 `kind`/payload 불일치를 줄일 수 있다.

## 6. Phase 3: Export/Restore Hooks

목표: `skd_agent_loop.py`에서 agent state를 partial state로 내보내고 복원할 수 있게 한다.

대상 파일:

```text
verl/experimental/agent_loop/skd_agent_loop.py
```

### 6.1 Add `_export_partial_state()`

```python
def _export_partial_state(
    self,
    agent_data: AgentData,
    next_state: AgentState,
    *,
    sample_id: str,
    logical_step: int,
    source_type: str,
) -> SkdPartialState:
    ...
```

검증해야 할 것:

```python
self._assert_teacher_alignment(agent_data)
assert agent_data.extra_fields["skd_last_committed_unit"] in allowed_committed_units
```

왜 필요한가:

- Carryover는 committed-unit boundary에서만 저장해야 한다.
- Export 시점에 alignment를 다시 확인해야 downstream 오류를 줄인다.

### 6.2 Add `_restore_partial_state()`

```python
def _restore_partial_state(
    self,
    partial_state: SkdPartialState,
) -> tuple[AgentData, AgentState]:
    ...
```

복원 대상:

- `messages`
- `prompt_ids`
- `teacher_prompt_ids`
- `response_mask`
- `teacher_ids_list`
- `teacher_logprobs_list`
- `assistant_turns`
- `user_turns`
- `image_data` reference if supported
- `metrics`
- `extra_fields`

왜 필요한가:

- Resumed sample은 old prefix 위에서 current version suffix를 이어서 생성한다.
- `teacher_prompt_ids`가 없으면 teacher verification 위치가 틀어진다.

### 6.3 Add `_can_export_partial_state()`

```python
def _can_export_partial_state(self, agent_data: AgentData) -> bool:
    unit = self._get_skd_last_committed_unit(agent_data)
    return unit in {
        "ASSISTANT_GEN_CHUNK",
        "ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT",
        "ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT",
    }
```

추가 rule:

- tool call이 완성되었고 아직 tool result가 없다면 `ASSISTANT_GEN_CHUNK` carryover를 되도록 금지한다.
- 이 경우 scheduler는 `ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT`까지 진행시키는 쪽을 선택한다.

## 7. Phase 4: Async SKD Worker Execution Primitives

목표: batch-level `asyncio.gather()`를 우회해 sample 단위 완료 이벤트를 얻는다.

대상 파일:

```text
verl/experimental/async_skd/worker.py
```

### 7.1 Add `AsyncSkdAgentLoopWorker`

```python
class AsyncSkdAgentLoopWorker(AgentLoopWorker):
    ...
```

`AsyncSkdAgentLoopWorker`는 `AgentLoopWorker`를 상속한다. 기존 batch path의 핵심 helper는 재사용하되, async SKD 전용 primitive는 base `AgentLoopWorker`에 추가하지 않는다.

`generate_skd_until_boundary()` is the only worker API that may return partial samples. `generate_sequence_single()` remains completed-only. `generate_skd_from_partial_to_completion()` is also completed-only, but starts from `SkdPartialState` instead of fresh `DataProto`. This separation is intentional: fresh base samples use the single-sample completion path, lookahead samples use the boundary path, and next-step carry-over samples use the partial-to-completion path.

**왜 carry-over는 반드시 완료해야 하는가.** Lookahead sample은 current batch에 포함되기 전의 투기적 실행이므로 pause가 허용된다. 반면 carry-over sample은 `next_current_batch()`가 반환한 시점에 current batch에 편입된다. Trainer는 이 step에서 B개를 학습한다는 전제로 동작하므로, carry-over가 또 pause되면 실제 학습 샘플 수가 B보다 적어지고 배치 예산 불변식이 깨진다. `version_lag <= 1` hard cap은 이 철학의 기술적 결과다. `generate_skd_from_partial_to_completion()`이 완료를 보증한다.

### 7.2 Move/Add `generate_sequence_single()`

```python
async def generate_sequence_single(
    self,
    sample: DataProto,
) -> DataProto:
    ...
```

구현 방향:

- 기존 `_run_agent_loop()`와 `_agent_loop_postprocess()`를 재사용한다.
- `len(sample) == 1`을 강제한다.
- 첫 구현에서는 completed single-sample `DataProto`만 반환한다.
- Manager가 반환된 `DataProto`를 필요 시 `AsyncSkdSample(kind="completed")`로 감싼다.
- partial state resume은 이 함수에 섞지 않는다. Lookahead/carryover resume은 SKD boundary driver를 사용하는 별도 worker entrypoint에서 처리한다.
- 완료되면 single-sample `DataProto`를 반환한다.
- pause decision은 manager가 담당한다. Paused payload는 `AsyncSkdSample(kind="partial", partial_state=...)`로 감싼다.

왜 필요한가:

- 현재 `generate_sequences()`는 batch 전체가 끝날 때까지 기다린다.
- Tail filling은 sample completion event가 있어야 가능하다.
- 이 primitive는 async SKD manager 전용이므로 `AsyncSkdAgentLoopWorker`에 둔다.

### 7.3 Add SKD Boundary Primitive

```python
async def generate_skd_until_boundary(...) -> AsyncSkdSample:
    ...
```

역할:

- fresh single-sample `DataProto` 또는 `SkdPartialState`를 받는다.
- `SkdAgentLoop._run_until_exportable_boundary()`를 호출한다.
- terminated이면 `AsyncSkdSample(kind="completed")`를 반환한다.
- exportable boundary면 `AsyncSkdSample(kind="partial")`를 반환한다.

이 함수는 SKD 전용이므로 base `AgentLoopWorker`에 두지 않는다.

### 7.4 Add SKD Partial-To-Completion Primitive

```python
async def generate_skd_from_partial_to_completion(
    partial_state: SkdPartialState,
    *,
    source_type: str = "resumed_current",
) -> AsyncSkdSample:
    ...
```

역할:

- `SkdAgentLoop.run_from_partial_to_completion()`을 호출한다.
- `_restore_partial_state()`로 복원한 뒤 `stop_after_committed_unit=False` 경로로 `TERMINATED`까지 진행한다.
- 반환값은 항상 `AsyncSkdSample(kind="completed", source_type="resumed_current")`다.
- 이 함수는 next-step current work에 들어온 carry-over sample용이다. Lookahead drain에는 사용하지 않는다.

이 함수도 SKD 전용이므로 base `AgentLoopWorker`에 두지 않는다.

### 7.5 Compatibility Checks

- single-sample output key가 batch path와 같아야 한다.
- `teacher_ids`, `teacher_logprobs`, `response_mask` shape가 기존과 맞아야 한다.

## 8. Phase 5: Manager-Level Sample Completion Path

목표: 기존 trainer/dataloader 계약을 건드리지 않고, manager가 sample 단위 완료 이벤트를 볼 수 있게 한다.

대상 파일:

```text
verl/experimental/async_skd/manager.py
```

### 8.1 Add `AsyncSkdAgentLoopManager`

```python
class AsyncSkdAgentLoopManager(AgentLoopManager):
    def __init__(...):
        ...
        self.agent_loop_workers_class = ray.remote(AsyncSkdAgentLoopWorker)

    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if mode == "sync":
            return await super().generate_sequences(prompts)
        if mode == "sample_async":
            return await self._generate_sequences_sample_async(prompts)
```

MVP constraint:

```text
actor_rollout_ref.rollout.n == 1
```

이 제약을 두는 이유:

- SKD direct distillation에서는 현재 `n=1`로 운용한다.
- `n > 1`은 prompt group, repeated rollout ordering, group-wise advantage와 충돌할 수 있다.
- lookahead accounting 단위가 prompt인지 trajectory인지 애매해지는 문제를 피한다.

### 8.2 Base Sample Submission Rule

`sample_async`는 worker당 하나씩만 submit하면 안 된다. Ray async actor는 여러 async method를 동시에 실행할 수 있고, 기존 `AgentLoopWorker.generate_sequences(chunk)`도 내부에서 chunk 내 sample을 `asyncio.gather()`로 동시에 실행한다.

따라서 base batch는 다음처럼 즉시 모두 submit한다.

```text
B = 96, workers = 4
worker 0: sample 0..23 all submitted
worker 1: sample 24..47 all submitted
worker 2: sample 48..71 all submitted
worker 3: sample 72..95 all submitted
```

이렇게 해야 기존 GPU/server request concurrency를 유지하면서, manager가 sample별 완료 이벤트를 볼 수 있다.

### 8.3 Completion Collection Rule

```python
active: dict[Task, input_pos]

while active:
    done, _ = await asyncio.wait(active.keys(), return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        pos = active.pop(task)
        outputs[pos] = await task

output = DataProto.concat([outputs[i] for i in range(len(outputs))])
```

Rule:

- 완료 순서가 아니라 input order로 `DataProto.concat()`한다.
- `uid`, `index`, reward metadata, rollout ordering을 보존한다.
- 이 단계에서는 아직 lookahead를 넣지 않는다.

## 9. Phase 6: Cooperative SKD Boundary Return

목표: lookahead sample을 강제로 cancel하지 않고, 다음 exportable committed-unit boundary에서 manager에게 제어권을 돌려준다.

대상 파일:

```text
verl/experimental/agent_loop/skd_agent_loop.py
```

### 9.1 Add `stop_after_committed_unit`

```python
async def _handle_generating_state(
    self,
    agent_data: AgentData,
    sampling_params: dict[str, Any],
    ignore_termination: bool = False,
    stop_after_committed_unit: bool = False,
) -> AgentState:
    ...
```

`stop_after_committed_unit=True`일 때:

- SKD chunk 하나를 생성한다.
- teacher verification과 first-rejection commit을 끝낸다.
- teacher rows와 `response_mask` alignment를 확인한다.
- `skd_last_committed_unit = ASSISTANT_GEN_CHUNK`를 기록한다.
- EOS/budget/max chunk가 아니면 `AgentState.GENERATING`을 반환한다.

기존 full rollout path는 `stop_after_committed_unit=False`이므로 그대로 끝까지 진행한다.

### 9.2 Preserve Pending Assistant Turn State

기존 `_handle_generating_state()`는 `turn_response_ids`를 local variable로 들고 있다가 handler 끝에서만 `agent_data.response_ids`에 넣는다. Chunk boundary에서 반환하려면 이 값이 export/restore를 지나 살아남아야 한다.

저장 위치:

```text
agent_data.extra_fields["skd_pending_turn_response_ids"]
```

Rule:

- chunk commit마다 `skd_pending_turn_response_ids += new_tokens`
- pause boundary에서는 `assistant_turns`를 증가시키지 않는다.
- restore 후 같은 assistant turn을 이어서 생성한다.
- assistant turn이 실제로 닫히면 pending field를 제거하고 `assistant_turns += 1`

### 9.3 Add `_run_until_exportable_boundary()`

```python
async def _run_until_exportable_boundary(
    self,
    agent_data: AgentData,
    state: AgentState,
    sampling_params: dict[str, Any],
) -> AgentState:
    while state != AgentState.TERMINATED:
        ...
        if self._can_export_partial_state(agent_data, state):
            return state
    return state
```

이 driver의 의미:

- `PENDING`은 prompt stream만 초기화하므로 export하지 않고 계속 진행한다.
- `GENERATING`은 `stop_after_committed_unit=True`로 chunk boundary까지 진행한다.
- `PROCESSING_TOOLS`는 tool macro-step 전체를 닫는다.
- `INTERACTING`은 interaction/user response 전체를 닫는다.
- `_can_export_partial_state()`가 허용하는 지점에서만 반환한다.

즉 manager가 실행 중 coroutine 내부를 polling하지 않고, lookahead task가 안전한 boundary에서 협력적으로 반환한다.

## 10. Phase 7: Lookahead Scheduling In Manager

목표: base sample 완료 이벤트를 이용해 step `k+1` sample을 제한적으로 시작하고, base barrier가 닫히면 active lookahead를 exportable boundary에서 drain한다.

초기 구현 위치:

```text
verl/experimental/async_skd/manager.py
```

별도 `queue.py` 또는 Ray actor queue는 만들지 않는다. Active task set, completed list, carryover list는 manager-local state로 둔다.

필수 state:

```text
base_active: dict[asyncio.Task, int]
lookahead_active: dict[asyncio.Task, int]
base_completed: list[DataProto | None]
promoted_lookahead: list[AsyncSkdSample]
carryover_partials: list[SkdPartialState]
lookahead_started_count: int
drain_requested: bool
```

새 `LookaheadTaskState` dataclass는 만들지 않는다. Active task는 아직 sample payload가 아니므로 `asyncio.Task` 자체로 추적한다. Base task에는 original input order 복원을 위해 position `int`만 붙인다. Lookahead task에는 promoted/carryover order 보존을 위한 admission order `int`만 붙인다. Sample metadata는 worker call 인자와 반환되는 `AsyncSkdSample`/`SkdPartialState` 안에 이미 존재한다.

Manager-local scheduler가 새로 도입하지 말아야 할 타입:

```text
LookaheadTaskState
SkdCompletedSample
SkdSampleSource enum
LookaheadResult
CarryoverSample
PromotedSample
```

결과 payload는 기존 envelope만 사용한다.

```text
completed result -> AsyncSkdSample(kind="completed", batch=DataProto)
partial result -> AsyncSkdSample(kind="partial", partial_state=SkdPartialState)
```

흐름:

```text
1. base samples are all submitted upfront.
2. each base completion creates an opportunity to admit lookahead.
3. lookahead task runs only until completed or exportable boundary.
4. if base is not fully done, exportable lookahead partial can be resumed for another unit.
5. once all base samples finish, set drain_requested=True.
6. active lookahead tasks finish their current safe unit and return.
7. completed lookahead -> promoted.
8. partial lookahead -> carryover.
9. train batch = base_completed + promoted_lookahead.require_completed().
```

### 10.1 Lookahead Admission Rule

Mode entry에서 한 번 검증할 구조적 invariant:

```text
rollout.n == 1
```

이 값은 sample-level scheduling의 전제다. Step 중 admission predicate에 매번 섞지 않는다.

Runtime admission predicate:

```text
lookahead target step = current_step + 1
lookahead_started_count <= L_prefetch
budget is not refilled within the same step
drain_requested == False
base_active is not empty
future source has next sample
```

`L_prefetch`는 sample 단위로 계산한다. `rollout.n == 1`이므로 prompt/sample/trajectory 단위가 일치한다.

### 10.2 Drain Rule

Base batch가 모두 끝나는 순간:

```text
drain_requested = True
```

이후 lookahead task는 강제 cancel하지 않는다. 이미 실행 중인 unit을 끝낸 뒤 다음 중 하나로 반환한다.

```text
completed -> promote
partial at exportable boundary -> carryover
```

drain 요청 후 추가로 허용되는 overshoot는 최대 하나의 committed generation chunk 또는 하나의 tool/interact macro-step이다.

## 11. Phase 8: Dataloader-Aware Future Sample Source

목표: lookahead admission이 사용할 step `k+1` sample을 기존 `StatefulDataLoader`/sampler/checkpoint semantics와 충돌하지 않게 공급한다.

초기 list-materialized accounting prototype은 제거했다. 해당 prototype은 전체 sample list를 메모리에 들고, promoted sample까지 next-step fresh quota에서 차감하는 오래된 규칙을 사용했기 때문에 실제 production source로 유지하지 않는다.

현재 구현된 MVP는 `verl/experimental/async_skd/data_source.py`의 `AsyncSkdDataSource`다. 이 클래스는 `StatefulDataLoader`를 대체하지 않는다. Dataloader iterator에서 collated `batch_dict`를 하나씩 받아 `DataProto.from_single_dict(batch_dict)`로 fresh buffer를 만들고, `DataProto[pos:pos+1]` 방식으로 single-sample `DataProto`를 공급한다.

MVP 책임:

```text
pop_fresh_sample()
reserve_lookahead(logical_step)
record_promoted(samples)
record_carryover(partials, input_batches=None)
next_current_batch(base_batch_size)
next_fresh_quota(base_batch_size)
state_dict()
load_state_dict(state)
```

Manager 결선:

```text
AsyncSkdAgentLoopManager.set_async_skd_data_source(source)
_next_lookahead_sample(logical_step) -> source.reserve_lookahead(logical_step)
completed lookahead -> keep AsyncSkdSample envelope until source.record_promoted(...)
partial lookahead -> source.record_carryover(...)
final train output -> base DataProto + promoted.require_completed()
```

MVP `state_dict()`는 source-local state만 저장한다.

```text
fresh_buffer
fresh_cursor
carryover_partials
carryover_input_batches
reserved_input_batches
trained_reserved_sample_ids
```

`StatefulDataLoader.state_dict()`와 함께 checkpoint에 저장하고 복원하는 trainer integration은 아직 별도 단계다.

실제 구현 방향:

```text
StatefulDataLoader iterator
  -> AsyncSkdDataSource
      -> small fresh buffer
      -> reserved lookahead buffer
      -> carryover partial buffer
```

Rule:

- 전체 dataset이나 전체 dataloader output을 `list()`로 materialize하지 않는다.
- `StatefulDataLoader.state_dict()`를 기존대로 존중한다.
- trainer checkpoint에는 dataloader state와 source-local buffer state를 함께 저장한다. 현재 MVP source는 source-local buffer state만 제공한다.
- curriculum sampler가 켜져 있으면 MVP에서는 lookahead를 비활성화한다.
- `rollout.n != 1`이면 MVP에서는 async SKD lookahead path를 거부한다.

### 11.1 Core Accounting Rule

Step `k`에서:

```text
promoted_count = Delta_k
carryover_count = R_k
```

라면 step `k+1`의 current work 구성은:

```text
resume_carryover_{k+1} = R_k
fresh_quota_{k+1} = B - R_k
```

이다. `Delta_k`는 step `k`에서 이미 학습된 reserved future samples 수다. 따라서 step `k+1` fresh quota에서 다시 차감하지 않는다. 대신 source/reservation ledger가 `Delta_k` sample을 다시 emit하지 않도록 보장해야 한다.

예시:

```text
B = 96
Delta_k = 18
R_k = 30

step k train batch = 96 + 18 = 114
step k+1 current work = resume 30 + fresh 66 = 96
```

`lookahead_started_count`는 step-local counter이며 step `k+1`에서 다시 0부터 시작한다.

### 11.2 Manager Current-Work Assembly

Trainer integration 전에 manager는 carry-over partial과 fresh prompts를 같은 current step work로 실행할 수 있어야 한다.

```python
async def generate_sequences_with_carryover(
    *,
    fresh_prompts: DataProto | None,
    carryover_partials: list[SkdPartialState],
) -> DataProto:
    ...
```

초기 contract:

```text
carryover_partials -> generate_skd_from_partial_to_completion()
fresh_prompts -> generate_sequence_single()
output order -> carryover completed first, fresh completed after
lookahead admission -> not enabled in this method yet
rollout.n -> must be 1
```

이 메서드는 next-step current work를 닫기 위한 중간 API다. 즉 `carryover 30 + fresh 66`을 모두 terminal completed `DataProto`로 만든다. Promoted lookahead까지 붙이는 trainer-level dynamic batch assembly는 이후 단계에서 처리한다.

Source assembly contract:

```text
AsyncSkdDataSource.next_current_batch(B)
  -> carryover_partials
  -> fresh_batch
  -> current_input_batch
```

`current_input_batch`는 carry-over input rows first, fresh rows after 순서다. 이 순서는 `generate_sequences_with_carryover()`의 output order와 동일해야 한다. Trainer는 이후 `current_input_batch.union(gen_batch_output)`을 수행하므로 row 수와 uid 순서가 맞아야 한다.

## 12. Phase 9: Trainer Integration

목표: 기존 PPO/SKD update path는 최대한 재사용한다.

처음에는 trainer subclass를 만들지 않는다. 기존 `RayPPOTrainer`가 지원하는 custom `AgentLoopManager` hook을 사용한다.

설정 예:

```text
actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager
actor_rollout_ref.rollout.agent.async_skd_mode=sample_async
actor_rollout_ref.rollout.n=1
```

이후 lookahead까지 붙으면 mode를 추가한다.

```text
actor_rollout_ref.rollout.agent.async_skd_mode=lookahead
```

Trainer loop 자체는 가능한 한 유지하되, batch acquisition은 helper로 분리한다.

```text
_iter_training_batches()
  sync/sample_async:
    -> [], batch, batch
  lookahead:
    -> carryover_partials, fresh_batch, current_input_batch
```

Trainer entry helpers:

```text
_ensure_batch_uid(batch)
_async_skd_mode()
_is_async_skd_lookahead_enabled()
_validate_async_skd_lookahead_constraints()
_ensure_async_skd_data_source()
```

MVP guard:

```text
rollout.n == 1
REMAX disabled
rollout skip disabled
curriculum sampler disabled
AsyncSkdAgentLoopManager-compatible rollout manager required
```

Lookahead mode generation:

```text
fresh_batch -> _get_gen_batch(fresh_batch)
carryover_partials + fresh_gen_batch -> generate_sequences_with_carryover(...)
current_input_batch.union(gen_batch_output)
```

Dynamic batch size `B + promoted_count`가 생기는 시점에는 reward/advantage/update path가 실제 token mask 기준으로 normalize되는지 별도 검증한다.

### 12.1 Promoted Dynamic Batch Assembly

This is the next required implementation step before adding more lookahead concurrency.

Problem:

```text
lookahead manager output = base_gen_output + promoted_gen_output
trainer current input batch = base_input_batch
DataProto.union() requires equal row count
```

Therefore promoted output cannot be returned without the matching promoted input rows.

Required source API:

```python
class AsyncSkdDataSource:
    def record_promoted(self, samples: list[AsyncSkdSample]) -> None:
        ...

    def pop_promoted_input_batches(self) -> list[DataProto]:
        ...
```

Required manager behavior:

```text
completed lookahead -> keep AsyncSkdSample envelope
source.record_promoted(promoted_samples)
manager returns base outputs + promoted outputs
```

Required trainer behavior:

```text
base_input_batch = batch from _iter_training_batches()
promoted_input_batch = source.pop_promoted_input_batches()
if promoted_input_batch:
    batch = DataProto.concat([base_input_batch, promoted_input_batch])
batch = batch.repeat(...)
batch = batch.union(gen_batch_output)
```

Ordering invariant:

```text
input rows:  base inputs, promoted inputs
output rows: base outputs, promoted outputs
```

The source must preserve promoted input order using the same admission order used by manager output ordering.

Tests:

```text
tests/skd/test_async_skd_data_source.py
  - record_promoted stores promoted input rows in output order
  - pop_promoted_input_batches returns and clears rows

tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
  - trainer assembles B + Delta input rows before union
  - legacy dataloader-only path is unchanged
```

## 13. Phase 10: Remaining Manager And Trainer Closure

목표: persistent rollout/trainer split에 들어가기 전에 현재 manager-local lookahead path의 row accounting, staleness cap, checkpoint, and repeated step behavior를 닫는다.

### 13.1 Stale Budget Enforcement

대상:

```text
verl/experimental/async_skd/manager.py
tests/skd/test_async_skd_manager_lookahead.py
```

Use existing hook:

```text
_can_continue_lookahead_partial(partial_state)
```

Do not create a new scheduler class for this. The required fields already live in `SkdPartialState`:

```text
rollout_min_version
rollout_max_version
committed_gen_chunks
committed_env_units
committed_prefix_tokens
```

The predicate must enforce:

```text
committed_gen_chunks <= max_old_gen_chunks
committed_env_units <= max_old_env_units
committed_prefix_tokens <= old_prefix_token_cap
version_span <= max_version_span
version_lag <= max_version_lag
```

### 13.2 Lookahead Admission During Carry-Over Current Work

대상:

```text
verl/experimental/async_skd/manager.py
tests/skd/test_async_skd_manager_lookahead.py
```

`generate_sequences_with_carryover()` should reuse the same task-loop structure as `_generate_sequences_lookahead()`. Do not copy the whole function. Extract or generalize the common loop so current work can contain:

```text
fresh base samples
resumed carry-over samples
```

The current-work output order remains:

```text
carry-over completed first
fresh completed second
promoted lookahead appended after current work
```

### 13.3 Source Checkpoint Integration

대상:

```text
verl/trainer/ppo/ray_trainer.py
tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
```

Extend existing `RayPPOTrainer._save_checkpoint()` and `_load_checkpoint()`. Do not add a separate checkpoint engine.

New `data.pt` format:

```python
{
    "train_dataloader": train_dataloader_state,
    "async_skd_source": async_skd_source_state,
}
```

Loader must support legacy format:

```text
legacy data.pt == train_dataloader_state
```

The checkpoint must preserve:

```text
fresh_buffer
fresh_cursor
reserved_input_batches
trained_reserved_sample_ids
carryover_partials
carryover_input_batches
promoted input rows if present
```

## 14. Phase 11: Drift and Source-Aware Metrics

목표: 논문 방어에 필요한 empirical diagnostics를 남긴다.

대상:

```text
async_skd/manager.py
skd_agent_loop.py
metric aggregation path
```

### 14.1 Required Metrics

```text
async_skd/promoted_count
async_skd/carryover_count
async_skd/drop_age_gt_1
async_skd/lookahead_budget_used
async_skd/intentional_idle_time
async_skd/version_lag_mean
async_skd/version_lag_max
async_skd/version_span_max
async_skd/stale_sample_ratio
async_skd/stale_token_ratio
async_skd/old_gen_chunks_p95
async_skd/old_env_units_max
async_skd/old_prefix_tokens_p95
async_skd/logprob_delta_mean
async_skd/logprob_delta_p95
```

### 14.2 Logprob Drift

계산:

```text
delta_t = log pi_current(a_t | h_t) - log pi_rollout(a_t | h_t)
```

보고:

```text
mean_abs_delta
p95_abs_delta
sequence_mean_delta
source_type별 delta
age별 delta
```

왜 필요한가:

- “현재 정책은 해당 trajectory를 낮은 확률로 만들지 않는가?”라는 반박에 대응한다.
- 수학적으로 완전 보장하지 않고, 실제 drift가 작다는 진단으로 방어한다.

## 15. Phase 12: Config

대상:

```text
verl/workers/config/distillation.py
verl/trainer/config/distillation/distillation.yaml
```

### 15.1 Add Config Dataclass

```python
@dataclass
class AsyncSkdLookaheadConfig(BaseConfig):
    enable: bool = False
    persistent_rollout_engines: bool = True
    max_version_lag: int = 1
    max_version_span: int = 1
    prefetch_admission_ratio: float = 0.25
    prefetch_pair_quota: bool = True
    max_old_gen_chunks: int = 2
    max_old_env_units: int = 1
    old_prefix_token_ratio: float = 0.125
    promote_finished: bool = True
    drop_age_gt: int = 1
    tool_macro_step_atomic: bool = True
    source_aware_metrics: bool = True
```

Then attach to:

```python
class DistillationConfig:
    async_skd_lookahead: AsyncSkdLookaheadConfig
```

왜 distillation config인가:

- 이 기능은 일반 PPO async가 아니라 SKD semantics에 묶여 있다.
- Teacher row alignment, SKD chunk, top-k teacher target과 직접 연결된다.

## 16. Phase 13: Persistent Rollout Weight Sync

목표: trainer update 후 student rollout engine에 weight를 sync한다.

주의: MVP에서 per-sample committed-boundary sync를 가정하지 않는다. vLLM/SGLang weight update는 engine-level operation일 수 있으므로, 처음에는 다음 중 안전한 정책을 사용한다.

```text
1. step boundary drain 후 sync
2. 새 sample admission부터 새 version worker에 배정
3. active old-version lookahead는 version_lag <= 1 안에서만 carryover 허용
```

기록해야 할 metadata:

```text
rollout_min_version
rollout_max_version
train_consume_version
version_lag
version_span
```

왜 필요한가:

- `version_lag <= 1`을 검증한다.
- Drift metric과 benchmark 해석의 기준이 된다.

Reference:

```text
recipe/gkd/megatron/ray_trainer.py::sync_rollout_weights()
recipe/gkd/megatron/megatron_workers.py::sync_rollout_weights()
verl/experimental/fully_async_policy/fully_async_trainer.py::_fit_update_weights()
```

Do not inherit the GKD trainer. Copy only the weight-sync mechanics and version metadata pattern.

## 17. Recommended Patch Order

한 번에 크게 바꾸지 않는다.

### Patch 1: Safe-Point Metadata Only

Files:

```text
skd_agent_loop.py
```

Changes:

- `_set_skd_last_committed_unit()`
- `_record_rollout_version_from_output()`
- `_increment_skd_prefix_stats()`
- generating/tool/interact committed-unit writes

Expected behavior:

- 기존 output은 변하지 않음.
- extra metrics만 추가.

### Patch 2: State Payload and Sample Envelope

Files:

```text
async_skd/state.py
async_skd/__init__.py
```

Changes:

- `SkdPartialState`
- `AsyncSkdSample`
- `SkdCommittedUnit`
- avoid `LookaheadTaskState`; manager active task bookkeeping uses `asyncio.Task` collections
- remove/avoid `SkdCompletedSample`
- remove/avoid `SkdSampleSource` enum; validate `source_type` strings in `AsyncSkdSample.validate()`

Expected behavior:

- Runtime path와 아직 연결하지 않음.
- Completed payload는 `DataProto`, partial payload는 `SkdPartialState`로만 접근 가능.

### Patch 3: Export/Restore

Files:

```text
skd_agent_loop.py
```

Changes:

- `_export_partial_state()`
- `_restore_partial_state()`
- `_can_export_partial_state()`

Expected behavior:

- 별도 debug path에서 export/restore roundtrip 가능.
- 기존 rollout path는 그대로.

### Patch 4: Async SKD Worker Primitive

Files:

```text
async_skd/worker.py
async_skd/manager.py
```

Changes:

- `AsyncSkdAgentLoopWorker(AgentLoopWorker)`
- `generate_sequence_single()`
- `AsyncSkdAgentLoopManager` uses `AsyncSkdAgentLoopWorker`

Expected behavior:

- batch size 1 output이 기존 `generate_sequences()` output과 compatible.
- base `AgentLoopWorker` remains generic.

### Patch 5: Manager Sample-Async Path

Files:

```text
async_skd/manager.py
```

Changes:

- `AsyncSkdAgentLoopManager`
- `sync` mode delegates to existing `AgentLoopManager`
- `sample_async` mode submits all base samples upfront
- `FIRST_COMPLETED` collection
- input-order `DataProto.concat`
- `rollout.n == 1` guard

Expected behavior:

- 기존 worker chunk path와 같은 request concurrency 유지.
- sample별 완료 이벤트를 manager가 볼 수 있음.
- output schema와 order가 기존 path와 compatible.

### Patch 6: Cooperative Boundary Return

Files:

```text
skd_agent_loop.py
async_skd/worker.py
```

Changes:

- `_handle_generating_state(..., stop_after_committed_unit=True)`
- `skd_pending_turn_response_ids`
- `_run_until_exportable_boundary()`
- tool/interact macro-step closure before export
- `AsyncSkdAgentLoopWorker.generate_skd_until_boundary()`

Expected behavior:

- 기존 full rollout path는 그대로.
- lookahead path는 `ASSISTANT_GEN_CHUNK` 또는 tool/interact closed boundary에서 반환 가능.
- chunk pause 후 export/restore/resume 가능.

### Patch 7: Manager Lookahead Scheduling

Changes:

- lookahead admission
- budget cap
- finished lookahead promotion
- active lookahead boundary return
- drain_requested handling
- unfinished lookahead carryover

Expected behavior:

- base sample 완료 즉시 lookahead 시작 가능.
- base barrier가 닫히면 active lookahead는 다음 exportable boundary에서 멈춤.
- completed lookahead는 promoted, partial lookahead는 carryover.

### Patch 8: Dataloader-Aware Future Sample Source

Changes:

- `AsyncSkdDataSource`
- no full dataset materialization
- small active buffers only
- `StatefulDataLoader.state_dict()` preservation
- curriculum sampler guard
- `rollout.n == 1` guard

Expected behavior:

- no duplicate or skipped prompt under lookahead.
- next fresh quota is `B - carryover_count`.
- promoted samples are tracked in the source ledger for duplicate prevention, not subtracted from next-step fresh quota.

### Patch 9: Promoted Dynamic Batch Assembly

Files:

```text
verl/experimental/async_skd/data_source.py
verl/experimental/async_skd/manager.py
verl/trainer/ppo/ray_trainer.py
tests/skd/test_async_skd_data_source.py
tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
```

Changes:

- store promoted input rows when lookahead samples complete.
- expose a source method that returns promoted input rows once, in promoted output order.
- assemble trainer input as `base_input + promoted_input` before `DataProto.union()`.
- keep `fresh_quota = B - carryover_count`; promoted rows are not subtracted from next-step fresh quota.

Expected behavior:

- if manager returns `B + Delta` generation outputs, trainer input batch also has `B + Delta` rows.
- `DataProto.union()` sees matching row counts.
- promoted sample `uid` appears exactly once in training.

### Patch 10: Stale Budget Enforcement

Files:

```text
verl/experimental/async_skd/manager.py
tests/skd/test_async_skd_manager_lookahead.py
```

Changes:

- implement `_can_continue_lookahead_partial()`.
- read `max_old_gen_chunks`, `max_old_env_units`, `old_prefix_token_ratio`, `max_version_lag`, and `max_version_span`.
- carry over or drop partials that exceed budget instead of continuing old-version rollout.

Expected behavior:

- lookahead partials do not keep advancing indefinitely while base LT is still running.
- stale prefix size is bounded before resume.

### Patch 11: Lookahead Admission During Carry-Over Current Work

Files:

```text
verl/experimental/async_skd/manager.py
tests/skd/test_async_skd_manager_lookahead.py
```

Changes:

- generalize current task loop so current work can be either base fresh samples or resumed carry-over samples.
- allow `generate_sequences_with_carryover()` to admit lookahead while carry-over + fresh current work is active.
- preserve current output order as carry-over completed first, fresh completed second.
- append promoted outputs after current outputs.

Expected behavior:

- every step can perform bounded tail filling, including steps that start with carry-over.
- current work still completes before train update.

### Patch 12: Source Checkpoint Integration

Files:

```text
verl/trainer/ppo/ray_trainer.py
tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
```

Changes:

- save `train_dataloader.state_dict()` and `AsyncSkdDataSource.state_dict()` together.
- load both when lookahead mode is enabled.
- support legacy `data.pt` files that contain only dataloader state.

Expected behavior:

- resume preserves fresh cursor, reserved input rows, promoted ledger, carry-over partials, and carry-over input rows.
- legacy checkpoint loading still works.

### Patch 13: Source-Aware Metrics

Files:

```text
verl/experimental/async_skd/manager.py
verl/experimental/async_skd/state.py
verl/trainer/ppo/ray_trainer.py
tests/skd/test_async_skd_manager_lookahead.py
```

Changes:

- add promoted/carry-over/drop counts.
- add stale prefix counters from `SkdPartialState`.
- add version lag/span counters when version metadata exists.
- place metrics in `DataProto.meta_info` or trainer `metrics` without changing train tensors.

Expected behavior:

- training logs can separate base current, promoted lookahead, and resumed carry-over samples.
- benchmark claims can be tied to observed staleness and drift proxies.

### Patch 14: Config Schema

Files:

```text
verl/workers/config/distillation.py
verl/trainer/config/distillation/distillation.yaml
examples/on_policy_distillation_trainer/run_*skd*.sh
```

Changes:

- add async SKD lookahead config fields.
- keep ad hoc config keys as backward-compatible aliases only if needed.
- validate `rollout.n == 1`, REMAX disabled, rollout skip disabled, and curriculum sampler disabled.

Expected behavior:

- experiment scripts can enable lookahead through explicit config fields.
- stale budget values are no longer hard-coded in manager methods.

### Patch 15: Persistent Weight Sync

Changes:

- student rollout engine sync after trainer update
- step-boundary/drain-safe sync
- version metadata

Expected behavior:

- `version_lag <= 1`.
- `version_span <= 1`.

Reference:

```text
recipe/gkd/megatron/ray_trainer.py::sync_rollout_weights()
recipe/gkd/megatron/megatron_workers.py::sync_rollout_weights()
verl/experimental/fully_async_policy/fully_async_trainer.py::_fit_update_weights()
```

Do not inherit the GKD trainer. Copy only the weight-sync mechanics and version metadata pattern.

### Patch 16: Drift Metrics

Changes:

- source-aware drift logs
- stale ratio logs
- intentional idle logs

Expected behavior:

- 논문 방어용 diagnostics 수집 가능.

## 18. Step-by-Step Validation Gates

각 patch마다 최소 확인해야 하는 항목이다.

### Gate A: Alignment

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

Tool span:

```text
response_mask=0 row count == dummy teacher row count
```

### Gate B: DataProto Compatibility

Trainer에 들어가는 batch가 기존 update path에 필요한 key를 가진다.

Required:

```text
prompts
responses
input_ids
attention_mask
position_ids
response_mask
teacher_ids
teacher_logprobs
```

### Gate C: Staleness

```text
max(version_lag) <= 1
max(version_span) <= 1
```

### Gate D: Lookahead Accounting

```text
started_lookahead <= L_prefetch
next_fresh_quota = B - carryover_count
promoted_count affects source duplicate prevention only
no k+2 sample admitted
```

### Gate E: Tool Atomicity

No snapshot while:

```text
tool execution running
tool response partially appended
response_mask updated but dummy teacher rows not appended
teacher_prompt_ids missing tool result delta
```

### Gate F: Drift Diagnostics

At minimum:

```text
age=0 vs age=1 logprob_delta_mean
age=0 vs age=1 distillation/loss
source_type별 response length
source_type별 tool call count
```

## 19. Expected First Implementation Scope

첫 구현은 conservative하게 잡는다.

```text
prefetch_admission_ratio = 0.25
max_version_lag = 1
max_version_span = 1
max_old_gen_chunks = 2
max_old_env_units = 1
old_prefix_token_ratio = 0.125
promote_finished = true
tool_macro_step_atomic = true
```

Feature scope:

- Direct SKD only.
- Text-only tool부터 시작.
- Promote-only lookahead를 먼저 구현.
- Carryover resume은 그 다음 단계.
- No exact IS correction.
- No mid-tool interruption.
- No live KV cache resume.

## 20. Core Rationale

이 구현은 GPU를 항상 100% 쓰는 fully streaming 시스템이 아니다. 정확한 목표는 다음이다.

```text
work-conserving under bounded-staleness constraints
```

즉 안전한 non-stale 또는 one-step-stale 작업이 있으면 idle pair에 배정한다. 하지만 lookahead budget이나 staleness cap을 넘으면 의도적으로 idle을 허용한다.

이 제약이 있어야 다음 주장을 할 수 있다.

- GKD처럼 staleness를 받아들이지만, batch 전체를 stale로 만들지는 않는다.
- Finished lookahead는 age 0이므로 current batch에 promote한다.
- Unfinished lookahead는 committed-unit boundary에서만 carryover한다.
- Tool result는 loss 대상이 아니지만 tool macro-step으로 atomic하게 다룬다.
- Current-policy forward가 stale logits 문제는 없애지만, context distribution shift는 남으므로 drift metric을 보고한다.
