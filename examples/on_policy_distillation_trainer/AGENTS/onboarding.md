# On-Policy Distillation 빠른 온보딩

## 이 문서의 역할

이 문서는 이 작업 영역의 **entry point** 다.  
목표는 처음 들어온 사람이 짧은 시간 안에 다음 네 가지를 바로 파악하게 하는 것이다.

1. 현재 실험이 무엇을 하려는지
2. 어디부터 코드를 읽어야 하는지
3. 어떤 로그를 먼저 봐야 하는지
4. 무엇이 correctness 이슈이고, 무엇이 systems/perf 이슈인지

더 자세한 설명은 [`AGENTS/imp_detail.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/imp_detail.md) 로 내려간다.

## 현재 작업의 한 줄 요약

현재 작업은 **tool-aware speculative knowledge distillation (SKD)** 를 `verl`의 agent-loop 위에 구현하고,  
학생 `Qwen3-1.7B`를 교사 `Qwen3-32B-FP8`로부터 on-policy distillation 하는 것이다.

현재 실험의 주요 특징은 다음과 같다.

- `skd_agent` 기반 chunked generation + teacher verification
- tool-aware multi-turn rollout (`code_interpreter`)
- teacher-only system prompt 지원
- bounded async SKD lookahead (`AsyncSkdAgentLoopManager`)
- event-log 기반 dashboard 관측
- train: `Nemotron-Cascade-RL-Math`
- validation: `AIME-2024` + `MATH500`
- validation sampling: `n=4` (`mean@4`, `best@4`)

## 가장 먼저 볼 파일

다음 순서로 보면 가장 빠르다.

1. 실행 스크립트
   [`run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh)
   현재 bounded async SKD 실험의 기본 entry point다. `AsyncSkdAgentLoopManager`, lookahead prefetch, event log, dashboard용 설정이 여기서 결정된다. 파일명에는 `longctx16k`가 남아 있지만 현재 기본값은 `MAX_PROMPT=512`, `MAX_RESPONSE_LENGTH=8192`다.

2. 동기 SKD 비교 스크립트
   [`run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt_v2.sh`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt_v2.sh)
   최근 teacher-prompt 실험의 학생/교사 모델, batch, validation `n=4`, teacher prompt, dataset 경로가 여기서 결정된다.
   이전 비교 기준이 필요하면 [`run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt.sh`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt.sh)도 함께 본다.

3. SKD 루프 본체
   [`skd_agent_loop.py`](/home/work/DDAI_revised/OSworld/verl/verl/experimental/agent_loop/skd_agent_loop.py)
   chunk generate, teacher verify, tool-aware teacher alignment, teacher-only prompt stream이 여기에 있다.

4. async SKD manager / source
   [`manager.py`](/home/work/DDAI_revised/OSworld/verl/verl/experimental/async_skd/manager.py), [`data_source.py`](/home/work/DDAI_revised/OSworld/verl/verl/experimental/async_skd/data_source.py)
   current work, lookahead, promoted, carryover, per-worker admission target, source ledger가 여기에 있다.

5. teacher interface
   [`teacher_manager.py`](/home/work/DDAI_revised/OSworld/verl/verl/experimental/teacher_loop/teacher_manager.py)
   teacher prompt-logprob contract와 backend별 차이를 여기서 흡수한다.

6. SGLang wrapper
   [`async_sglang_server.py`](/home/work/DDAI_revised/OSworld/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py)
   `prompt_logprobs_start_len`, teacher payload 반환 규약, max context error가 여기서 나온다.

7. teacher tensor 재구성
   [`agent_loop.py`](/home/work/DDAI_revised/OSworld/verl/verl/experimental/agent_loop/agent_loop.py)
   SKD가 online으로 모은 teacher rows를 downstream distillation 경로가 이해하는 형태로 다시 맞춘다.

## 현재 실험에서 중요한 입력/리소스

### 모델
- student: [`checkpoints/Qwen3-1.7B`](/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-1.7B)
- teacher: [`checkpoints/Qwen3-32B-FP8`](/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-32B-FP8)

### 데이터
- train: [`data/nemotron_cascade_rl_math_multiturn_w_tool/train.parquet`](/home/work/DDAI_revised/OSworld/verl/data/nemotron_cascade_rl_math_multiturn_w_tool/train.parquet)
- val:
  - [`data/aime-2024.parquet`](/home/work/DDAI_revised/OSworld/verl/data/aime-2024.parquet)
  - [`data/math500/test.parquet`](/home/work/DDAI_revised/OSworld/verl/data/math500/test.parquet)

### teacher prompt
- primary / recent probe: [`teacher_system_prompt_v2.txt`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v2.txt)
- older baseline: [`teacher_system_prompt_v1.txt`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v1.txt)

현재 async SKD 스크립트의 기본값은 `teacher_system_prompt_v1.txt`다. 동기 teacher-prompt v2 스크립트는 `teacher_system_prompt_v2.txt`를 기본값으로 쓴다.

### async SKD 기본값
- `TRAIN_PROMPT_BSZ=96`
- `PREFETCH_LIMIT=64`
- `PREFETCH_WORKER_TARGET=20`
- `SKD_CHUNK_SIZE=64`
- `SKD_VERIFY_TOP_K=3`
- `SKD_MAX_CHUNKS=512`
- `MAX_RESPONSE_LENGTH=8192`
- `STUDENT_MAX_TOKEN_LEN_PER_GPU=12000`

`PREFETCH_LIMIT=0`은 lookahead prefetch만 끈다. 이 경우에도 `AsyncSkdAgentLoopManager`의 sample-level current scheduling은 남으므로 기존 동기 SKD와 완전히 같다고 보면 안 된다. 동기 baseline은 동기 SKD 스크립트를 따로 사용한다.

## 구현에서 기억할 핵심 계약

### 1. SKD는 first-rejection 방식이다

학생 chunk 안에서 첫 rejection이 나오면:
- reject 이전 accepted prefix만 유지
- reject 위치는 teacher top-1로 교체
- 그 뒤 학생 suffix는 버린다

즉 distillation target도 항상 **실제로 커밋된 경로**에만 맞아야 한다.

### 2. tool/user span은 `response_mask=0`이다

tool-aware trajectory에서는 response 안에:
- assistant token
- tool response token
- interaction/user token

이 섞인다.  
현재 구현은 tool/user span에도 dummy teacher row를 같이 넣어서,

- `len(response_mask) == len(teacher_ids_list)`

를 유지한다.

### 3. teacher는 별도 prompt stream을 쓴다

teacher-only system prompt가 들어가면 student prefix와 teacher prefix가 달라진다.  
그래서 현재는:

- student: `agent_data.prompt_ids`
- teacher: `agent_data.extra_fields["teacher_prompt_ids"]`

를 따로 유지한다.

teacher verification은 항상 teacher prompt stream 기준으로 이뤄진다.

### 4. teacher-only prompt를 쓰면 teacher budget reserve가 자동으로 붙는다

teacher prompt가 student prompt보다 길어지므로,
config 레벨에서 teacher inference budget에 고정 `512` 토큰 reserve를 자동으로 더한다.

즉 run script의 raw `max_model_len` 값만 보면 teacher 실제 예산을 과소평가할 수 있다.

### 5. bounded async SKD는 promoted와 carryover를 분리한다

`lookahead` sample이 step barrier 전에 `TERMINATED`가 되면 promoted sample이다. trainer는 promoted input/output pair를 현재 train batch 뒤에 붙인다. 단, final train batch가 DP size로 나누어지도록 append 가능한 수만 붙이고, 남은 promoted pair는 source ledger에 pending으로 남긴다.

`lookahead` sample이 완료되지 못하면 partial carryover다. carryover는 다음 step의 current work 앞쪽에 들어가고, 그만큼 fresh quota가 줄어든다. promoted row는 next-step fresh quota에서 차감하지 않는다.

## 로그에서 먼저 볼 것

### correctness
- `teacher_mass_max`
- `distillation/loss_max`
- `AssertionError`
- `Prompt length ... exceeds ...`

### rollout dynamics
- `[SKD] ... done=eos / max_chunks / budget_exhausted`
- `avg_tok/chunk`
- `accept`, `reject`, `rate`
- `student=...ms`, `teacher=...ms`
- `[ASYNC_SKD] step_input`
- `[ASYNC_SKD] rollout`
- `[ASYNC_SKD] train`

### async SKD dashboard

dashboard는 event log를 tail해서 scheduler worker, student replica, teacher replica, LT candidate, carryover/promoted 상태를 보여준다.

```bash
cd /home/work/DDAI_revised/OSworld/verl

nohup /home/work/DDAI_revised/miniconda3/envs/kd/bin/python -m verl.experimental.async_skd.dashboard \
  --event-log /home/work/DDAI_revised/OSworld/verl/logs/async_skd_events_live.jsonl \
  --host 0.0.0.0 \
  --port 10001 \
  > /home/work/DDAI_revised/OSworld/verl/logs/async_skd_dashboard_live.log 2>&1 &
```

훈련은 `VERL_ASYNC_SKD_EVENT_LOG`가 가리키는 JSONL에 이벤트를 쓴다. 기록 보존이 필요하면 run별 event log를 만들고 `logs/async_skd_events_live.jsonl`을 symlink로 연결한다.

### training / systems
- `step:`
- `time/step`
- `throughput`
- `torch.OutOfMemoryError`
- `ActorDiedError`

## 현재까지 자주 나온 문제 유형

1. **teacher row alignment 문제**  
   tool response가 중간에 들어가는데 teacher row를 assistant token만 기준으로 쌓으면 `1280` pathology가 재발한다.

2. **teacher prompt로 인한 context budget 초과**  
   teacher-only prompt를 넣으면 teacher prefix가 길어지므로 별도 reserve가 필요하다.

3. **tool runtime startup side effect**  
   dataset 단계에서 tool backend를 실제로 띄우면 startup이 불안정해진다.

4. **actor-side OOM**  
   긴 response + 큰 batch + update_actor backward에서 자주 난다. 이 경우 teacher가 아니라 actor mini-batch budget 문제로 보는 게 맞다.

## 무엇을 먼저 의심할 것인가

문제가 생기면 아래 순서로 본다.

1. config/context 문제인가  
   - `teacher_system_prompt_path`
   - teacher budget
   - validation `n`

2. teacher alignment 문제인가  
   - `teacher_mass_max`
   - `loss_max`

3. rollout 자체가 너무 긴가  
   - `done=budget_exhausted`
   - `avg_tok/chunk`
   - `resp_len`

4. actor update memory 문제인가  
   - traceback이 `update_actor` / `loss.backward()`인지 확인

## 현재 문서 이후 읽기

구현 의도와 시스템 최적화를 더 자세히 보려면 바로 다음 문서로 간다.

- [`AGENTS/imp_detail.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/imp_detail.md)
