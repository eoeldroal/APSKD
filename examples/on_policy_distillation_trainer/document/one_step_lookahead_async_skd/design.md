# One-Step Lookahead Async SKD Design

## 0. Purpose

이 문서는 현재 `on_policy_distillation_trainer` 축에서 논의한
**1-step lookahead asynchronous speculative knowledge distillation** 설계를 정리한다.

목표는 다음 네 가지다.

1. tool-aware SKD에서 long trajectory 때문에 생기는 GPU idle을 줄인다.
2. direct SKD distillation의 의미를 최대한 보존한다.
3. stale sample correction 없이도 수학적으로 제어 가능한 범위를 명확히 한다.
4. 실제 구현에서 필요한 state, pseudocode, state machine, 로그 검증 항목을 놓치지 않는다.

이 문서의 설계는 아래 조건을 전제로 한다.

- Student rollout은 `skd_agent` 기반이다.
- Teacher는 rollout 시점에만 사용한다.
- Trainer는 teacher를 다시 호출하지 않는다.
- Distillation은 우선 `use_policy_gradient=False`인 direct supervised KD 경로를 대상으로 한다.
- RL policy-gradient correction과 KD correction은 분리해서 본다.
- 첫 구현에서는 exact IS correction을 쓰지 않는다.
- 첫 구현에서는 `partial_rollout=True`식 arbitrary interruption이 아니라, 완전히 commit된 unit boundary에서만 interruption을 허용한다.

### 0.1 Current Position After Code Review

`recipe/gkd`의 one-step-off는 이미 batch pipeline overlap을 구현한다. 즉 batch `k`로 actor update를 하는 동안 batch `k+1` 전체 rollout을 시작한다. 이 구조는 actor와 rollout worker를 분리하고 weight sync를 수행한다는 점에서 참고할 수 있지만, 이 문서의 목표와는 다르다.

이 문서의 목표는 **intra-step tail filling**이다. Step `k` rollout 내부에서 long trajectory가 tail을 만들 때, 이미 자기 몫을 끝낸 student-teacher pair가 제한된 수의 step `k+1` sample을 미리 생성한다. 따라서 핵심은 batch-level pipeline이 아니라 sample-level scheduling, bounded lookahead, handler-return export predicate다.

현재 논의에서 채택한 구현 방향은 다음과 같다.

- Student trainer instance와 student rollout engine instance를 분리해 고정적으로 띄운다.
- Teacher rollout engine도 고정적으로 띄운다.
- Trainer update 후 student rollout engine에만 weight sync를 수행한다.
- Teacher는 고정 모델이므로 sync하지 않는다.
- Rollout sample에는 생성 시점의 student version을 기록한다.
- 학습 시에는 항상 trainer의 현재 student forward로 KD loss를 계산한다.
- 따라서 stale rollout logit을 쓰는 문제는 없다.
- 하지만 stale policy가 방문한 context/trajectory distribution shift는 남는다.
- 이 residual shift는 version lag, lookahead budget, old-prefix cap, tool macro-step atomicity로 제한하고 로그로 계측한다.

Tool 사용 시에는 용어를 분리해야 한다. `skd_chunk_size`로 자르는 SKD chunk는 student가 생성한 assistant token에만 적용된다. Tool result는 student chunk 안에 있지 않다. 다만 scheduler와 staleness accounting 관점에서는 assistant tool-call 생성부터 tool result append와 dummy teacher row 정렬까지를 하나의 **tool macro-step**으로 취급한다.

### 0.2 Updated Implementation Direction

현재 구현 방향은 trainer 전체를 새로 만드는 것이 아니라, 기존 verl 경계를 최대한 보존한다.

```text
RayPPOTrainer
  -> DataProto
  -> AgentLoopManager.generate_sequences()
  -> DataProto
  -> 기존 update path
```

초기 async SKD path는 custom manager로 들어간다.

```text
actor_rollout_ref.rollout.agent.agent_loop_manager_class
  = verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager
```

MVP constraint:

```text
actor_rollout_ref.rollout.n == 1
```

이 값은 mode entry에서 한 번 검증하는 구조적 invariant다. Lookahead admission 시점마다 다시 평가하는 runtime condition이 아니다.

`sample_async` mode는 lookahead 없이 base batch 전체를 sample 단위 remote task로 upfront submit한다. 이는 GPU/server request concurrency를 낮추기 위한 것이 아니다. 기존 `generate_sequences(chunk)` 내부의 `asyncio.gather()`와 같은 수준의 동시 요청을 유지하면서, manager가 sample별 completion event를 볼 수 있게 하기 위한 것이다.

```text
worker.generate_sequences(24 samples):
  worker 내부에서 24 requests 동시 실행
  manager는 24개 전체 완료 후에만 알 수 있음

worker.generate_sequence_single(sample) x 24:
  Ray async actor에서 24 async calls 동시 실행
  manager는 sample 1개 완료 시점부터 알 수 있음
```

이 single-sample primitive는 base `AgentLoopWorker`가 아니라 `AsyncSkdAgentLoopWorker(AgentLoopWorker)`에 둔다. `AgentLoopWorker`는 기존 generic batch path를 유지하고, SKD async 전용 primitive는 별도 worker subclass에 격리한다.

`AsyncSkdAgentLoopWorker` exposes three execution primitives. `generate_sequence_single()` fully completes a single fresh base sample and returns `DataProto`. `generate_skd_until_boundary()` advances a fresh or resumed SKD sample until `TERMINATED` or the next handler-return export boundary and returns `AsyncSkdSample`. `generate_skd_from_partial_to_completion()` resumes a carry-over `SkdPartialState` as current-step work and runs it to terminal completion.

이 세 primitive의 근본적인 차이는 **약속의 유무**다.

`generate_skd_until_boundary()`가 처리하는 lookahead sample은 아직 current batch에 포함되지 않은 투기적(speculative) 실행이다. step barrier 전에 완료되면 promoted, 완료되지 못하면 pause → carryover로 넘어간다. 약속이 없으므로 pause가 허용된다.

`generate_skd_from_partial_to_completion()`이 처리하는 carry-over sample은 `next_current_batch()`가 반환한 순간 current batch에 편입된다. Trainer는 이 step에서 B개를 학습한다는 전제로 동작하므로, 편입된 sample이 또 pause되면 실제 학습 샘플 수가 B보다 적어지고 배치 예산 불변식이 깨진다. 따라서 current batch에 편입된 carry-over는 **이번 step에서 반드시 완료하거나 drop**한다. `version_lag <= 1` hard cap은 이 철학의 기술적 결과다. carry-over가 step k+1에서 또 pause되면 step k+2에서 `version_lag = 2`가 되어 제약을 위반한다.

Lookahead는 이 sample-level completion visibility 위에 붙인다. Base batch가 끝나면 `drain_requested=True`가 되고, active lookahead task는 강제 cancel하지 않는다. 대신 `skd_agent_loop.py`가 제공하는 cooperative export predicate를 사용한다.

```text
_handle_generating_state(..., stop_after_skd_chunk=True)
_run_until_exportable_boundary(...)
```

이 경로는 handler 내부 중간 상태에서 snapshot을 만들지 않는다. Export 후보는 handler가 반환한 뒤에만 생긴다. Partial carry-over 가능 여부는 label enum이 아니라 다음 실질 조건으로 판단한다.

```text
next_state == GENERATING
teacher row alignment is valid
Qwen/Hermes tool boundary is exportable
```

Qwen/Hermes tool boundary rule:

```text
EOS 없음 + open tool-call prefix:
  export 가능, tool 실행 금지

EOS 없음 + closed </tool_call> block:
  export 금지, 다음 chunk에서 EOS까지 진행

EOS 있음 + valid tool call:
  PROCESSING_TOOLS로 진행하고 tool result append 후 export 가능

EOS 있음 + no valid tool call:
  interaction 또는 termination으로 진행
```

## 1. High-Level Idea

현재 동기식 SKD mixed rollout에서는 batch 내 long trajectory 하나가 전체 step의 tail latency를 결정한다.

예를 들어 base batch size가 `B=128`이고, student-teacher pair가 4개라고 하자. 각 pair는 대략 32개 sample을 맡는다. 특정 pair가 매우 긴 trajectory, 이하 `LT`, 를 처리하고 있으면 나머지 pair는 자기 quota를 끝낸 뒤 idle 상태가 된다.

제안하는 방식은 idle pair가 다음 step의 sample 일부를 미리 가져와 old student version으로 rollout하는 것이다. 단, 이 작업은 무제한으로 하지 않는다.

핵심 규칙은 다음과 같다.

- Step `k`의 base batch `B`는 항상 정상적으로 완료한다.
- Idle pair는 step `k+1`의 sample을 제한된 수만큼 lookahead로 시작한다.
- Lookahead sample이 step `k` barrier 전에 terminal까지 끝나면 current step에 승격한다.
- Lookahead sample이 terminal까지 끝나지 않으면 다음 handler-return export boundary까지만 진행한 뒤 pause한다.
- Paused sample은 step `k+1`에서 current version으로 resume한다.
- Step `k+1`에서도 쓰지 못한 carry-over sample은 drop한다.

즉 이 방법은 open-ended fully async training이 아니라, **bounded one-step lookahead**다.

## 2. Terms

### 2.1 Base Batch

Step `k`에서 원래 학습하려던 기본 rollout batch다. 크기를 `B`라고 둔다.

### 2.2 Student-Teacher Pair

SKD rollout에서 한 student rollout server와 한 teacher verification server가 묶여 sample을 처리하는 실행 단위다. Pair 수를 `N_pair`라고 둔다.

### 2.3 LT

Long trajectory다. Tool call, repeated rejection, long reasoning, max chunk 근접 등으로 인해 step tail latency를 만드는 trajectory를 뜻한다.

### 2.4 Lookahead Sample

Step `k` 중 idle 자원을 이용해 step `k+1`에서 사용할 예정이던 prompt를 미리 rollout한 sample이다.

### 2.5 Promoted Sample

Lookahead sample 중 step `k` barrier 전에 terminal까지 완료된 sample이다. 이 sample은 age 0이므로 current step에 포함한다.

### 2.6 Paused Carry-Over Sample

Lookahead sample 중 terminal까지 끝나지 않았지만 export predicate를 만족하는 handler-return boundary에서 멈춘 sample이다. Step `k+1`에서 current version으로 이어서 생성한다.

### 2.7 Export Predicate

Partial carry-over는 label enum이 아니라 serialized state의 실제 일관성으로 판단한다.

Canonical state:

```text
prompt_ids
response_ids
response_mask
teacher_prompt_ids
teacher_ids_list
teacher_logprobs_list
messages
assistant_turns
user_turns
tool_rewards
turn_scores
committed_gen_chunks
committed_env_units
committed_prefix_tokens
```

Export predicate:

```text
1. export check runs only after an outer handler returns.
2. next_state must be GENERATING.
3. response_mask length must match teacher row lengths.
4. Qwen/Hermes tool-call boundary must be exportable.
```

The implementation should not use `SkdCommittedUnit`, `RESUMABLE_COMMITTED_UNITS`, `last_committed_unit`, or `skd_last_committed_unit` as correctness gates.

### 2.8 Tool Macro-Step

Tool-aware SKD에서는 두 종류의 chunk를 구분한다.

```text
SKD generation chunk:
student가 생성한 assistant token 블록.
teacher verification과 first-rejection correction의 대상.
response_mask=1.

Tool macro-step:
assistant tool-call 생성
+ tool parser가 tool call 추출
+ tool execution
+ tool result message 생성
+ tool result token append
+ response_mask=0 append
+ dummy teacher row append
+ teacher_prompt_ids append
```

Tool result token은 KD loss 대상이 아니다. 그러나 async pause/resume과 stale-prefix budget 관점에서는 tool macro-step 전체가 atomic environment transition이다.

### 2.9 Policy Version

Trainer가 update를 한 번 끝낼 때마다 student parameter version을 하나 증가시킨다.

```text
rollout_version = sample 생성 또는 prefix 생성에 사용된 student rollout snapshot version
train_version = trainer가 해당 sample을 consume할 때의 current version
version_lag = train_version - rollout_version
```

첫 구현에서는 다음을 hard constraint로 둔다.

```text
version_lag <= 1
version_span <= 1
```

`version_span`은 하나의 trajectory 안에서 관측된 `max_rollout_version - min_rollout_version`이다. Weight sync가 export boundary에서만 적용되더라도, old prefix와 new suffix가 섞인 resumed sample은 `version_span=1`이 될 수 있다.

## 3. Design Hyperparameters

### 3.1 Lookahead Admission Budget

Step `k`에서 새로 시작할 수 있는 lookahead sample 총수다.

권장값:

```text
L_prefetch = min(floor(B / N_pair), floor(0.25 * B))
```

예시:

```text
B = 128
N_pair = 4
L_prefetch = min(32, 32) = 32
```

중요한 점은 이 값이 **step 중간에 재충전되지 않는 총 admission budget**이라는 것이다.

즉 lookahead sample이 빨리 끝나 current step에 promote되더라도, 같은 step에서 추가 lookahead budget이 다시 열리지 않는다. 이렇게 해야 runtime-conditioned fast sample이 current step을 과도하게 지배하지 않는다.

### 3.2 Stale Generation Chunk Cap

Paused carry-over sample 하나가 old version으로 진행할 수 있는 committed SKD generation chunk 수의 상한이다.

권장값:

```text
max_old_gen_chunks = 2
```

이 값은 수학적 stale mismatch를 직접 제어하는 주 cap이다.

### 3.3 Old Environment Unit Cap

Old version prefix 위에서 완료할 수 있는 full tool/interact unit 수의 상한이다.

권장값:

```text
max_old_env_units = 1
```

이 값은 주로 atomicity, context blow-up, resume complexity를 제어한다.

### 3.4 Old Prefix Token Cap

Old version으로 진행된 prefix가 전체 response budget을 과도하게 잠식하지 않도록 하는 보조 guard다.

권장값:

```text
old_prefix_token_cap = floor(max_response_length / 8)
```

예시:

```text
max_response_length = 8192  -> old_prefix_token_cap = 1024
max_response_length = 12288 -> old_prefix_token_cap = 1536
```

이 값은 수학적 primary cap이 아니라 시스템 safety guard다. 주된 stale 제어는 `max_old_gen_chunks`가 담당한다.

## 4. Mathematical Formulation

### 4.1 Direct SKD Objective

현재 논의의 중심은 `use_policy_gradient=False`인 direct supervised SKD distillation이다.

Step `k`의 student parameter를 `theta_k`라고 하자. Student version `theta`가 만드는 handler-return trajectory 분포를 `P_theta`라고 둔다.

Trajectory `z`에 대해 direct SKD loss 합을 다음처럼 둔다.

```text
F_theta(z) = sum_{t in A(z)} ell_SKD(theta; h_t, q_T(. | h_t))
```

여기서:

- `A(z)`는 `response_mask=1`인 assistant-supervised 위치 집합이다.
- `h_t`는 해당 token의 prefix/history다.
- `q_T(. | h_t)`는 teacher distribution이다.
- Teacher는 고정 모델이다.

그러면 동기식 on-policy direct SKD 목적함수는:

```text
J_k(theta_k) = E_{z ~ P_{theta_k}} [F_{theta_k}(z)]
```

현재 구현에는 distillation loss clamp와 max response budget이 있으므로, 어떤 상수 `C_dist`가 있어:

```text
0 <= F_theta(z) <= C_dist
```

보수적으로는:

```text
C_dist <= max_response_length * loss_max_clamp
```

라고 둘 수 있다.

### 4.1.1 What Current-Policy Forward Corrects

현재 구현에서 rollout은 inference engine이 수행하지만, 학습 loss는 trainer의 현재 student parameter로 forward해서 계산한다.

즉 stale rollout sample `z`가 있더라도 KD loss의 student side는 다음처럼 계산된다.

```text
ell_SKD(theta_train; h_t^z, q_T(. | h_t^z))
```

여기서 `theta_train`은 trainer가 현재 보유한 최신 parameter다. 따라서 stale rollout engine이 생성 당시 계산한 student logprob가 gradient graph에 직접 들어가지는 않는다.

이 구조가 완화하는 것은 다음이다.

- stale rollout logprob로 gradient를 구성하는 문제
- inference engine이 gradient graph를 갖지 않는 문제
- teacher target과 비교하는 student distribution이 old snapshot에 묶이는 문제

하지만 다음은 보정하지 못한다.

- stale policy가 어떤 context `h_t`를 방문했는가
- stale policy가 어떤 tool call path를 선택했는가
- current policy라면 같은 first-rejection pattern이 나왔을지
- current policy라면 해당 trajectory 자체를 높은 확률로 생성했을지

따라서 residual off-policy bias는 logit-side bias가 아니라 **visited context distribution shift**다. 이 문서의 staleness control은 이 distribution shift를 제한하기 위한 장치다.

이를 식으로 쓰면, 이상적인 on-policy 목적은:

```text
J_on(theta_k) = E_{z ~ P_{theta_k}} [F_{theta_k}(z)]
```

stale rollout을 포함한 목적은:

```text
J_async(theta_k) = E_{z ~ P_{theta_{k-d}} or hybrid} [F_{theta_k}(z)]
```

이다. 차이는:

```text
|J_async(theta_k) - J_on(theta_k)|
<= C_dist * TV(P_{theta_{k-d}} or hybrid, P_{theta_k})
```

처럼 해석할 수 있다. 이 bound는 tight한 성능 보장이 아니라, 남는 오차가 trajectory/context distribution shift임을 명시하는 용도다.

실제 구현에서는 이 TV를 직접 계산하지 않고, token-level logprob drift를 proxy로 측정한다.

```text
drift_logp =
mean_{t in assistant positions}
| log pi_current(a_t | h_t) - log pi_rollout(a_t | h_t) |
```

### 4.1.2 Teacher Top-K Tube Constraint

SKD의 teacher verification은 stale rollout 문제를 없애지는 않지만, stale student proposal이 committed trajectory에 미치는 자유도를 제한한다.

고정된 context `h`에서 teacher distribution을 `q_T(. | h)`라고 두고, teacher top-k support를:

```text
S_T^K(h) = TopK(q_T(. | h))
```

라고 하자. Student proposal token을:

```text
y ~ pi_theta(. | h)
```

라고 할 때, SKD의 단일 token commit rule은 단순화해서 다음처럼 쓸 수 있다.

```text
if y in S_T^K(h):
    a = y
else:
    a = argmax q_T(. | h)
```

따라서 committed token `a`는 항상:

```text
a in S_T^K(h)
```

를 만족한다. Chunk 단위에서도 같은 성질이 성립한다. Student가 제안한 chunk에서 teacher top-k 밖 token이 처음 등장하면, SKD는 그 위치에서 suffix를 버리고 해당 token을 teacher top-1로 교체한다. 따라서 최종 committed assistant token은 teacher top-k support 밖으로 나가지 않는다.

```text
P_SKD(a_t notin S_T^K(h_t) | h_t) = 0
```

이 성질을 이 문서에서는 **teacher top-k tube constraint**라고 부른다. 즉 committed SKD trajectory는 각 visited context에서 teacher top-k tube 안에 놓인다.

Raw student proposal kernel을 `G_theta(. | h)`라고 하고, teacher correction을 거친 committed SKD kernel을:

```text
\tilde G_theta(. | h) = C_T # G_theta(. | h)
```

라고 두자. 여기서 `C_T`는 teacher top-k correction map이고, `#`는 pushforward를 뜻한다.

고정된 context `h`에서 `C_T`는 teacher에 의해 정해지는 post-processing이다. Total variation distance는 deterministic post-processing에서 증가하지 않으므로:

```text
TV(\tilde G_{theta_old}(. | h), \tilde G_{theta_new}(. | h))
<=
TV(G_{theta_old}(. | h), G_{theta_new}(. | h))
```

가 성립한다.

해석하면, SKD teacher correction을 통과한 뒤의 old/current committed kernel 차이는 raw student proposal kernel 차이보다 커질 수 없다. 특히 teacher top-k 밖 tail token들에서 old student와 current student가 크게 다르더라도, 그 차이는 committed path에 token identity 그대로 전달되지 않고 teacher top-1 replacement로 collapse된다.

따라서 unfinished carry-over 분석에서 raw policy drift 대신 corrected SKD drift를 사용할 수 있다.

```text
delta_k^raw =
sup_h TV(G_{theta_k}(. | h), G_{theta_{k+1}}(. | h))

delta_k^skd =
sup_h TV(\tilde G_{theta_k}(. | h), \tilde G_{theta_{k+1}}(. | h))
```

그러면:

```text
delta_k^skd <= delta_k^raw
```

이고, carry-over mismatch bound는 더 정확히:

```text
TV(P_{k -> k+1}^{(m)}, P_{theta_{k+1}})
<= m * delta_k^skd
<= m * delta_k^raw
```

처럼 쓸 수 있다.

이 논증의 의미는 제한적이지만 중요하다.

- SKD는 stale rollout을 unbiased on-policy sample로 만들지 않는다.
- 하지만 stale student proposal의 token-level 자유도를 teacher top-k support 안으로 제한한다.
- 고정 context에서 teacher correction은 old/current proposal mismatch를 증가시키지 않는다.
- 따라서 staleness의 proposal-side mismatch는 student-only stale rollout보다 완화된다.

남는 문제도 명확하다.

- Teacher top-k 안에서 old/current student가 서로 다른 token을 고르는 branch drift는 남는다.
- 어떤 context와 tool state에 도달했는지에 대한 distribution shift는 사라지지 않는다.
- Tool call이 teacher top-k token들로 구성되어도, tool execution으로 생긴 environment transition은 별도 atomic unit으로 관리해야 한다.
- `skd_verify_top_k`가 클수록 tube가 넓어져 correction에 의한 contraction 효과는 약해질 수 있다.

### 4.2 Finished Lookahead Promotion

Step `k`에서 lookahead로 시작해 terminal까지 끝난 sample은 전부 `theta_k`로 생성된 sample이다. 따라서 next step으로 넘기면 age 1 stale이 되고, current step에 포함하면 age 0이다.

Promotion한 lookahead sample 분포를 `Q_k`라고 하고, current step에 `Delta_k`개가 promote되었다고 하자. Current step의 실제 training distribution은:

```text
mu_k =
  (B / (B + Delta_k)) * P_{theta_k}
  + (Delta_k / (B + Delta_k)) * Q_k
```

따라서 objective deviation은:

```text
| E_{mu_k}[F_{theta_k}] - E_{P_{theta_k}}[F_{theta_k}] |
<= C_dist * (Delta_k / (B + Delta_k)) * TV(Q_k, P_{theta_k})
```

Worst case에서 `TV <= 1`이므로:

```text
| E_{mu_k}[F] - E_{P_{theta_k}}[F] |
<= C_dist * (L_prefetch / (B + L_prefetch))
```

즉 promotion으로 인한 step-local skew는 admission budget 비율로 상한이 생긴다.

중요한 해석:

- Promotion은 stale mismatch를 만들지 않는다.
- 대신 runtime-conditioned fast subset skew를 만든다.
- 이 skew는 `L_prefetch`를 작게 두면 bounded 된다.

### 4.3 Why Finished Samples Should Be Promoted

Finished lookahead sample을 current step에 쓰면:

```text
E_{z ~ Q_k}[F_{theta_k}(z)]
```

다음 step으로 미루면:

```text
E_{z ~ Q_k}[F_{theta_{k+1}}(z)]
```

즉 defer는 이미 age 0인 sample을 age 1로 만들어 stale mismatch를 새로 도입한다.

따라서 finished lookahead sample은 다음 원칙을 따른다.

```text
cap how many lookahead samples are started;
promote every finished lookahead sample.
```

### 4.4 Unfinished Carry-Over Resume

Student generation safe transition을 `G_theta`라고 두자. 한 `G_theta`는 committed SKD generation chunk 하나를 의미한다.

Tool 또는 interaction full step을 environment operator `E`라고 두자. `E`는 student parameter를 직접 포함하지 않는다고 가정한다.

Paused carry-over sample이 old version `theta_k`로 generation chunk를 `m`개 끝내고, next step에서 `theta_{k+1}`로 resume되면 hybrid distribution은 대략:

```text
P_{k -> k+1}^{(m)}
= rho * (prod_{i=1}^{m} G_{theta_k} E_i)
        (prod_{i=m+1}^{H} G_{theta_{k+1}} E_i)
```

Next-step on-policy distribution은:

```text
P_{theta_{k+1}}
= rho * prod_{i=1}^{H} G_{theta_{k+1}} E_i
```

Consecutive version 사이 raw generation kernel drift를:

```text
delta_k^raw = sup_s TV(G_{theta_k}(. | s), G_{theta_{k+1}}(. | s))
```

라고 두자. SKD에서는 실제 committed token이 teacher correction operator를 통과하므로, 더 직접적인 drift는:

```text
delta_k^skd = sup_s TV(\tilde G_{theta_k}(. | s), \tilde G_{theta_{k+1}}(. | s))
```

이다. 4.1.2의 teacher top-k tube constraint에 의해:

```text
delta_k^skd <= delta_k^raw
```

가 성립한다. 따라서 hybrid argument로:

```text
TV(P_{k -> k+1}^{(m)}, P_{theta_{k+1}}) <= m * delta_k^skd <= m * delta_k^raw
```

따라서 bounded loss에 대해:

```text
| E_{P_{k -> k+1}^{(m)}}[F_{theta_{k+1}}]
  - E_{P_{theta_{k+1}}}[F_{theta_{k+1}}] |
<= C_dist * m * delta_k^skd
<= C_dist * m * delta_k^raw
```

`max_old_gen_chunks = M_gen`을 두면:

```text
objective deviation <= C_dist * M_gen * delta_k^skd
```

이 식이 unfinished carry-over의 핵심 bound다.

즉 `age <= 1`만으로는 충분하지 않고, old generation chunk 수 cap이 필요하다.

중요한 해석은 다음이다. SKD teacher correction은 carry-over를 unbiased하게 만들지는 않지만, stale/current proposal drift가 committed token distribution에 전달되는 정도를 raw student-only rollout보다 줄일 수 있다. 따라서 bound의 핵심 항은 raw drift `delta_k^raw`가 아니라 corrected SKD drift `delta_k^skd`로 보는 것이 더 적절하다.

### 4.5 Tool Macro-Step Atomicity

Tool step은 중간에서 끊으면 안 된다. 현재 `skd_agent_loop.py`는 tool result를 SKD generation chunk 안에 넣지 않는다. Assistant가 생성한 tool-call token은 SKD generation chunk의 일부일 수 있지만, 실제 tool result token은 `PROCESSING_TOOLS` state에서 별도 span으로 append된다.

Tool result span은 다음 의미를 갖는다.

```text
response_mask = 0
teacher row = dummy zero row
student prompt stream에는 append
teacher prompt stream에도 append
KD loss에서는 mask로 제외
```

따라서 scheduler 관점에서는 다음 전체를 하나의 `TOOL_MACRO_STEP`으로 취급한다.

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

Manager-level export는 handler 내부에 끼어들지 않는다. 따라서 `teacher verification 전`, `assistant tokens append 후 teacher rows append 전`, `tool result tokens append 후 dummy teacher rows append 전` 같은 상태는 정상 cooperative export 후보가 아니다.

Tool-aware export의 실제 위험은 assistant-side tool-call text의 형식이다.

```text
open tool-call prefix:
  아직 tool execution 상태가 아니다.
  assistant prefix로 carry-over 가능하다.

closed </tool_call> block without EOS:
  Hermes parser가 parse 가능한 block을 볼 수 있어도 Qwen/Hermes assistant turn은 닫히지 않았다.
  export하지 말고 generation을 더 진행한다.

EOS + valid tool call:
  tool execution으로 진행한다.
  tool result append, response_mask=0 span, dummy teacher rows, teacher prompt stream update까지 끝난 뒤 export 후보가 된다.
```

따라서 tool result는 all-or-none environment macro-step이다. 다만 export safety는 tool-result label이 아니라 handler-return boundary, alignment, EOS-gated parser state로 판정한다.

### 4.6 Two-Step Sample Accounting

Step `k`에서 promoted sample 수를 `Delta_k`, paused carry-over sample 수를 `R_k`라고 하자.

Step `k`의 학습 sample 수는:

```text
B + Delta_k
```

Step `k+1`에서는 carry-over `R_k`를 먼저 resume하고 fresh quota를:

```text
B - R_k
```

로 둔다. Promoted `Delta_k`는 이미 step `k`에서 학습에 들어간 reserved future sample이다. 따라서 step `k+1` fresh quota에서 다시 차감하지 않는다. 대신 source/reservation ledger가 해당 sample을 다시 emit하지 않도록 보장한다.

Step `k+1`의 학습 sample 수는:

```text
(B - R_k) + R_k = B
```

두 step 합은:

```text
(B + Delta_k) + B = 2B + Delta_k
```

즉 finished lookahead promotion은 실제 train sample throughput을 늘린다. Sample identity accounting은 source ledger로 보존한다. 같은 prompt가 중복 학습되면 안 되며, promoted prompt는 다음 step에서 다시 fresh로 나오면 안 된다.

예시:

```text
B = 96
Delta_k = 18
R_k = 30

step k train batch = 96 + 18 = 114
step k+1 current work = resume 30 + fresh 66 = 96
```

### 4.7 Dataset Sampling And Reservation Layer

Two-step accounting을 실제로 지키려면 trainer loop의 dataloader 소비 방식도 한 단계 더 세밀해야 한다.

현재 `RayPPOTrainer`의 기본 구조는 step마다 `StatefulDataLoader`에서 batch 하나를 통째로 소비한다.

```python
for batch_dict in self.train_dataloader:
    batch = DataProto.from_single_dict(batch_dict)
```

Bounded lookahead에서는 이 구조 위에 sample-level reservation 계층이 필요하다. 이유는 step `k` 중에 step `k+1` sample 일부를 미리 reserve해야 하고, 그중 unfinished carry-over sample만큼 step `k+1`의 fresh sample 소비량을 줄여야 하기 때문이다. Finished promoted sample은 next-step fresh quota에서 차감하지 않는다. 대신 source/reservation ledger가 해당 sample을 다시 emit하지 않도록 중복 방지 대상으로 기록한다.

첫 구현에서는 PyTorch sampler 자체를 바꾸지 않는다. 또한 전체 dataset이나 dataloader output을 `list()`로 materialize하지 않는다. 초기 list-materialized accounting prototype은 현재 quota 규칙과 실제 dataloader contract에 맞지 않아 제거했고, production lookahead path는 dataloader-aware source로 구현한다.

```text
StatefulDataLoader / existing sampler
  -> AsyncSkdDataSource
      -> fresh sample buffer
      -> lookahead reservation buffer
      -> carryover buffer
```

역할 분리는 다음과 같다.

```text
Sampler / StatefulDataLoader:
  dataset order, shuffle, checkpoint/resume state를 담당한다.

AsyncSkdDataSource:
  step별 fresh quota, lookahead reservation, promoted/carryover accounting을 담당한다.
```

이렇게 해야 기존 sampler의 checkpoint/resume, curriculum sampler update, distributed shuffle semantics를 크게 건드리지 않고 async SKD 전용 sample accounting을 추가할 수 있다. MVP에서는 curriculum sampler가 켜져 있으면 lookahead를 비활성화하고, `rollout.n != 1`이면 lookahead path를 거부한다.

현재 구현된 MVP `AsyncSkdDataSource`는 `StatefulDataLoader` 자체를 바꾸지 않는다. Dataloader iterator가 반환한 collated `batch_dict`를 `DataProto.from_single_dict(batch_dict)`로 변환한 뒤, 내부 fresh buffer에서 `DataProto[pos:pos+1]` 형태로 single-sample `DataProto`를 공급한다. Source가 부여한 `uid`는 lookahead reservation, promoted ledger, carry-over accounting의 sample identity로 사용한다.

MVP `state_dict()`는 source-local state만 저장한다.

```text
fresh_buffer
fresh_cursor
carryover_partials
carryover_input_batches
reserved_input_batches
trained_reserved_sample_ids
```

`StatefulDataLoader.state_dict()`와 source state를 같은 trainer checkpoint에 묶는 작업은 trainer integration 단계에서 수행한다.

`AsyncSkdDataSource`와 trainer integration이 지켜야 할 핵심 불변식은 다음이다.

```text
1. 동일 sample은 active window 안에서 중복 소비되지 않는다.
2. lookahead는 current_step + 1 sample까지만 reserve한다.
3. step 중 lookahead budget은 refill하지 않는다.
4. promoted sample은 source/reservation ledger에서 중복 방지 대상으로 기록한다.
5. carry-over sample은 next-step fresh quota에서 차감한다.
6. source-local state는 `StatefulDataLoader` state와 함께 checkpoint 가능해야 한다.
```

따라서 mixed batch 구성은 단순히 tensor를 concat하는 문제가 아니다. 그 전에 “어떤 fresh sample을 얼마나 새로 꺼낼지”를 결정하는 reservation layer가 필요하다.

## 5. What Is Mathematically Guaranteed

보증 가능한 것은 다음이다.

- Finished lookahead promotion이 만드는 current-step distribution skew는 `L_prefetch / (B + L_prefetch)`로 상한이 있다.
- Unfinished carry-over resume의 mismatch는 `max_old_gen_chunks * delta_k`로 상한이 있다.
- `age > 1` sample을 drop하면 sample age는 hard cap 된다.
- Tool result를 atomic unit으로 묶으면 mid-tool state corruption은 구조적으로 방지된다.
- `fresh_quota_{k+1} = B - R_k`로 두면 step `k+1` current work size는 `B`로 보존된다. Promoted `Delta_k`는 next-step fresh quota에서 차감하지 않고, source/reservation ledger에서 duplicate emission만 막는다.
- `old_prefix_token_cap = max_response_length / 8`은 old prefix가 전체 response budget을 과도하게 잠식하는 것을 막는다.
- 학습 loss의 student side는 항상 current trainer parameter로 forward되므로, stale rollout logprob가 gradient에 직접 들어가지 않는다.
- SKD teacher correction은 committed assistant token을 teacher top-k tube 안으로 제한한다.
- 고정 context에서 teacher correction은 raw old/current student proposal TV distance를 증가시키지 않는다.

## 6. What Is Not Mathematically Guaranteed

보증되지 않는 것은 다음이다.

- `delta_k` 자체가 항상 작다는 보장은 없다.
- Aggressive optimizer update, 큰 learning rate, 많은 local update, 불안정한 policy drift에서는 bound가 loose해진다.
- Promotion subset은 runtime-conditioned fast subset이라 current step 분포가 short/easy trajectory로 치우칠 수 있다.
- Resumed carry-over는 exact on-policy trajectory가 아니라 mixed-version hybrid trajectory다.
- Dynamic batch size는 optimizer noise scale을 바꿀 수 있다.
- Tool execution이 외부 상태 또는 nondeterminism에 강하게 의존하면 environment operator 안정성 가정이 약해진다.
- Current-policy forward는 stale trajectory의 visited context distribution shift를 제거하지 못한다.
- Current policy가 해당 trajectory를 낮은 확률로 생성할 가능성은 수학적으로 배제할 수 없다.
- Teacher top-k tube constraint는 token-level support constraint이지, visited context나 tool state distribution을 보정하지 않는다.
- Teacher top-k 안에서 서로 다른 branch를 고르는 drift는 여전히 남는다.
- Benchmark score non-degradation은 수학만으로 보장되지 않는다.

## 7. How To Validate The Unstable Parts

이론적으로 완전히 보장되지 않는 부분은 로그와 benchmark로 보여야 한다.

### 7.1 Required Training Logs

반드시 source-aware metric을 남긴다.

Source type:

- `base_current`
- `lookahead_promoted`
- `lookahead_resumed`
- `dropped_expired`

필수 로그:

- `lookahead/promoted_count`
- `lookahead/promoted_ratio`
- `lookahead/carryover_count`
- `lookahead/carryover_ratio`
- `lookahead/drop_count`
- `lookahead/drop_ratio`
- `lookahead/old_gen_chunks_mean`
- `lookahead/old_gen_chunks_p95`
- `lookahead/old_gen_chunks_max`
- `lookahead/old_env_units_hist`
- `lookahead/old_prefix_tokens_mean`
- `lookahead/old_prefix_tokens_p95`
- `lookahead/old_prefix_tokens_max`
- `lookahead/admission_budget_used`
- `lookahead/idle_tokens_or_time_filled`
- `lookahead/drain_wait_ms`
- `lookahead/resume_count`
- `lookahead/tool_atomic_pause_count`
- `lookahead/version_lag_mean`
- `lookahead/version_lag_max`
- `lookahead/version_span_max`
- `lookahead/stale_sample_ratio`
- `lookahead/stale_token_ratio`
- `lookahead/logprob_delta_mean`
- `lookahead/logprob_delta_p95`
- `skd/rejection_rate_by_source`
- `skd/rejection_rate_by_age`
- `skd/committed_teacher_rank_mean`
- `skd/committed_teacher_rank_p95`
- `skd/outside_teacher_topk_committed_rate`

Source별 distillation metric:

- `distillation/loss`
- `distillation/loss_max`
- `distillation/teacher_mass`
- `distillation/student_mass`

Source별 rollout dynamic:

- response length
- number of turns
- tool call count
- SKD accept rate
- reject count
- chunks per sample
- max-chunks ratio

### 7.2 Optional Drift Probe

가능하면 resumed sample에 대해 current policy와 rollout policy의 logprob drift를 잰다.

```text
drift_logp =
mean_{t in assistant positions}
| log pi_current(a_t | h_t) - log pi_birth(a_t | h_t) |
```

이 값은 이론의 `delta_k`를 직접 계산하는 것은 아니지만, practical proxy로 쓸 수 있다.

### 7.3 Required Ablations

최소 비교:

- sync SKD baseline
- lookahead + promote only
- lookahead + promote + resume
- lookahead admission budget sweep
- `max_old_gen_chunks` sweep
- `old_prefix_token_ratio` sweep

보고해야 하는 결과:

- throughput
- time per step
- GPU idle ratio
- validation score
- benchmark score
- source-aware loss stability
- stale/resume sample quality

## 8. Pseudocode

### 8.1 Atomic Unit Advancement

```python
def advance_one_atomic_unit(sample, version):
    unit = peek_next_atomic_unit(sample)

    if unit == "PENDING_INIT":
        init_student_and_teacher_prompt_streams(sample)
        sample.agent_state = "GENERATING"
        return sample, False

    if unit == "GEN_CHUNK":
        result = run_exactly_one_skd_chunk(sample, version)
        # student propose -> teacher verify -> first-rejection commit
        # append committed assistant tokens
        # append teacher rows for committed assistant tokens
        sample.old_gen_chunks += 1
        sample.old_prefix_tokens += result.committed_token_count
        if result.terminal:
            sample.status = "DONE"
            return sample, True
        return sample, False

    if unit == "TOOL_STEP":
        result = run_full_tool_step(sample)
        # full tool execution
        # full tool response serialization
        # response_mask += zeros
        # teacher dummy rows += len(tool_response_ids)
        sample.old_env_units += 1
        sample.old_prefix_tokens += result.appended_token_count
        return sample, False

    if unit == "INTERACT_STEP":
        result = run_full_interact_step(sample)
        sample.old_env_units += 1
        sample.old_prefix_tokens += result.appended_token_count
        return sample, False

    if unit == "TERMINATED":
        return sample, True
```

### 8.2 Budget Check

```python
def within_old_budget(sample, next_unit, cfg):
    if next_unit == "GEN_CHUNK":
        return (
            sample.old_gen_chunks + 1 <= cfg.max_old_gen_chunks
            and sample.old_prefix_tokens + cfg.skd_chunk_size <= cfg.old_prefix_token_cap
        )

    if next_unit == "TOOL_STEP":
        ub = estimate_next_tool_step_token_upper_bound(sample)
        return (
            sample.old_env_units + 1 <= cfg.max_old_env_units
            and sample.old_prefix_tokens + ub <= cfg.old_prefix_token_cap
        )

    if next_unit == "INTERACT_STEP":
        ub = estimate_next_interact_step_token_upper_bound(sample)
        return (
            sample.old_env_units + 1 <= cfg.max_old_env_units
            and sample.old_prefix_tokens + ub <= cfg.old_prefix_token_cap
        )

    return True
```

### 8.3 Lookahead Runner

```python
def run_lookahead_until_pause_or_finish(sample, old_version, cfg, drain_flag):
    while True:
        next_unit = peek_next_atomic_unit(sample)

        if next_unit == "TERMINATED":
            sample.status = "READY_PROMOTABLE"
            return snapshot(sample)

        if drain_flag:
            sample.status = "PAUSED_CARRYOVER"
            return snapshot(sample)

        if not within_old_budget(sample, next_unit, cfg):
            sample.status = "PAUSED_CARRYOVER"
            return snapshot(sample)

        sample, done = advance_one_atomic_unit(sample, old_version)

        if done:
            sample.status = "READY_PROMOTABLE"
            return snapshot(sample)
```

### 8.4 Step Scheduler

```python
for step_id in training_steps:
    current_version = get_current_param_version()
    prefetch_started = 0
    drain_flag = False

    base_batch = fetch_base_current_batch(B)
    submit_all_base_samples_upfront(base_batch, current_version)

    while base_or_lookahead_tasks_active():
        result = wait_first_completed()

        if result.kind == "base_completed":
            collect_base(result)
            if prefetch_started < L_prefetch:
                prompt = reserve_future_prompt(step_id + 1)
                submit_lookahead_until_exportable_boundary(
                    prompt=prompt,
                    birth_step=step_id,
                    birth_version=current_version,
                )
                prefetch_started += 1

            if all_base_samples_finished():
                drain_flag = True

        elif result.kind == "lookahead_completed":
            promoted.append(result)

        elif result.kind == "lookahead_partial":
            if drain_flag:
                carryover_next.append(result)
            elif prefetch_started <= L_prefetch and within_old_budget(result):
                resume_lookahead_until_exportable_boundary(result)
            else:
                carryover_next.append(result)

        if drain_flag and no_active_lookahead_tasks():
            break

    train_batch = base_batch + promoted

    train_on(train_batch)
    update_student()

    next_fresh_quota = B - len(carryover_next)
    next_fresh_quota = max(0, next_fresh_quota)

    next_step_buffer = {
        "carryover": carryover_next,
        "fresh_quota": next_fresh_quota,
        "already_trained_reserved": promoted,
    }

    drop_all_samples_with_age_gt_1()
```

### 8.5 Next-Step Assembly

```python
def assemble_next_step_batch(step_id, current_version, buffer):
    carryover = buffer.carryover
    valid_carryover = []

    for sample in carryover:
        age = current_version - sample.birth_version
        if age <= 1:
            valid_carryover.append(sample)
        else:
            mark_dropped(sample)

    resumed = []
    for sample in valid_carryover:
        resumed_sample = resume_from_snapshot(sample, current_version)
        resumed.append(resumed_sample)

    fresh = fetch_fresh_batch(buffer.fresh_quota)
    return resumed + fresh
```

## 9. Implementation Additions

### 9.1 Files To Extend

Core files:

- `verl/experimental/agent_loop/skd_agent_loop.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/trainer/ppo/ray_trainer.py` for batch acquisition, promoted input assembly, source checkpoint hooks, and metric handoff
- `verl/workers/config/distillation.py`

New async SKD files:

- `verl/experimental/async_skd/state.py`
- `verl/experimental/async_skd/worker.py`
- `verl/experimental/async_skd/manager.py`
- `verl/experimental/async_skd/data_source.py`

Reference files to mimic, not blindly copy:

- `verl/experimental/fully_async_policy/fully_async_rollouter.py`
- `verl/experimental/fully_async_policy/fully_async_trainer.py`
- `verl/experimental/fully_async_policy/agent_loop/agent_loop.py`
- `recipe/gkd/megatron/ray_trainer.py`

Experiment files:

- `examples/on_policy_distillation_trainer/run_*fully_async*skd*.sh`
- `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md`
- `examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md`

### 9.1.1 Reuse And Inheritance Decisions

The implementation should reuse existing verl boundaries, but should not inherit from unrelated async trainers.

Already adopted inheritance:

```text
AsyncSkdAgentLoopWorker(AgentLoopWorker)
AsyncSkdAgentLoopManager(AgentLoopManager)
```

This is the correct inheritance boundary. `AgentLoopWorker` already owns agent-loop instantiation, server manager access, tokenizer/processor state, reward loop handles, teacher server handles, trace config, and postprocess logic. `AgentLoopManager` already owns rollout server initialization, worker creation, teacher wake/sleep, `DataProto.concat`, and timing aggregation. Async SKD should keep using these hooks.

Do not add new inheritance from:

```text
FullyAsyncAgentLoopManager
FullyAsyncRollouter
FullyAsyncTrainer
recipe/gkd/megatron/ray_trainer.py trainer classes
```

Reasons:

- `FullyAsyncAgentLoopManager` rejects distillation-enabled use and its partial rollout is abort/resume based, not SKD handler-return export-predicate based.
- `FullyAsyncRollouter` and `FullyAsyncTrainer` assume a producer/consumer queue split. The current bounded lookahead path is manager-local.
- `MessageQueue` drops the oldest sample when full. SKD sample accounting cannot silently drop reserved or carry-over samples.
- GKD trainer implements batch pipeline overlap. This design needs intra-step sample-level tail filling.
- GKD Megatron workers are useful for rollout-only worker and weight sync reference code, but not as parent classes for the agent-loop SKD manager.

Reuse map:

| Remaining work | Reuse mechanism | Do not use |
|---|---|---|
| Promoted dynamic batch assembly | `DataProto.concat`, `DataProto.union`, `AsyncSkdDataSource` reservation ledger | new trainer subclass |
| Carry-over current work scheduling | existing `AsyncSkdAgentLoopManager` task loop | `FullyAsyncRollouter` queue |
| Stale budget enforcement | `_can_continue_lookahead_partial()` hook | new scheduler class |
| Checkpoint integration | `RayPPOTrainer._save_checkpoint()` and `_load_checkpoint()` hooks | separate checkpoint engine |
| Source-aware metrics | `DataProto.meta_info`, existing trainer metrics dict | `FullyAsync` metric names copied verbatim |
| Persistent rollout weight sync | GKD `sync_rollout_weights()` as reference | GKD trainer inheritance |

The next implementation priority is not another subclass. It is closing trainer input/output row accounting for promoted samples. When lookahead returns `B + Delta_k` generation rows, trainer input batch must also contain `B + Delta_k` input rows before `DataProto.union()`.

### 9.2 New Config Keys

Suggested config namespace:

```yaml
async_skd_lookahead:
  enable: false
  persistent_rollout_engines: true
  max_version_lag: 1
  max_version_span: 1
  prefetch_admission_ratio: 0.25
  prefetch_pair_quota: true
  max_old_gen_chunks: 2
  max_old_env_units: 1
  old_prefix_token_ratio: 0.125
  promote_finished: true
  drop_age_gt: 1
  tool_macro_step_atomic: true
  source_aware_metrics: true
```

Derived values:

```text
L_prefetch = min(B / N_pair, prefetch_admission_ratio * B)
old_prefix_token_cap = max_response_length * old_prefix_token_ratio
```

### 9.3 Per-Sample Type Discipline

구현 타입은 최소화한다. Manager-local scheduler가 결과 payload로 직접 다루는 top-level type은 `AsyncSkdSample` 하나로 둔다. Active task bookkeeping은 payload가 아니므로 별도 dataclass를 만들지 않고 `asyncio.Task` 기반 collection으로 처리한다.

```text
SkdPartialState:
  unfinished/carry-over trajectory를 resume하기 위한 payload.

AsyncSkdSample:
  manager-local scheduling envelope.
  completed payload와 partial payload를 하나의 인터페이스로 감싼다.
```

Completed sample의 실체는 `DataProto`다. Partial sample의 실체는 `SkdPartialState`다. 둘을 각각 별도 completed/partial wrapper로 늘리지 않는다.

Manager 내부 task bookkeeping은 다음 정도로 충분하다.

```python
current_active: dict[asyncio.Task, tuple[int, int]]
lookahead_active: dict[asyncio.Task, tuple[int, int]]
current_completed: list[DataProto | None]
promoted_lookahead: list[tuple[int, AsyncSkdSample]]
carryover_partials: list[tuple[int, SkdPartialState]]
```

새 `LookaheadTaskState`는 만들지 않는다. `sample_id`, `source_type`, `logical_step`은 worker call 인자와 반환되는 `AsyncSkdSample`/`SkdPartialState`에 이미 있다. Active task 값은 `(output_or_admission_order, worker_idx)` tuple이면 충분하다. Completed lookahead도 최종 `DataProto` 반환 직전까지 `AsyncSkdSample` envelope로 보존한다. 그래야 `source.record_promoted(...)`가 `sample_id`를 잃지 않는다.

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
    train_consume_version: int | None = None

    committed_gen_chunks: int = 0
    committed_env_units: int = 0
    committed_prefix_tokens: int = 0

    drop_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
```

Payload invariants:

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

Access rule:

```text
Trainer batch assembly must call require_completed().
Resume path must call require_partial().
Direct access to nullable payload fields is forbidden outside AsyncSkdSample validation helpers.
Scheduler should construct envelopes through from_completed() / from_partial() helpers.
```

This prevents the unified type from becoming a loose nullable object.

Removed type discipline:

```text
SkdCommittedUnit
RESUMABLE_COMMITTED_UNITS
last_committed_unit
skd_last_committed_unit
```

These labels are not canonical state. Export/restore must not depend on them.

### 9.4 New Per-Sample Fields

```python
sample_id: str
kind: str
logical_step: int
source_type: str
rollout_birth_version: int
rollout_min_version: int
rollout_max_version: int
train_consume_version: int
status: str
agent_state: str
old_gen_chunks: int
old_env_units: int
old_prefix_tokens: int
messages: list
prompt_ids: list[int]
teacher_prompt_ids: list[int]
response_ids: list[int]
response_mask: list[int]
response_logprobs: list[float]
teacher_ids_list: list[list[int]]
teacher_logprobs_list: list[list[float]]
assistant_turns: int
user_turns: int
tool_calls: list
tool_macro_steps: int
metrics: dict
extra_fields: dict
```

Fields from `messages` through `extra_fields` live inside `SkdPartialState` for partial samples. Completed samples keep their trainable tensors in `DataProto`; source and stale metadata live in the `AsyncSkdSample` envelope and may be mirrored into `DataProto.non_tensor_batch` for logging.

### 9.5 New Global Scheduler Fields

```python
base_batch_size: int
num_pairs: int
prefetch_admission_budget: int
prefetch_started_this_step: int
drain_flag: bool
current_param_version: int
pending_sync_version: int | None
promotable_finished_pool: list
carryover_paused_pool: list
next_step_reserved_quota: int
intentional_idle_time: float
```

### 9.6 New Functions

Agent loop:

```python
peek_next_atomic_unit(agent_data) -> AtomicUnit
estimate_next_atomic_unit_token_upper_bound(agent_data, unit) -> int
advance_one_atomic_unit(agent_data, version) -> tuple[AgentData, bool]
snapshot_agent_state(agent_data) -> LookaheadSnapshot
restore_agent_state(snapshot) -> AgentData
export_partial_state(agent_data, next_state) -> SkdPartialState
restore_partial_state(partial_state) -> tuple[AgentData, AgentState]
run_from_partial_to_completion(partial_state) -> AgentLoopOutput
can_export_partial_state(agent_data, next_state) -> bool
is_qwen_hermes_tool_boundary_exportable(agent_data) -> bool
```

SKD:

```python
run_exactly_one_skd_chunk(agent_data, version) -> ChunkResult
append_committed_teacher_rows(agent_data, teacher_ids, teacher_logprobs)
append_dummy_teacher_rows_for_env_span(agent_data, span_len)
assert_teacher_alignment(agent_data)
```

Rollouter:

```python
try_admit_lookahead_sample()
within_old_budget(sample, next_unit, cfg)
run_lookahead_until_pause_or_finish(sample, version, cfg, drain_flag)
run_carryover_until_completion(partial_state, current_version)
drain_all_lookaheads_to_exportable_boundary()
collect_finished_and_paused_lookaheads()
promote_finished_samples()
request_weight_sync(version)
apply_weight_sync_at_exportable_boundary(worker_id, version)
```

Current-step manager assembly:

```python
generate_sequences_with_carryover(fresh_prompts, carryover_partials) -> DataProto
```

This path completes carry-over samples first and fresh samples second. It uses the same worker-slot refill scheduler as the base lookahead path, so carry-over + fresh current work can admit bounded next-step lookahead while current work is still active.

Trainer:

```python
ensure_batch_uid(batch)
iter_training_batches() -> tuple[carryover_partials, fresh_batch, current_input_batch]
assemble_step_batch(base_samples, promoted_samples)
compute_next_step_fresh_quota(B, carryover_count)
mark_promoted_samples_trained(promoted_samples)
drop_expired_carryover_samples()
log_source_aware_metrics(batch)
compute_logprob_drift(batch)
sync_student_rollout_weights(version)
```

## 10. State Machines

### 10.1 Sample-Level State Machine

```text
RESERVED_FRESH
  -> CURRENT_ACTIVE
  -> LOOKAHEAD_ACTIVE_OLD

LOOKAHEAD_ACTIVE_OLD
  -> READY_PROMOTABLE
  -> PAUSED_CARRYOVER

READY_PROMOTABLE
  -> DONE_FOR_TRAIN

PAUSED_CARRYOVER
  -> RESUMED_ACTIVE_CURRENT
  -> DROPPED

RESUMED_ACTIVE_CURRENT
  -> DONE_FOR_TRAIN

CURRENT_ACTIVE
  -> DONE_FOR_TRAIN
```

State meanings:

| State | Meaning |
|---|---|
| `RESERVED_FRESH` | Prompt reserved but not rolled out |
| `CURRENT_ACTIVE` | Base current-step rollout sample |
| `LOOKAHEAD_ACTIVE_OLD` | Future sample being rolled out with old version |
| `READY_PROMOTABLE` | Lookahead sample fully finished before barrier |
| `PAUSED_CARRYOVER` | Lookahead sample paused at an exportable handler-return boundary |
| `RESUMED_ACTIVE_CURRENT` | Carry-over sample resumed with current version |
| `DONE_FOR_TRAIN` | Sample can enter trainer batch |
| `DROPPED` | Sample discarded due to age or budget |

### 10.2 AgentLoop Outer State Machine

`ToolAgentLoop`와 `SkdAgentLoop`가 공유하는 기본 state machine은 다음이다.

```python
state = AgentState.PENDING
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await _handle_pending_state(...)
    elif state == AgentState.GENERATING:
        state = await _handle_generating_state(...)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await _handle_processing_tools_state(...)
    elif state == AgentState.INTERACTING:
        state = await _handle_interacting_state(...)
```

Outer `AgentState`의 의미는 control-flow다.

| AgentState | Handler | Meaning |
|---|---|---|
| `PENDING` | `_handle_pending_state` | prompt stream 초기화 |
| `GENERATING` | `_handle_generating_state` | assistant response 생성 |
| `PROCESSING_TOOLS` | `_handle_processing_tools_state` | assistant tool call에 대한 tool result 반영 |
| `INTERACTING` | `_handle_interacting_state` | interaction/user result 반영 |
| `TERMINATED` | none | sample 종료 |

### 10.3 Handler-Return Export Boundaries

Export는 handler 내부에서 발생하지 않는다. Manager는 outer handler가 반환한 뒤에만 partial snapshot을 만들 수 있다.

```text
PENDING
  _handle_pending_state
    prompt streams initialized
    -> GENERATING

GENERATING
  _handle_generating_state
    repeat:
      student proposes one SKD chunk
      teacher verifies the chunk
      first rejected token is replaced by teacher top-1
      committed assistant tokens are appended
      response_mask += 1 span
      actual teacher rows are appended
      alignment is asserted
      if EOS:
        parse tool calls
        -> PROCESSING_TOOLS | INTERACTING | TERMINATED
      if no EOS and stop_after_skd_chunk:
        -> GENERATING

PROCESSING_TOOLS
  _handle_processing_tools_state
    tool call is executed
    tool result tokens are appended
    response_mask += 0 span
    dummy teacher rows are appended
    teacher_prompt_ids receives the tool-result delta
    alignment is asserted
    -> GENERATING | TERMINATED

INTERACTING
  _handle_interacting_state
    interaction/user result tokens are appended
    response_mask += 0 span
    dummy teacher rows are appended
    teacher_prompt_ids receives the interaction delta
    alignment is asserted
    -> GENERATING | TERMINATED
```

Because export is checked only after `await handler(...)` returns, internal Python substeps are not export candidates. Crash recovery during a handler is outside this cooperative lookahead design.

### 10.4 Partial Carry-Over Interpretation

If a lookahead task must be paused:

```text
1. return from the current outer handler.
2. require next_state == GENERATING.
3. assert teacher alignment.
4. apply Qwen/Hermes tool-boundary predicate.
5. export serialized state.
```

`AgentState.TERMINATED` is the outer completion state. A terminated sample is not resumed; it is either promoted into the current train batch or consumed as a completed sample.

Qwen/Hermes tool-boundary predicate:

```text
EOS 없음 + open tool-call prefix:
  export 가능

EOS 없음 + closed </tool_call>:
  export 금지

EOS 있음 + valid tool call:
  PROCESSING_TOOLS로 진행, tool result append 후 next_state == GENERATING에서 export 가능

EOS 있음 + no valid tool call + interaction enabled:
  INTERACTING으로 진행, interaction result append 후 next_state == GENERATING에서 export 가능

EOS 있음 + no valid tool call + no interaction:
  TERMINATED
```

### 10.5 Global Scheduler State Machine

```text
STEP_ROLLOUT_ACTIVE
  -> SLOT_REFILL_LOOKAHEAD_ADMISSION
  -> BARRIER_DRAIN
  -> TRAIN_UPDATE
  -> NEXT_STEP_ASSEMBLE
  -> STEP_ROLLOUT_ACTIVE
```

State meanings:

| State | Meaning |
|---|---|
| `STEP_ROLLOUT_ACTIVE` | Base batch rollout is running |
| `SLOT_REFILL_LOOKAHEAD_ADMISSION` | A worker request slot becomes free and admits bounded future work |
| `BARRIER_DRAIN` | Current LT finishes, lookahead tasks stop at next exportable handler-return boundary |
| `TRAIN_UPDATE` | Train on base + promoted samples |
| `NEXT_STEP_ASSEMBLE` | Build next step from carry-over + fresh quota |

### 10.6 Worker-Slot Refill Policy

verl sends one SGLang request per sample. It does not send the worker chunk as one batched tensor request. SGLang performs continuous batching internally over outstanding requests. Therefore lookahead scheduling should control the number of in-flight sample requests, not a tensor batch.

Initial policy:

```text
worker_capacity = ceil(base_batch_size / num_agent_loop_workers)
worker_active_count[worker_idx] = current samples + lookahead samples assigned to that worker
```

When a current-step sample finishes:

```text
1. decrement worker_active_count[worker_idx].
2. if drain_requested is false and global prefetch budget remains:
   reserve one lookahead sample.
3. launch that lookahead sample on the same worker_idx.
4. increment worker_active_count[worker_idx].
```

This is worker-replica aware, not exact GPU-number aware. Exact CUDA device id is not needed for the first implementation. Under TP=1 and one SGLang server per replica, worker/server-replica identity is the useful scheduling unit.

The first implementation is worker-slot aware. It does not require exact CUDA device ids and does not force preferred SGLang server routing. Server replica ids are observed through output metadata so worker-level scheduling can be compared against actual SGLang server distribution.

Server-replica observability should be added before server-directed routing:

```text
AsyncLLMServerManager.generate records rollout_server_id in TokenOutput.extra_fields.
Async SKD manager logs worker slot capacity, max active count, started lookahead count, and per-worker completed counts.
```

Do not add preferred-server routing in the first pass. If metrics show worker-level refill does not correlate with server-replica load, then add a later API:

```python
server_manager.generate(..., preferred_server_id="sglang_server_2_0")
```

That later API requires load-balancer changes and should be separate from worker-slot refill.

## 11. Invariants

The implementation must preserve the following invariants.

### 11.1 SKD Alignment Invariants

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

For tool/user spans:

```text
response_mask = 0
teacher row = dummy row
```

### 11.2 Sample Accounting Invariants

- A prompt is consumed exactly once.
- A promoted sample is recorded in the source ledger and must not be emitted again.
- A carry-over sample is either resumed in the next step or dropped.
- No sample with age greater than 1 enters training.
- Step-level batch size may be dynamic.
- Current-step work size is preserved by `fresh_quota = B - carryover_count`; promoted samples increase throughput.
- Lookahead admission is bounded by both global prefetch budget and per-worker active request capacity.
- A free worker slot may be filled only before `drain_requested=True`.

### 11.3 Atomicity Invariants

- Tool response is never partially appended.
- Interaction response is never partially appended.
- Generation chunk is never paused before teacher verification and commit.
- Snapshot state must be CPU-serializable and reconstructable without live KV cache.

### 11.4 Source Tracking Invariants

Every sample must retain:

```text
kind
source_type
logical_step
rollout_birth_version
rollout_min_version
rollout_max_version
old_gen_chunks
old_env_units
old_prefix_tokens
```

These fields must survive manager-local scheduling and, later, checkpoint serialization. For completed samples, the `AsyncSkdSample` envelope is the canonical source of async SKD metadata. For partial samples, `SkdPartialState` is the canonical resume payload and `AsyncSkdSample` is the scheduling envelope. If metadata is mirrored into `DataProto.non_tensor_batch` for logging, conflicts must be resolved in favor of the envelope before training metrics are computed.

### 11.5 Unified Envelope Invariants

- Manager-local active collections store `asyncio.Task`, not custom task-state dataclasses.
- Manager-local completed/carryover result handling uses only `AsyncSkdSample`, `DataProto`, and `SkdPartialState`.
- The manager must call `sample.validate()` before storing.
- A Ray actor queue should not be introduced unless producer and consumer become independent long-lived actors.
- Trainer batch assembly uses only `sample.require_completed()`.
- Resume path uses only `sample.require_partial()`.
- `LookaheadTaskState`, `SkdSampleSource`, `SkdCompletedSample`, `LookaheadResult`, `CarryoverSample`, and `PromotedSample` should not be introduced; they duplicate existing envelope or manager-local collection responsibilities.

## 12. Risks

### 12.1 Runtime-Conditioned Promotion Bias

Promoted samples are more likely to be short or easy. Admission cap bounds the magnitude but does not remove the bias.

### 12.2 Hybrid Trajectory Bias

Paused carry-over samples are mixed-version trajectories. Old prefix was produced by `theta_k`; suffix is produced by `theta_{k+1}`.

### 12.3 Dynamic Batch Size

Batch size becomes `B + Delta_k` for current step. Loss normalization must use actual valid tokens or actual batch metadata.

### 12.4 Tool Non-Determinism

If tool output depends on external state, replay/resume semantics can become less stable. The design assumes tool result is already materialized and stored once produced.

### 12.5 Serialization Cost

Snapshotting full agent state can be expensive. Compact representation should avoid dense full teacher tensors where possible.

### 12.6 Promoted Input/Output Row Mismatch

Finished lookahead promotion changes train output size from `B` to `B + Delta_k`. `DataProto.union()` requires the input batch and generated output batch to have identical row counts. Therefore promoted generation outputs cannot be appended alone. The source layer must also return the corresponding promoted input rows, and trainer batch assembly must concatenate:

```text
train_input_batch = base_input_batch + promoted_input_batch
train_gen_output = base_gen_output + promoted_gen_output
```

The row order of these two concatenations must match exactly. Otherwise `uid`, reward metadata, teacher metadata, and generated tensors can be paired with the wrong sample.

## 13. Recommended MVP

MVP should be conservative.

Config:

```text
prefetch_admission_budget = min(B / N_pair, 0.25B)
max_old_gen_chunks = 2
max_old_env_units = 1
old_prefix_token_cap = max_response_length / 8
promote_finished = true
drop_age_gt = 1
partial_rollout = false
```

Scope:

- Direct SKD only
- Tool step atomicity enforced
- Finished lookahead promotion enabled
- Paused carry-over resume enabled
- No IS correction
- No mid-tool interruption
- No live KV cache resume

Implementation priority after the current MVP state:

```text
1. promoted input batch assembly
2. stale budget enforcement
3. lookahead admission while completing carry-over + fresh current work
4. source checkpoint integration
5. source-aware metrics
6. config schema
7. persistent rollout/trainer split and weight sync
```

This order keeps row accounting correct before adding more concurrency.

## 14. Paper Framing

This method should not be described as an unbiased async KD estimator.

Recommended framing:

```text
We propose a bounded one-step lookahead scheme for tool-aware speculative
knowledge distillation. The method opportunistically fills tail-induced idle
time with future rollout work, while enforcing a one-version stale window and
chunk-safe resume boundaries. Finished lookahead trajectories are promoted
eagerly to avoid unnecessary staleness, whereas unfinished trajectories are
resumed in the next step only if their stale prefix satisfies strict semantic
and token-budget constraints.
```

Mathematical claim:

```text
The method introduces bounded bias rather than exact correction. Promotion bias
is controlled by the lookahead admission budget, and resume bias is controlled
by the number of old-version SKD generation chunks and the local policy drift
between consecutive synchronization versions.
```

Empirical claim to verify:

```text
Despite not using IS correction for direct KD, source-aware logs show that
promoted and resumed samples have comparable distillation loss, teacher mass,
response length, and benchmark performance to base current samples under the
chosen stale-prefix budgets.
```

## 15. Relation To Existing verl Async Paths

### 15.1 `recipe/gkd` One-Step-Off

`recipe/gkd`는 actor/trainer와 rollout worker를 분리하고, batch `k` actor update와 batch `k+1` rollout을 겹친다. 이 방식은 weight sync와 persistent rollout worker 구조를 참고할 가치가 있다.

하지만 이 문서의 목적과는 다르다.

```text
recipe/gkd:
whole batch k+1 rollout overlaps with batch k update
staleness is batch-level
tail inside rollout batch is still governed by the slowest trajectory

this design:
step k rollout 내부에서 idle pair가 제한된 k+1 sample을 생성
staleness is sample/prefix-level
long-tail idle을 직접 줄임
```

따라서 GKD scheduler를 그대로 이식하는 것이 아니라, 다음 요소만 모방한다.

- actor/trainer와 rollout engine 분리
- rollout engine을 persistent instance로 유지
- trainer update 후 student rollout weight sync
- version metadata 기록

### 15.2 `fully_async_policy`

`fully_async_policy`는 Rollouter, Trainer, MessageQueue, ParameterSynchronizer를 분리한다. Active task 관리, backpressure 개념, staleness metric은 참고할 수 있다.

하지만 현재 fully async agent loop는 distillation이 켜져 있으면 막혀 있다. 또한 기존 queue는 full일 때 oldest sample을 drop하는 방식이라 SKD에는 맞지 않는다. 초기 SKD path는 Ray actor queue를 만들지 않고 `AsyncSkdAgentLoopManager` 내부 state로 처리한다.

SKD async path에서는 다음이 필요하다.

- silent drop 없는 manager-local sample accounting
- teacher row alignment를 보존하는 partial state
- tool macro-step atomicity
- `version_lag <= 1` hard cap
- `k+2` 이상 prefetch 금지

For the current manager-local design, reuse only the active-task pattern:

```text
asyncio.wait(..., return_when=FIRST_COMPLETED)
task -> worker_idx bookkeeping
worker_active_count-based refill
```

Do not import the queue design. `MessageQueue` drops samples when full, which conflicts with source ledger accounting.

### 15.3 Persistent Instance Interpretation

최종 방향은 persistent generator/trainer 구조다.

```text
Student trainer:
GPU 0-3, current theta_k 보유, KD loss와 optimizer update 수행

Student rollout engines:
GPU 4-5, theta_bar_v snapshot으로 SKD student chunk 생성

Teacher rollout engines:
GPU 6-7, 고정 teacher로 verification 수행
```

Trainer가 update를 끝내면 student rollout engines에만 weight sync를 건다. Sync는 handler-return export boundary에서 적용한다. 이 구조는 GPU utilization을 높이지만, 안전한 work가 없으면 의도적으로 idle을 허용한다.

즉 목표는 unconditional 100% utilization이 아니라:

```text
work-conserving under bounded-staleness constraints
```

이다.

## 16. Summary

The design is best summarized as:

```text
Bound lookahead admission.
Promote every finished lookahead sample.
Pause unfinished samples only at handler-return export boundaries that satisfy real-state predicates.
Resume only one step later.
Bound old generation chunks and old prefix length.
Drop anything older.
Track source-aware metrics.
```

This is a conservative middle ground between synchronous SKD and fully asynchronous RL. It does not attempt exact IS correction for KD. Instead, it controls the deviation from synchronous on-policy SKD through explicit admission, stale-prefix, and atomicity constraints.
