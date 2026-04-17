# On-Policy Distillation 주의 사항

이 문서는 이 작업 영역에서 에이전트가 반드시 지켜야 하는 운영 규칙을 정리한다.  
구현 상세는 [`AGENTS/imp_detail.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/imp_detail.md), 빠른 진입은 [`AGENTS/onboarding.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/onboarding.md)를 본다. 본 문서는 환경, 실행, 수정 범위, 로그 해석에서의 주의점만 다룬다.

## 서론

이 영역의 실수는 대부분 모델 구조보다 운영 방식에서 발생했다.  
특히 다음 네 가지를 지키지 않으면 같은 오류가 반복되기 쉽다.

1. 올바른 conda 환경을 쓰지 않는 경우
2. 읽기 전용 환경을 수정하려는 경우
3. sandbox/tool readiness 확인 없이 RL을 먼저 태우는 경우
4. teacher 문제, student 문제, actor memory 문제를 섞어 해석하는 경우

## 본론

### 1. 기본 작업 환경은 `kd`다

코드 확인, 테스트, 전처리 점검, 로그 분석은 기본적으로 `kd` 환경에서 수행한다.

```bash
source /home/work/DDAI_revised/miniconda3/etc/profile.d/conda.sh
conda activate kd
```

별도 지시가 없으면:

- Python 확인
- `pytest`
- `py_compile`
- 데이터셋 점검
- 실행 스크립트 sanity check

는 모두 `kd`에서 수행한다.

### 2. `kd` 환경은 읽기 전용으로 취급한다

`kd`는 실험 재현용 기준 환경이다. 에이전트는 이 환경을 수정하지 않는다.

금지 사항:

- `pip install`
- `uv pip install`
- `conda install`
- `poetry install`
- 패키지 업그레이드/다운그레이드
- 환경 변수 파일, site-packages, interpreter 경로 수정

즉 `kd`는 **사용만 하고 변경하지 않는 환경**으로 취급한다.  
환경 문제가 생기면 코드나 실행 설정을 먼저 본다. `kd` 자체를 손보는 쪽으로 가지 않는다.

### 3. SandboxFusion 서버는 별도 환경에서 다룬다

`code_interpreter` 경로를 쓰는 실험은 RL보다 먼저 SandboxFusion readiness를 확인해야 한다.

서버 구동은 보통 `sandbox` 같은 별도 환경에서 처리한다.

```bash
source /home/work/DDAI_revised/miniconda3/etc/profile.d/conda.sh
conda activate sandbox
```

최소 확인 순서:

1. 서버가 떠 있는지 확인
2. `curl` 또는 단일 `code_interpreter` smoke test 수행
3. 실제 출력이 반환되는지 확인
4. 그 다음에만 RL 실행

tool call 로그가 보인다고 해서 sandbox execution이 실제로 성공했다고 가정하면 안 된다.

### 4. RL 실행 전에 기존 프로세스를 먼저 정리한다

새 실험을 시작하기 전에는 기존 rollout / trainer / sandbox 관련 프로세스가 남아 있지 않은지 먼저 확인한다.  
이 영역은 장시간 실행 프로세스가 많기 때문에, 이전 런이 남아 있으면 로그 해석과 자원 상태를 쉽게 오염시킨다.

특히 다음 계열은 중복 실행을 피한다.

- `main_ppo`
- `TaskRunner`
- `AgentLoopWorker`
- `SGLangHttpServer`
- `ExecutionWorker`
- `wandb`
- `gpu_stats`

### 5. 수정은 코드와 설정에 한정하고, 환경 자체는 건드리지 않는다

문제가 발생했을 때 우선순위는 다음과 같다.

1. 실행 스크립트 설정 확인
2. config/dataclass 계약 확인
3. rollout / training 코드 확인
4. tool readiness 확인

환경 수정은 마지막 수단이 아니라, 이 작업 영역에서는 원칙적으로 피하는 선택이다.

### 6. Validation은 train-time teacher guidance와 분리해서 본다

이 축에서 validation은 student policy 자체를 평가하는 단계다.  
따라서 validation 동작을 해석할 때는 teacher-guided train rollout과 구분해야 한다.

추가로 `best@4`, `mean@4`를 보려면:

- `val_kwargs.n=4`
- `do_sample=True`

가 함께 필요하다.  
`n=4`만 주고 sampling이 꺼져 있으면 사실상 greedy 복제에 가깝다.

### 7. Teacher 문제와 actor OOM을 같은 문제로 보지 않는다

이 축에서 자주 생기는 오판은 다음과 같다.

- teacher가 느리다
- 그래서 teacher가 메모리 크래시의 원인일 것이다

실제로는 다를 수 있다.  
최근 런에서는 teacher 병목과 별개로, 최종 크래시는 `update_actor -> loss.backward()`에서의 actor-side OOM으로 나타났다.

따라서 로그를 볼 때는 다음을 분리한다.

- teacher latency 문제
- student rollout 누적 시간 문제
- actor backward memory 문제

특히 `torch.OutOfMemoryError`가 `update_actor` 아래에서 나면, teacher가 아니라 actor training memory로 본다.

### 8. `ppo_max_token_len_per_gpu`는 sample truncation이 아니다

이 값은 “sample을 여기까지만 학습한다”가 아니라, GPU 한 장이 한 번에 처리하는 총 token budget 상한에 가깝다.  
즉 값을 줄이면 sample을 버리는 것이 아니라, 더 작은 micro-batch로 나눠 처리하게 된다.

이 점을 모르면:

- token budget 조정
- batch size 조정

의 의미를 혼동하기 쉽다.

### 9. Teacher-only prompt를 쓰면 teacher budget reserve를 염두에 둔다

teacher-only system prompt를 넣으면 teacher prefix는 student prefix보다 길어진다.  
따라서 teacher context budget은 student 기준 값과 같다고 가정하면 안 된다.

현재 구현은 config 레이어에서 reserve를 자동으로 더하지만, 로그 해석이나 실행 설정을 볼 때는 이 사실을 알고 있어야 한다.

### 10. AGENTS 문서를 수정할 때는 편의보다 규칙을 따른다

이 디렉터리의 문서는 에이전트가 반복적으로 읽는 문서다.  
따라서 수정할 때는:

- 구현 상세는 `imp_detail.md`
- 빠른 진입은 `onboarding.md`
- 운영 규칙은 `warning.md`

처럼 역할을 분리한다.  
문서 길이와 중복도 함께 관리한다.

## 결론

이 작업 영역에서 가장 중요한 준수 사항은 다음 네 가지다.

1. 기본 작업은 `conda activate kd`에서 수행한다.
2. `kd` 환경은 읽기 전용으로 취급하고 수정하지 않는다.
3. tool 실험은 SandboxFusion readiness를 먼저 확인한 뒤 실행한다.
4. teacher 병목, student rollout 문제, actor OOM을 분리해서 해석한다.

위 네 가지를 지키면, 같은 유형의 시행착오를 상당 부분 줄일 수 있다.
