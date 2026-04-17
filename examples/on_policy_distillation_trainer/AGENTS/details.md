# On-Policy Distillation 운영 메모

이 문서는 [`AGENTS/onboarding.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/onboarding.md) 와 [`AGENTS/imp_detail.md`](/home/work/DDAI_revised/OSworld/verl/examples/on_policy_distillation_trainer/AGENTS/imp_detail.md) 사이에 있는 내부 메모다.  
목적은 구현을 미화하는 것이 아니라, 실제로 반복해서 부딪힌 실수와 판별 기준을 남겨 같은 실수를 다시 하지 않게 하는 데 있다.

## 서론

이번 축에서 반복적으로 문제를 만든 것은 세 가지였다.

1. 레이어 책임 혼동  
   dataset, runtime, config, manager, loop의 책임을 섞으면 startup failure와 원인 미분리가 곧바로 따라왔다.
2. 정렬 계약 위반  
   tool/user span이 response 안에 섞이는 순간 teacher target 정렬이 조금만 틀어져도 `teacher_mass_max=128`, `distillation/loss_max=1280`이 바로 재발했다.
3. 병목과 크래시의 혼동  
   teacher가 느린 것, student rollout이 무거운 것, actor backward가 죽는 것은 서로 다른 문제였다.

이 문서는 위 세 가지를 기준으로 정리한다.

## 본론

### 1. Dataset 단계에서는 실제 tool을 띄우지 않는다

초기에는 prompt filtering이나 schema 조회 단계에서도 tool config를 instantiate했다. 잘못된 판단이었다. dataset이 필요로 하는 것은 실행기가 아니라 `tool_schema`다. 실제 tool runtime을 dataset init에서 띄우면 startup 시 sandbox actor가 먼저 생기고, rollout 이전부터 side effect가 생긴다. 현재 원칙은 단순하다. dataset은 YAML에서 static schema만 읽고, 실제 backend는 rollout worker가 책임진다.

### 2. Sample마다 agent loop를 새로 만들면 대규모 배치에서 런이 무너진다

초기 구조에서 sample마다 `ToolAgentLoop`를 다시 만들었고, 그 결과 sample마다 tool 객체와 sandbox execution worker도 함께 재생성됐다. 증상은 `ExecutionWorker` abort, `ActorDiedError`, step `0` 장기 정체였다. 이 문제는 알고리즘 문제가 아니라 lifetime 설계 문제였다. 현재는 worker-level cache로 정리되어, 같은 `AgentLoopWorker` 안에서는 agent loop와 tool runtime을 재사용한다.

### 3. SandboxFusion 서버 readiness를 확인하지 않으면 tool 실험은 성립하지 않는다

tool call이 로그에 보인다고 해서 실제 tool execution이 들어간다고 보면 안 된다. `localhost:8080/run_code`에 서버가 안 떠 있으면 tool 호출은 형식상 수행되지만 실제 결과는 `Connection refused`이고, 모델은 실패 응답만 본다. 최소 체크는 두 가지다. SandboxFusion 서버를 먼저 띄우고, `curl` 또는 단일 `code_interpreter` smoke test로 실제 출력이 반환되는지 확인한다. 이 두 단계를 건너뛰고 RL을 먼저 태우면 런은 돌아가는 척해도 실험 의미는 틀어진다.

### 4. Optional runtime component는 생성 경로뿐 아니라 정리 경로도 `None` 안전해야 한다

`enable_global_rate_limit=False`일 때도 `release.remote`를 호출하는 경로가 남아 있어서 `NoneType.release`가 터진 적이 있다. 이 유형은 흔하다. optional component는 “안 만든다”로 끝나지 않는다. acquire, release, cleanup 모두가 `None` 안전해야 한다. 따라서 optional actor는 항상 “생성 여부”와 “생성 안 된 상태의 후속 호출”을 함께 점검한다.

### 5. Tool-aware SKD에서 핵심은 teacher row 정렬이다

single-turn SKD만 보면 놓치기 쉽지만, tool/user span이 response 안에 들어오면 정렬이 가장 먼저 깨진다. 예를 들어 `AAA TTT AAA`에 대해 mask가 `111 000 111`이면, assistant token에 대해서만 teacher row를 누적할 경우 뒤쪽 `AAA`가 왼쪽으로 밀린다. loss는 `response_mask=1` 위치만 보기 때문에, 그 위치가 dummy row를 먹기 시작하는 순간 pathology가 난다. 초기에 `im_end` 같은 특수 토큰 문제로 보기도 했지만 본질은 특정 토큰이 아니라 `0-mask` span 전체를 teacher 쪽에서 정확히 반영하지 않은 것이었다.

### 6. Dummy는 supervision이 아니라 dense tensor 계약을 위한 placeholder다

“어차피 loss는 `response_mask`만 보는데 dummy가 왜 필요한가”라는 질문은 자연스럽지만, 현재 `verl` distillation 경로는 full-sequence dense tensor를 기대한다. 즉 dummy는 학습적으로 중요한 row가 아니라, full-sequence shape와 left-shift slicing 계약을 맞추기 위한 placeholder다. 정리하면, loss는 mask만 보지만 현재 코드 구조는 dense teacher tensor를 먼저 요구한다. 따라서 dummy는 불필요한 장식이 아니라 인터페이스 적합성의 일부다.

### 7. Dummy width는 runtime 추론보다 config `topk`를 source of truth로 두는 편이 낫다

초기에는 첫 teacher row 길이에서 dummy width를 추론하는 방식을 떠올리기 쉽다. 그러나 현재 구조에선 `distillation.distillation_loss.topk`가 더 올바른 기준이다. teacher top-k width는 config 계약이고, runtime에서 조용히 추론하면 mismatch를 따라가 버린다. 이런 경우는 조용히 맞추는 것보다, config를 source of truth로 두고 실제 row width와 다르면 바로 assert로 잡는 편이 안전하다.

### 8. Teacher-only system prompt는 teacher prompt stream을 별도로 가져가야 한다

teacher에만 추가 prompt를 주고 싶다고 해서 teacher manager에서 token 몇 개를 앞에 붙이는 식으로 처리하면 안 된다. teacher manager는 이미 token sequence만 보고, message 구조와 chat template를 모른다. 또 `logprob_start_len`은 prefix 길이에 직접 민감하다. 현재 구조에서 안전한 해법은 student와 teacher prompt stream을 분리하는 것이다. 즉 `agent_data.prompt_ids`와 `agent_data.extra_fields["teacher_prompt_ids"]`를 따로 유지하고, teacher verify는 언제나 teacher stream 기준으로 수행한다.

### 9. Teacher-only prompt를 넣으면 teacher budget reserve도 자동으로 따라가야 한다

teacher-only system prompt를 넣고도 teacher `max_model_len`을 student 기준으로 그대로 두면, 실제 teacher prefix가 길어지면서 context overflow가 난다. 이건 한 번 크게 터졌다. 현재는 teacher prompt path가 켜져 있으면 teacher inference budget에 고정 512-token margin을 자동으로 더한다. 여기서 또 하나 배운 점이 있다. 이 로직을 넣을 때 `self.skd.teacher_system_prompt_path`처럼 dataclass attribute 접근을 쓰면 Hydra 경로에서 `self.skd`가 dict일 때 config init이 터진다. config layer의 기존 기조에 맞춰 nested config는 `get()` 방식으로 읽는 것이 맞다.

### 10. Validation은 student 평가 단계이지 teacher-guided rollout 재현 단계가 아니다

validation을 train과 동일한 `skd_agent` 경로로 두면 student 자체 평가와 teacher guidance가 섞인다. 현재 원칙은 train은 `skd_agent`, validation은 `tool_agent`다. validation은 어디까지나 student policy 자체를 본다. 또 `n=4`만 준다고 `best@4`가 생기지 않는다. `do_sample=False`면 greedy 결과를 4번 복제하는 것과 다르지 않다. 실제 `mean@4`, `best@4`를 보려면 `val_kwargs.n=4`와 `do_sample=True`가 함께 필요하다.

### 11. Teacher가 느려 보여도 항상 teacher가 병목인 것은 아니다

초기에는 32B teacher와 repeated prompt-logprob 때문에 teacher가 확실한 병목이었다. 하지만 Nemotron과 long response 설정으로 오면서 sample-level summary 기준으로 student cumulative time이 teacher보다 큰 런도 생겼다. 특히 `AsyncLLMServerManager.generate.total`만 보고 병목을 판단하면 student와 teacher 경로가 섞여 오판하기 쉽다. 병목 판단은 가능한 한 sample-level `[SKD] student=... teacher=...` 누적 summary를 기준으로 한다.

### 12. 최종 크래시는 teacher가 아니라 actor backward OOM이었다

teacher를 크게 바꾸고 FP8로 올리면 자연스럽게 teacher memory나 teacher latency를 먼저 의심하게 된다. 하지만 실제 최종 크래시는 rollout 이후 `update_actor`의 `loss.backward()`에서 났다. 즉 메모리 리스크는 teacher inference보다 `b128`, 긴 response, actor backward accumulation 조합에 있었다. 이 점이 중요하다. teacher top-k를 `128 -> 32`로 줄이면 teacher payload와 extraction cost는 줄지만 actor backward activation memory는 거의 직접 줄지 않는다. teacher 최적화와 actor OOM 대응은 별개로 본다.

### 13. `ppo_max_token_len_per_gpu`는 sample truncation이 아니라 micro-batch budget이다

이 값은 “한 sample을 몇 토큰까지만 backprop한다”는 뜻이 아니다. 정확히는 GPU 한 장이 동시에 처리하는 총 token budget 상한에 가깝다. 값을 줄이면 한 번에 묶는 sample 수가 줄고, 필요하면 micro-batch 수가 늘어난다. 즉 학습 의미를 자르는 것이 아니라 동시 처리량을 낮춰 메모리를 맞추는 것이다. 다만 이 값만 조금 줄이는 미세 조정에는 한계가 있었다. 이미 12개 micro-batch까지 쪼개고도 OOM이 난 상황에서는 `18000 -> 16000` 같은 조정보다 `batch 128 -> 64`처럼 배치 자체를 줄이는 편이 더 직접적이었다.

### 14. Nemotron train set은 mixed-task라는 점을 잊지 않는다

`Nemotron-Cascade-RL-Math`는 단순 direct solving dataset이 아니다. 일반 수학 문제와 solution critique, earliest-error index task가 섞여 있다. 현재는 이걸 전부 train으로 쓴다. 따라서 전처리의 목적은 task를 억지로 하나로 바꾸는 것이 아니라 의미를 보존하는 데 있다. 현재 원칙은 `problem`을 그대로 쓰고, `answer`도 가공 없이 ground truth로 두며, `task_type`과 `source`를 추적 가능하게 남기고, `\boxed{}` carrier instruction만 제거하되 `decimal`, `base 10`, `without units` 같은 semantic format constraint는 유지하는 것이다.

### 15. Reward는 대부분 버티지만, 모든 sample이 깔끔하다고 가정하면 안 된다

현재 reward는 사실상 `math_verify` 기반이고, 대부분의 direct math sample에는 잘 맞는다. 하지만 mixed-task dataset이므로 index answer, decimal/base-10 formatting, 드문 비수식 answer는 잠재 리스크다. 한동안 마지막 `\boxed{}`만 먼저 채점하거나 긴 unboxed 출력을 바로 `0`으로 두는 wrapper를 실험했지만, 실제 dump 기준으로 under-score가 확인되어 그 분기는 원복했다. 현재 운영 원칙은 다시 단순하다. `solution_str` 전체를 `math_verify`에 넘기는 기존 경로를 유지하고, parseability audit이나 reward 수정은 실제 failure pattern이 충분히 쌓였을 때만 좁혀 들어간다.

### 16. `math_verify` hang 문제는 scorer만이 아니라 실행 경계를 같이 봐야 한다

한동안 base RL run이 step 후반에서 멈췄고, 표면상으로는 `RewardLoopWorker.compute_score`가 끝나지 않는 것처럼 보였다. 처음에는 단순히 scorer 안에 바깥 timeout만 더 두는 식으로 생각하기 쉽다. 그러나 실제 call chain은 `RewardLoopWorker -> NaiveRewardManager.run_single() -> run_in_executor(None, compute_score)`였고, 여기서 핵심은 `math_verify`가 내부 timeout 구현에 `signal.alarm()`을 쓴다는 점이었다.

즉 문제의 본질은 “느린 sample이 있다”보다 더 정확히는, **timeout이 필요한 라이브러리를 thread executor 안에 넣었다**는 데 있었다. 이 상태에서 바깥 `future.result(timeout=...)`만 걸면 호출자 대기만 끊기고, 실제 scoring thread는 worker 안에서 계속 남는다. pathological case가 한 번 들어오면 worker 안에 CPU를 먹는 stray task가 남고, 이런 상태가 누적되면 결국 전체 reward path가 막힌다.

그래서 이번에 건드린 축은 scorer 로직 자체보다 **실행 경계**였다. 현재 `math_verify` 경로는 lazily initialized process pool 안에서 돌고, `parse`와 `verify`는 그 child process의 main thread에서 자신의 timeout을 정상적으로 쓴다. 즉 “timeout이 필요한 코드를 main thread에서 실행한다”는 라이브러리의 전제를 맞춘 것이다. 이건 단순한 기워내기보다, 기존 thread 기반 실행 모델이 애초에 라이브러리와 맞지 않았다는 판단에 따른 수정이다.

다만 이것으로 구조 문제가 완전히 끝난 것은 아니다. 지금 패치는 `math_verify` 경로를 안전하게 만든 것이지, reward execution policy 전체를 일반화한 것은 아니다. 더 근본적으로는 `RewardLoopWorker` 또는 `NaiveRewardManager` 레벨에서 sync custom reward를 어떤 격리 경계에서 돌릴지, timeout/crash를 어떻게 처리할지, worker를 언제 recycle할지를 framework policy로 올리는 편이 맞다. 정리하면 현재 patch는 **현재 장애의 직접 원인에는 맞는 수정**이지만, 장기적으로는 reward isolation 자체를 scorer 바깥 계층으로 승격해야 한다.

### 17. constrained decoding은 보류했다

Hermes tool calling 위에 SGLang의 trigger-based constrained decoding을 붙이는 방향을 검토했고, 실제로 parser span 계산, `response_mask` 보정, `structural_tag` 주입까지 작은 프로토타입도 만들었다. 그러나 현재 `verl`의 SKD 구조와는 정합성이 충분하지 않아, 이번 축에서는 전부 원복했다.

보류 이유는 두 가지다.

1. **SKD는 fresh-request chunk generation이다.**  
   학생이 chunk를 한 번 생성할 때마다 새 request를 보내므로, partial tool-call 상태를 online grammar state로 자연스럽게 이어받기 어렵다.

2. **우리가 원한 건 사후 loss masking이 아니라 online verification 제외였다.**  
   구조 토큰을 정말 teacher supervision에서 빼려면, turn 종료 후 `response_mask`를 고치는 것이 아니라 SKD accept/reject 단계에서부터 그 토큰들을 건너뛰어야 한다. 이건 partial tool-call 상태를 안정적으로 해석하는 추가 설계가 필요하다.

즉 현재 결론은 명확하다.

- constrained decoding은 아이디어 자체는 유망하다
- 하지만 지금 `verl`의 SKD 경로에 억지로 붙이면 구현 복잡도와 semantics mismatch가 크다
- 따라서 현재 코드베이스에서는 **기존 Hermes 자유 생성 + 사후 파싱**을 유지한다

이번 단계에서 하지 않는 것은 다음과 같다.

- exact jump-forward provenance 복원
- partial tool-call 상태를 위한 online structural masking
- constrained decoding을 SKD online verification에 연결하는 것
- nested / parallel tool call 지원
- Qwen3-Coder 별도 XML-ish format 동시 지원

즉 이 문서 기준의 현재 상태는 **constrained decoding 실험은 철회했고, 기존 `verl` tool calling contract를 유지한다**는 것이다.

## 결론

이번 작업에서 다시 반복하지 말아야 할 실수는 세 가지다.

1. 레이어 책임을 섞지 않는다.  
   dataset, runtime, config, manager, loop의 책임이 다르다.
2. tool-aware SKD에서는 정렬을 최우선으로 본다.  
   0-mask span이 끼는 순간 teacher row 정렬이 깨지면 loss pathology가 바로 난다.
3. 병목과 크래시는 분리해서 본다.  
   teacher가 느린 것, student rollout이 무거운 것, actor backward가 죽는 것은 같은 문제가 아니다.

실제로 안정화에 가장 크게 기여한 것은 화려한 최적화보다, 위 세 가지를 뒤늦게라도 분리해서 본 것이었다.
