# Implementation Detail

## 개요

본 구현의 목표는 speculative knowledge distillation (SKD)를 mixed rollout 환경에서 안정적으로 실행하는 데 있다. On-policy SKD는 일반적인 decode-heavy serving과 다른 **prefill-dominated multi-call workload**를 만든다. 학생은 커밋된 prefix 위에서 짧은 speculative chunk를 반복 생성하고, 교사는 같은 prefix 위에서 그 chunk를 반복 검증한다. 따라서 핵심 systems 문제는 단순한 decode throughput이 아니라, **trajectory 단위로 증가하는 committed prefix에 대응하는 KV state를 얼마나 안정적으로 유지하고 다시 쓰느냐**다.

학생 생성과 교사 검증은 모두 **고속 추론 엔진** 위에서 수행한다. 구현의 초점은 tool이 섞인 mixed rollout을 끝까지 유지하면서, teacher verification과 supervision 처리를 같은 실행 경로 안에서 함께 맞추는 데 있다. 이를 위해 우리는 세 요소를 결합한다. 첫째, RL 시스템에서 널리 쓰이는 추론-서빙 구조를 참고해 학생 생성, 교사 검증, tool execution이 함께 도는 **비동기 mixed rollout runtime**을 구성하고, 그 안에서 teacher-only asymmetric conditioning을 해결한다. 둘째, 반복 teacher verification이 **replica-local GPU KV cache**를 안정적으로 재사용하도록, 같은 trajectory의 요청을 동일 replica에 고정하고 각 verification step을 full-prefix fresh request로 구성한다. 셋째, teacher는 전체 `prefix + chunk`를 보되 반환은 suffix rows만 하도록 만들어, mixed rollout에서 필요한 teacher signal은 유지하면서 불필요한 postprocessing 비용은 줄인다.

## On-Policy SKD Loop

Algorithm 1 summarizes the semantic loop implemented by our system.

```text
Algorithm 1. On-policy SKD loop with first-rejection teacher verification

Input:
  committed prefix P
  chunk size C
  teacher retrieval top-k K_ret
  verification top-k K_ver
  max chunks M

Initialize aligned teacher targets T = []

for step = 1, 2, ..., M do
    chunk <- StudentGenerate(P, C)
    teacher_rows <- TeacherTopK(P + chunk, K_ret)
    rejection_pos <- None

    for k = 1, 2, ..., |chunk| do
        if chunk[k] not in teacher_rows[k][:K_ver] then
            rejection_pos <- k
            break
        end if
    end for

    if rejection_pos exists then
        chunk[rejection_pos] <- teacher_rows[rejection_pos][1]   # teacher top-1
        chunk <- chunk[1:rejection_pos]
    end if

    T <- T + teacher_rows[1:|chunk|]
    P <- P + chunk

    if EOS(chunk) or response budget exhausted then
        break
    end if
end for

return final committed prefix P and aligned teacher targets T
```

SKD loop에서 학생은 현재까지 커밋된 prefix를 기준으로 길이 `C` 이하의 chunk를 생성한다. 이후 교사는 `prefix + chunk`에 대해 위치별 **teacher retrieval top-k** 분포를 계산하고, 학생 chunk를 왼쪽부터 순서대로 검증한다. 학생 토큰이 **verification top-k** 안에 계속 포함되면 해당 토큰은 그대로 커밋된다. 반면 첫 번째 rejection이 발생하면, 그 위치의 학생 토큰은 교사의 top-1 토큰으로 교체되고, 그 뒤의 학생 suffix는 모두 버려진다. 다음 반복은 이 수정된 committed prefix를 기준으로 다시 시작된다.

핵심은 first-rejection semantics를 그대로 유지한다는 점이다. 교사의 top-1 교체는 실제 rollout state를 직접 갱신한다. 학생이 제안한 suffix 전체를 유지하는 완화된 방식이 아니라, **실제로 커밋된 경로만 다음 rollout의 상태로 이어지는 엄격한 speculative distillation**을 따른다. teacher target도 이 committed path에 대해서만 누적되므로, rollout semantics와 distillation target semantics가 자연스럽게 일치한다.

다음 step은 항상 **수정된 committed prefix**를 기준으로 시작된다. 학생과 교사 양쪽 모두에서 요청 패턴은 "긴 prefix 위에 짧은 continuation을 반복적으로 붙이는 형태"로 수렴한다. 이 섹션의 핵심은 SKD 알고리즘 자체를 다시 소개하는 데 있지 않다. 핵심은 이 반복 구조를 유지한 채 teacher verification, KV cache reuse, supervision alignment가 함께 성립하도록 실행 경로를 구성하는 데 있다. 바로 그 반복 구조 때문에 본 구현은 prefix-cache-sensitive workload가 된다.

## RL-Inspired Asynchronous Mixed Rollout Runtime

본 구현은 SKD를 single-turn reasoning에 한정하지 않고, 학생 생성, teacher verification, tool execution이 함께 돌아가는 **비동기 mixed rollout runtime**으로 확장한다. 학생, teacher, tool은 하나의 trajectory를 공유하지만 서로 다른 실행 역할을 가진다. 학생 생성과 교사 검증은 고속 추론 엔진에서 처리하고, trajectory 상태 갱신과 tool execution은 별도의 제어 경로에서 처리하도록 실행 구조를 나눴다. 이 구성은 RL 시스템에서 널리 쓰이는 추론 경로와 제어 경로의 분리 방식에서 출발한다.

이 runtime에서는 assistant generation, tool execution, 외부 observation이 하나의 trajectory 안에서 번갈아 등장한다. 시스템은 단순한 "프롬프트-응답" 루프가 아니라, 상태를 이어 가는 다단계 실행기로 동작한다. 학생 생성, tool 호출, 외부 관찰 반영이 반복되는 동안 메시지 상태와 토큰 상태를 계속 유지하고, 각 단계의 결과가 다음 단계의 문맥과 teacher target에 바로 반영되도록 구성했다.

여기에 **teacher-only asymmetric conditioning**을 같은 runtime 안에 통합한다. 학생은 실제 rollout policy가 따르는 문맥을 그대로 유지하고, 교사는 별도의 verification 문맥을 따른다. 초기 단계에서 학생과 교사는 서로 다른 system guidance를 포함한 문맥으로 시작하고, 이후 학생이 생성한 token과 tool/user 관찰은 두 문맥에 함께 반영된다. 학생의 행동 정책은 바뀌지 않고, 교사에게는 더 강한 verification prior만 추가된다.

Mixed rollout에서는 teacher target이 중간에 쉽게 밀린다. assistant span 사이에 tool response나 외부 observation이 끼어들기 때문이다. 이를 막기 위해 우리는 **응답 위치 기준의 online 정렬 규약**을 사용한다. 학생이 생성한 span에는 실제 teacher supervision을 기록하고, 학생이 생성하지 않은 span에는 같은 길이의 placeholder를 삽입한다. 그 결과 tool response가 중간에 들어와도 이후 학생 생성 구간의 teacher signal은 같은 응답 위치에 그대로 남는다.

## Prefix-Local Teacher Verification via Replica-Local KV Cache Reuse

teacher verification은 하나의 긴 decode가 아니라, 증가하는 committed prefix 위에서 반복적으로 발행되는 짧은 요청들의 연속이다. 이 단계의 핵심 systems 문제는 prefix 문자열을 반복해서 보내는 것 자체가 아니라, **증가하는 committed prefix에 대응하는 KV cache state를 replica 내부에서 얼마나 안정적으로 재사용하느냐**에 있다.

현재 구현은 같은 trajectory의 teacher verification 요청을 항상 같은 replica로 라우팅하고, 각 verification step을 **full-prefix fresh request**로 구성한다. 각 요청은 이전 요청과 긴 prefix를 공유하므로, runtime은 prefix matching을 통해 기존 cache를 다시 사용할 수 있다. 이 실행 방식은 single-GPU 단일 세션 경로를 전제하지 않는다. 학생 생성과 교사 검증이 여러 inference replica에 걸쳐 비동기적으로 수행되는 환경을 전제로 하며, 바로 그 이유 때문에 **replica-local GPU KV cache locality**를 명시적으로 보존해야 한다.

이 구조의 핵심은 committed 경로에 필요한 prefix KV를 정확히 다시 쓸 수 있다는 점이다. 예를 들어 학생이 `P + a b c d`를 생성하고 교사가 `c`에서 rejection을 일으켜 실제 committed path가 `P + a b T`가 되면, 다음 요청은 `P + a b T ...`로 진행되고 runtime은 이전 cache로부터 공통 prefix인 `P + a b`를 다시 매치한다. 현재 구현의 핵심 이득은 **committed prefix reuse의 정확성**에 있다.

다만 현재 구조는 reject된 suffix KV를 즉시 폐기하지는 못한다. accept/reject 판단이 학생 request가 끝난 뒤, 즉 엔진 바깥에서 이루어지기 때문이다. 따라서 현재 남아 있는 systems gap은 prefix reuse failure가 아니라, **rejected suffix over-retention**이다. 다시 말해, 재사용해야 할 KV는 이미 모두 들고 있지만, 재사용하지 않을 suffix KV를 한동안 더 들고 있게 된다.

## Efficient and Stable Teacher Supervision Path

teacher verification은 전체 prefix를 조건으로 유지해야 하지만, 실제로 distillation에 필요한 정보는 현재 speculative suffix에 대응하는 supervision뿐이다. 순진한 구현은 매 step마다 `prefix + chunk` 전체에 대한 teacher 분포를 다시 만들고 전달하게 된다. 우리는 **teacher는 전체 `prefix + chunk`를 보되, 반환은 suffix rows만 하도록** 구성한다. 그러면 반환되는 결과의 길이는 `prefix_len + chunk_len`이 아니라 `chunk_len`으로 줄어든다. 폭은 그대로 top-k다.

이 설계는 teacher verification semantics를 바꾸지 않으면서도, suffix 길이에 비례한 처리만 남긴다. 따라서 teacher 결과를 Python/Ray 경로로 받고, 필요한 row를 정리하고, teacher ids와 logprobs를 누적하고, 이후 학습용 tensor로 조립하는 **postprocessing 비용**도 함께 줄어든다. compute 측 핵심 전제는 여전히 prefix-local KV reuse이고, 여기서는 그 위의 반환물과 postprocessing만 줄인다.

동시에 이렇게 얻은 supervision은 mixed rollout 전반에서 **응답 위치와 맞물린 상태**로 유지된다. trajectory 안에는 학생이 직접 생성한 assistant span뿐 아니라 tool response와 외부 observation span도 함께 존재하므로, teacher signal은 응답 위치와 일대일로 대응되도록 온라인으로 누적되어야 한다. 학생이 생성한 span에는 실제 teacher supervision을 기록하고, tool response나 외부 관찰처럼 학생이 생성하지 않은 span에는 같은 길이의 placeholder를 함께 삽입한다. 그 결과 teacher supervision은 suffix rows만 회수하면서도, 학습 단계에서는 mixed rollout 전체와 같은 길이와 순서를 유지한다. 불필요한 postprocessing은 줄고, mixed rollout에서 필요한 supervision 안정성은 그대로 유지된다.

## 요약

정리하면, 우리의 구현은 단순히 SKD 아이디어를 연결한 수준이 아니다. 우리는 (1) tool-aware asymmetric mixed rollout runtime, (2) replica-local KV cache reuse에 기반한 prefix-local teacher verification, (3) 효율적이면서도 안정적인 teacher supervision path를 하나의 학습 시스템 안에 묶었다. 이 조합을 통해 **prefix-cache-sensitive on-policy distillation workload를 실제로 안정적으로 실행 가능한 시스템**으로 만들었다는 점이 본 구현의 핵심 기여다.

## 작성 TODO

- **그림 추가**: mixed rollout 전체 흐름을 한 장으로 보여 주는 그림이 필요하다. 학생 chunk generation, teacher-only asymmetric conditioning, tool calling, tool response 및 외부 observation 삽입까지 한 번에 드러나야 한다.
- **구현 및 공개 맥락 정리**: 실제 구현이 `verl` 코드베이스 위에서 이루어졌고, 추론 엔진으로 SGLang을 사용했다는 점을 뒤쪽에서 정리해야 한다. 특히 teacher logprob 확보가 사실상 SGLang 경로에 기대고 있다는 설명도 함께 넣어야 한다.
- **teacher retrieval top-k 효율성 근거 보강**: 전체 vocabulary 분포를 매 step 저장·반환하지 않고 `teacher retrieval top-k`만 사용하는 이유를 정량적으로 설명해야 한다. Qwen3 기준으로 full distribution을 사용할 때의 메모리 및 전송 비용도 같이 제시해야 한다.
- **`top-k = 32` 선택 근거 추가**: 현재 사용하는 `teacher retrieval top-k = 32`가 충분하다는 통계와 실험 수치를 채워 넣어야 한다. 이 부분은 현재 서술만 있고 수치가 비어 있다.
- **용어 점검**: `teacher retrieval top-k`, `verification top-k`, `teacher top-1`이 문서 전체에서 일관되게 쓰이는지 마지막에 한 번 더 점검해야 한다.
