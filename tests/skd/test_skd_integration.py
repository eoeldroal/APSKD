"""
Speculative Knowledge Distillation (SKD) 통합 테스트.

실제 vLLM HTTP 서버 (Student + Teacher)를 사용하여 SKD 로직을 검증한다.
서버를 먼저 띄운 뒤 실행: bash tests/skd/launch_test_servers.sh

Usage:
    python tests/skd/test_skd_integration.py
    python tests/skd/test_skd_integration.py --chunk-size 512 --verify-top-k 25
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from transformers import AutoTokenizer


# ============================================
# Config
# ============================================

STUDENT_URL = "http://127.0.0.1:8001"
TEACHER_URL = "http://127.0.0.1:8002"
STUDENT_MODEL = "/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-1.7B"
TEACHER_MODEL = "/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-8B"
TOKENIZER_PATH = "/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-1.7B"

# Qwen3 special tokens
EOS_TOKEN_ID = 151645   # <|im_end|>
THINK_START = 151667    # <think>
THINK_END = 151668      # </think>
TOOL_CALL_START = 151657
TOOL_CALL_END = 151658


# ============================================
# HTTP Server Wrappers
# ============================================


@dataclass
class TokenOutput:
    """verl의 TokenOutput과 동일한 인터페이스."""
    token_ids: list[int]
    finish_reason: str
    log_probs: Optional[list[float]] = None
    extra_fields: dict = field(default_factory=dict)


class HttpStudentServer:
    """Student vLLM HTTP 서버 wrapper.
    verl의 AsyncLLMServerManager.generate()와 동일한 인터페이스를 제공."""

    def __init__(self, base_url: str, model_name: str, tokenizer):
        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str = "",
        **kwargs,
    ) -> TokenOutput:
        max_tokens = sampling_params.get("max_tokens", 256)
        temperature = sampling_params.get("temperature", 0.7)
        top_p = sampling_params.get("top_p", 0.95)

        payload = {
            "model": self.model_name,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        resp = await self.client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        generated_text = choice["text"]
        finish_reason = choice["finish_reason"]

        # text → token ids 변환
        token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        return TokenOutput(
            token_ids=token_ids,
            finish_reason=finish_reason,
        )


@dataclass
class TeacherLogprobOutput:
    """Teacher의 prompt_logprobs 결과.
    verl의 compute_teacher_logprobs_single() 반환값과 동일한 인터페이스."""
    ids: list[list[int]]           # [seq_len, K]
    logprobs: list[list[float]]    # [seq_len, K]


class HttpTeacherServer:
    """Teacher vLLM HTTP 서버 wrapper.
    verl의 AsyncTeacherLLMServerManager.compute_teacher_logprobs_single()과
    동일한 인터페이스를 제공."""

    def __init__(self, base_url: str, model_name: str, tokenizer, loss_top_k: int = 128):
        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.loss_top_k = loss_top_k
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        **kwargs,
    ) -> TeacherLogprobOutput:
        payload = {
            "model": self.model_name,
            "prompt": sequence_ids,
            "max_tokens": 1,
            "temperature": 1.0,  # prompt_logprobs requires temperature=1.0 in vLLM
            "prompt_logprobs": self.loss_top_k,  # vLLM 확장 파라미터: 최상위 필드로 전달
        }

        resp = await self.client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # prompt_logprobs 파싱
        # vLLM OpenAI API 응답: choices[0].prompt_logprobs = list of (dict or None)
        # 각 dict: {token_id_str: {logprob: float, rank: int, ...}, ...}
        prompt_logprobs_raw = data["choices"][0].get("prompt_logprobs", [])

        all_ids = []
        all_logprobs = []

        # Match verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs:
        # skip the first None entry and append one final dummy row.
        for entry in prompt_logprobs_raw[1:]:
            ranked = []
            for token_id_str, info in entry.items():
                if isinstance(info, dict):
                    rank = info.get("rank", 9999)
                    logprob = info.get("logprob", -100.0)
                else:
                    rank = 9999
                    logprob = float(info) if isinstance(info, (int, float)) else -100.0
                ranked.append((rank, int(token_id_str), logprob))

            ranked.sort(key=lambda x: x[0])
            ranked = ranked[:self.loss_top_k]

            ids_at_pos = [r[1] for r in ranked]
            lps_at_pos = [r[2] for r in ranked]

            while len(ids_at_pos) < self.loss_top_k:
                ids_at_pos.append(0)
                lps_at_pos.append(-100.0)

            all_ids.append(ids_at_pos)
            all_logprobs.append(lps_at_pos)

        all_ids.append([0] * self.loss_top_k)
        all_logprobs.append([0.0] * self.loss_top_k)

        return TeacherLogprobOutput(ids=all_ids, logprobs=all_logprobs)


# ============================================
# SKD 핵심 로직 (mock 테스트와 동일한 구조)
# ============================================


async def skd_generate(
    student_server: HttpStudentServer,
    teacher_server: HttpTeacherServer,
    tokenizer,
    prompt_ids: list[int],
    chunk_size: int,
    verify_top_k: int,
    max_response_length: int,
    eos_token_id: int = EOS_TOKEN_ID,
    sampling_params: dict[str, Any] = None,
) -> dict[str, Any]:
    """SKD 핵심 로직 — 실제 vLLM 서버 사용."""
    sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.95}

    accumulated_ids: list[int] = []
    accumulated_teacher_ids: list[list[int]] = []
    accumulated_teacher_logprobs: list[list[float]] = []
    metrics = {
        "accept_count": 0,
        "reject_count": 0,
        "chunk_count": 0,
        "chunk_details": [],
    }
    t_total_start = time.time()

    # ANSI color codes
    C_RESET = "\033[0m"
    C_GREEN = "\033[32m"       # accepted tokens
    C_RED_BG = "\033[41;37;1m" # rejected student token (red bg, white bold)
    C_CYAN_BG = "\033[46;30;1m"  # teacher replacement (cyan bg, black bold)
    C_DIM = "\033[2m"          # dim for meta info
    C_YELLOW = "\033[33m"      # chunk boundary marker

    print()
    print(f"  {C_DIM}[실시간 SKD 스트리밍]{C_RESET}")
    print(f"  {C_DIM}  {C_GREEN}초록{C_RESET}{C_DIM} = Student 수용  "
          f"{C_RED_BG}빨강{C_RESET}{C_DIM} = Student 거부  "
          f"{C_CYAN_BG}청록{C_RESET}{C_DIM} = Teacher 교체{C_RESET}")
    print(f"  {C_DIM}{'─' * 70}{C_RESET}")
    print()

    import sys

    while len(accumulated_ids) < max_response_length:
        current_prompt = prompt_ids + accumulated_ids
        remaining = max_response_length - len(accumulated_ids)
        actual_chunk_size = min(chunk_size, remaining)

        # 1. Student 청크 생성
        t0 = time.time()
        chunk_output = await student_server.generate(
            prompt_ids=current_prompt,
            sampling_params={**sampling_params, "max_tokens": actual_chunk_size},
        )
        chunk = chunk_output.token_ids
        t_student = time.time() - t0

        if not chunk:
            break

        # 2. Teacher 검증
        t0 = time.time()
        verify_sequence = current_prompt + chunk
        teacher_output = await teacher_server.compute_teacher_logprobs_single(
            sequence_ids=verify_sequence,
        )
        t_teacher = time.time() - t0

        # 3. Accept/Reject 판정
        chunk_start = len(current_prompt)
        rejection_pos = None
        rejected_student_token = None
        rejected_teacher_token = None
        for k in range(len(chunk)):
            if chunk_start + k - 1 < 0 or chunk_start + k - 1 >= len(teacher_output.ids):
                break
            teacher_topk = teacher_output.ids[chunk_start + k - 1][:verify_top_k]
            if chunk[k] not in teacher_topk:
                rejection_pos = k
                rejected_student_token = chunk[k]
                rejected_teacher_token = teacher_output.ids[chunk_start + k - 1][0]
                break

        # 4. 누적 + 실시간 토큰 출력
        if rejection_pos is not None:
            accepted = chunk[:rejection_pos]
            new_tokens = list(accepted) + [rejected_teacher_token]
            metrics["accept_count"] += len(accepted)
            metrics["reject_count"] += 1

            # 수용된 토큰 출력 (초록)
            if accepted:
                accepted_text = tokenizer.decode(accepted)
                sys.stdout.write(f"{C_GREEN}{accepted_text}{C_RESET}")

            # 거부된 토큰 (빨강 취소선 효과)
            student_tok_str = tokenizer.decode([rejected_student_token])
            sys.stdout.write(f"{C_RED_BG}{student_tok_str}{C_RESET}")

            # Teacher 교체 토큰 (청록)
            teacher_tok_str = tokenizer.decode([rejected_teacher_token])
            sys.stdout.write(f"{C_CYAN_BG}{teacher_tok_str}{C_RESET}")

            sys.stdout.flush()
        else:
            new_tokens = list(chunk)
            metrics["accept_count"] += len(chunk)

            # 전체 수용 (초록)
            accepted_text = tokenizer.decode(chunk)
            sys.stdout.write(f"{C_GREEN}{accepted_text}{C_RESET}")
            sys.stdout.flush()

        accumulated_ids.extend(new_tokens)

        # Teacher logprobs 누적
        for k in range(len(new_tokens)):
            pos = chunk_start + k - 1
            if 0 <= pos < len(teacher_output.ids):
                accumulated_teacher_ids.append(teacher_output.ids[pos])
                accumulated_teacher_logprobs.append(teacher_output.logprobs[pos])

        metrics["chunk_count"] += 1
        metrics["chunk_details"].append({
            "chunk_idx": metrics["chunk_count"],
            "chunk_len": len(chunk),
            "new_tokens": len(new_tokens),
            "accepted": len(new_tokens) - (1 if rejection_pos is not None else 0),
            "rejected": 1 if rejection_pos is not None else 0,
            "rejection_pos": rejection_pos,
            "rejected_student": tokenizer.decode([rejected_student_token]).strip() if rejected_student_token else None,
            "rejected_teacher": tokenizer.decode([rejected_teacher_token]).strip() if rejected_teacher_token else None,
            "student_time": round(t_student, 2),
            "teacher_time": round(t_teacher, 2),
            "accumulated_len": len(accumulated_ids),
        })

        # 5. 종료 조건
        if chunk_output.finish_reason == "stop":
            sys.stdout.write(f"\n\n  {C_DIM}[EOS]{C_RESET}\n")
            sys.stdout.flush()
            break
        if eos_token_id in new_tokens:
            sys.stdout.write(f"\n\n  {C_DIM}[EOS]{C_RESET}\n")
            sys.stdout.flush()
            break

    t_total = time.time() - t_total_start
    total_tokens = len(accumulated_ids)
    accept_rate = metrics["accept_count"] / total_tokens if total_tokens > 0 else 0
    total_student_time = sum(d["student_time"] for d in metrics["chunk_details"])
    total_teacher_time = sum(d["teacher_time"] for d in metrics["chunk_details"])

    # 최종 통계
    print(f"\n  {'=' * 75}")
    print(f"  SKD 생성 완료")
    print(f"  {'=' * 75}")
    print(f"    총 토큰:         {total_tokens}")
    print(f"    청크 수:         {metrics['chunk_count']}")
    print(f"    수용 토큰:       {metrics['accept_count']}  ({accept_rate:.1%})")
    print(f"    거부 토큰:       {metrics['reject_count']}  ({1 - accept_rate:.1%})")
    print(f"  {'─' * 75}")
    print(f"    총 시간:         {t_total:.1f}s")
    print(f"    Student 합산:    {total_student_time:.1f}s  ({total_student_time/t_total*100:.0f}%)")
    print(f"    Teacher 합산:    {total_teacher_time:.1f}s  ({total_teacher_time/t_total*100:.0f}%)")
    print(f"    Student tok/s:   {total_tokens / total_student_time:.0f}" if total_student_time > 0 else "")
    print(f"    Teacher tok/s:   {sum(d['accumulated_len'] + d['chunk_len'] for d in metrics['chunk_details']) / total_teacher_time:.0f} (prefill)" if total_teacher_time > 0 else "")
    print(f"  {'─' * 75}")
    if metrics["chunk_details"]:
        teacher_times = [d["teacher_time"] for d in metrics["chunk_details"]]
        print(f"    Teacher 시간 추이: {' → '.join(f'{t:.1f}s' for t in teacher_times)}")
        if len(teacher_times) >= 2:
            growth = teacher_times[-1] / teacher_times[0] if teacher_times[0] > 0 else 0
            print(f"    Teacher 시간 증가율: 첫 청크 {teacher_times[0]:.1f}s → 마지막 {teacher_times[-1]:.1f}s ({growth:.1f}×)")
            print(f"    (prefix cache 우회 영향: 시간이 청크마다 증가하면 재계산 발생)")

    return {
        "response_ids": accumulated_ids,
        "teacher_ids": accumulated_teacher_ids,
        "teacher_logprobs": accumulated_teacher_logprobs,
        "metrics": {
            **metrics,
            "total_tokens": total_tokens,
            "accept_rate": accept_rate,
        },
    }


# ============================================
# 테스트 실행
# ============================================


async def run_test(args):
    print("=" * 65)
    print("SKD 통합 테스트 (vLLM HTTP 서버)")
    print("=" * 65)

    # Tokenizer 로드
    print(f"\n[1/4] Tokenizer 로드: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")

    # 서버 health check
    print(f"\n[2/4] 서버 연결 확인")
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{STUDENT_URL}/health")
            print(f"  Student ({STUDENT_URL}): OK")
        except Exception as e:
            print(f"  Student ({STUDENT_URL}): FAIL — {e}")
            print("  서버를 먼저 띄워주세요: bash tests/skd/launch_test_servers.sh")
            return
        try:
            r = await client.get(f"{TEACHER_URL}/health")
            print(f"  Teacher ({TEACHER_URL}): OK")
        except Exception as e:
            print(f"  Teacher ({TEACHER_URL}): FAIL — {e}")
            return

    # 서버 wrapper 생성
    student = HttpStudentServer(STUDENT_URL, STUDENT_MODEL, tokenizer)
    # vLLM serve 기본 --max-logprobs=20. 서버 재시작 없이 테스트하기 위해 20 사용.
    # 실제 학습에서는 verl 내부 경로(Ray)로 128을 사용하며, 이 제한이 없음.
    teacher = HttpTeacherServer(TEACHER_URL, TEACHER_MODEL, tokenizer, loss_top_k=20)

    # 테스트 프롬프트 — AIME 2024-II-14 (정답: 211, 벤치마크에서 두 모델 모두 미해결)
    test_prompt = (
        r"Let $b \geq 2$ be an integer. Call a positive integer $n$ $b$\textit{-eautiful} "
        r"if it has exactly two digits when expressed in base $b$, and these two digits sum to $\sqrt{n}$. "
        r"For example, $81$ is $13$-eautiful because $81=\underline{6}\underline{3}_{13}$ and $6+3=\sqrt{81}$. "
        r"Find the least integer $b \geq 2$ for which there are more than ten $b$-eautiful integers. "
        r"Please reason step by step, and put your final answer within \boxed{}."
    )
    messages = [{"role": "user", "content": test_prompt}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print(f"\n[3/4] 테스트 프롬프트")
    print(f"  프롬프트: {test_prompt[:80]}...")
    print(f"  토큰 수: {len(prompt_ids)}")
    print(f"  chunk_size: {args.chunk_size}")
    print(f"  verify_top_k: {args.verify_top_k}")
    print(f"  max_response: {args.max_response}")

    # SKD 실행
    print(f"\n[4/4] SKD 생성 시작")
    t_start = time.time()

    result = await skd_generate(
        student_server=student,
        teacher_server=teacher,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        chunk_size=args.chunk_size,
        verify_top_k=args.verify_top_k,
        max_response_length=args.max_response,
        sampling_params={"temperature": 0.7, "top_p": 0.95},
    )

    t_total = time.time() - t_start

    # 결과 출력
    print(f"\n{'=' * 65}")
    print("결과")
    print(f"{'=' * 65}")

    response_text = tokenizer.decode(result["response_ids"], skip_special_tokens=True)
    print(f"\n  생성된 텍스트:")
    print(f"  {response_text[:500]}{'...' if len(response_text) > 500 else ''}")

    m = result["metrics"]
    print(f"\n  메트릭:")
    print(f"    총 토큰: {m['total_tokens']}")
    print(f"    청크 수: {m['chunk_count']}")
    print(f"    수용 토큰: {m['accept_count']}")
    print(f"    거부 토큰: {m['reject_count']}")
    print(f"    수용률: {m['accept_rate']:.1%}")
    print(f"    총 시간: {t_total:.1f}s")

    print(f"\n  청크별 상세:")
    print(f"    {'#':>4s} | {'chunk_len':>9s} | {'accepted':>8s} | {'rejected':>8s} | {'student':>8s} | {'teacher':>8s} | {'누적':>6s}")
    print(f"    {'':->4s}-+-{'':->9s}-+-{'':->8s}-+-{'':->8s}-+-{'':->8s}-+-{'':->8s}-+-{'':->6s}")
    for d in m["chunk_details"]:
        print(
            f"    {d['chunk_idx']:>4d} | {d['chunk_len']:>9d} | {d['accepted']:>8d} | {d['rejected']:>8d} | "
            f"{d['student_time']:>7.2f}s | {d['teacher_time']:>7.2f}s | {d['accumulated_len']:>6d}"
        )

    # 정합성 검증
    print(f"\n  정합성 검증:")
    assert len(result["teacher_ids"]) == len(result["response_ids"]), (
        f"teacher_ids({len(result['teacher_ids'])}) != response_ids({len(result['response_ids'])})"
    )
    print(f"    teacher_ids 길이 = response_ids 길이: {len(result['teacher_ids'])} ✓")

    assert len(result["teacher_logprobs"]) == len(result["response_ids"]), (
        f"teacher_logprobs({len(result['teacher_logprobs'])}) != response_ids({len(result['response_ids'])})"
    )
    print(f"    teacher_logprobs 길이 = response_ids 길이: {len(result['teacher_logprobs'])} ✓")

    for i, tid in enumerate(result["teacher_ids"]):
        assert len(tid) == 20, f"position {i}: teacher_ids length={len(tid)}, expected 20"
    print(f"    각 position의 teacher_ids 길이 = 20: ✓")

    print(f"\n  모든 검증 통과!")

    # cleanup
    await student.client.aclose()
    await teacher.client.aclose()


def main():
    parser = argparse.ArgumentParser(description="SKD Integration Test")
    parser.add_argument("--chunk-size", type=int, default=256, help="SKD 청크 크기")
    parser.add_argument("--verify-top-k", type=int, default=25, help="Teacher 검증 top-K")
    parser.add_argument("--max-response", type=int, default=1024, help="최대 응답 길이")
    args = parser.parse_args()

    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
