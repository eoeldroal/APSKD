# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import itertools
import logging
import math
import multiprocessing
import os
import threading
import time

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_GOLD_TARGETS = (LatexExtractionConfig(),)
_PRED_TARGETS = (ExprExtractionConfig(), LatexExtractionConfig())

# math_verify uses signal.alarm() for its own timeout path, which only works
# in a process main thread. We therefore isolate scoring in a lazily created
# process pool so child imports do not recursively create nested pools.
_PROCESS_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_PROCESS_POOL_LOCK = threading.Lock()
_DEFAULT_TIMEOUT_SECONDS = 30
_OUTER_TIMEOUT_GRACE_SECONDS = 1.0
_REWARD_DEBUG = os.getenv("VERL_REWARD_DEBUG", "0") == "1"
_REWARD_DEBUG_SLOW_MS = float(os.getenv("VERL_REWARD_DEBUG_SLOW_MS", "200"))
_REWARD_DEBUG_PREVIEW_CHARS = int(os.getenv("VERL_REWARD_DEBUG_PREVIEW_CHARS", "160"))
_REQUEST_COUNTER = itertools.count(1)
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_REQUESTS: dict[int, dict] = {}

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preview_text(text: str) -> str:
    preview = text.replace("\n", " ")
    if len(preview) <= _REWARD_DEBUG_PREVIEW_CHARS:
        return preview
    return preview[: _REWARD_DEBUG_PREVIEW_CHARS] + "..."


def _pool_queue_size() -> int:
    if _PROCESS_POOL is None:
        return 0
    work_queue = getattr(_PROCESS_POOL, "_call_queue", None)
    if work_queue is None or not hasattr(work_queue, "qsize"):
        return -1
    try:
        return work_queue.qsize()
    except Exception:
        return -1


def _get_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        with _PROCESS_POOL_LOCK:
            if _PROCESS_POOL is None:
                _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _PROCESS_POOL


def _register_request(req_id: int, model_output: str, ground_truth: str) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT_REQUESTS[req_id] = {
            "start_time": time.perf_counter(),
            "timed_out": False,
            "output_chars": len(model_output),
            "ground_truth": ground_truth,
            "has_boxed": "\\boxed{" in model_output,
            "has_tool": "<tool_call>" in model_output or "<tool_response>" in model_output,
            "preview": _preview_text(model_output),
        }


def _mark_request_timeout(req_id: int) -> None:
    with _INFLIGHT_LOCK:
        meta = _INFLIGHT_REQUESTS.get(req_id)
        if meta is not None:
            meta["timed_out"] = True


def _inflight_size() -> int:
    with _INFLIGHT_LOCK:
        return len(_INFLIGHT_REQUESTS)


def _on_future_done(req_id: int, future: concurrent.futures.Future) -> None:
    with _INFLIGHT_LOCK:
        meta = _INFLIGHT_REQUESTS.pop(req_id, None)
        inflight_after = len(_INFLIGHT_REQUESTS)

    if meta is None:
        return

    total_ms = (time.perf_counter() - meta["start_time"]) * 1000.0
    should_log = meta["timed_out"] or _REWARD_DEBUG or total_ms >= _REWARD_DEBUG_SLOW_MS
    if not should_log:
        return

    status = "ok"
    score = None
    error_type = None
    try:
        score = future.result()
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__

    logger.warning(
        "REWARD DEBUG math_verify.future_done req=%s status=%s total_ms=%.2f timed_out=%s inflight_after=%s queue=%s "
        "output_chars=%s has_boxed=%s has_tool=%s gt=%s score=%s error=%s preview=%s",
        req_id,
        status,
        total_ms,
        meta["timed_out"],
        inflight_after,
        _pool_queue_size(),
        meta["output_chars"],
        meta["has_boxed"],
        meta["has_tool"],
        meta["ground_truth"],
        score,
        error_type,
        meta["preview"],
    )


def _remaining_timeout_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutException("math_verify overall timeout exceeded before stage execution")
    # math_verify's timeout path is integer-second based. Passing floats like
    # 10.0 or 0.9 silently changes semantics and can yield empty parses / all-False
    # verification. Keep the budget integer and never return 0 while time remains.
    return max(1, int(math.floor(remaining)))


def _compute_score_inner(
    model_output: str,
    ground_truth_boxed: str,
    timeout_seconds: float | None,
    req_id: int | None = None,
) -> float:
    total_start = time.perf_counter()
    deadline = (total_start + timeout_seconds) if timeout_seconds is not None and timeout_seconds > 0 else None
    stage = "parse_gold"

    try:
        phase_start = time.perf_counter()
        extracted_gold = parse(ground_truth_boxed, _GOLD_TARGETS, parsing_timeout=_remaining_timeout_seconds(deadline))
        parse_gold_ms = (time.perf_counter() - phase_start) * 1000.0

        stage = "parse_pred"
        phase_start = time.perf_counter()
        extracted_pred = parse(model_output, _PRED_TARGETS, parsing_timeout=_remaining_timeout_seconds(deadline))
        parse_pred_ms = (time.perf_counter() - phase_start) * 1000.0

        stage = "verify"
        phase_start = time.perf_counter()
        score = 0.0
        if extracted_gold and extracted_pred:
            for p in extracted_pred:
                matched = False
                for g in extracted_gold:
                    if verify(g, p, timeout_seconds=_remaining_timeout_seconds(deadline)):
                        matched = True
                        break
                if matched:
                    score = 1.0
                    break
        verify_ms = (time.perf_counter() - phase_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        if _REWARD_DEBUG or total_ms >= _REWARD_DEBUG_SLOW_MS:
            logger.warning(
                "REWARD DEBUG math_verify.inner req=%s total_ms=%.2f parse_gold_ms=%.2f parse_pred_ms=%.2f verify_ms=%.2f "
                "n_gold=%s n_pred=%s output_chars=%s gt=%s preview=%s",
                req_id,
                total_ms,
                parse_gold_ms,
                parse_pred_ms,
                verify_ms,
                len(extracted_gold) if extracted_gold else 0,
                len(extracted_pred) if extracted_pred else 0,
                len(model_output),
                ground_truth_boxed,
                _preview_text(model_output),
            )

        return score
    except TimeoutException:
        total_ms = (time.perf_counter() - total_start) * 1000.0
        logger.warning(
            "REWARD DEBUG math_verify.inner_timeout req=%s total_ms=%.2f stage=%s output_chars=%s gt=%s preview=%s",
            req_id,
            total_ms,
            stage,
            len(model_output),
            ground_truth_boxed,
            _preview_text(model_output),
        )
        raise

def compute_score(
    model_output: str, ground_truth: str, timeout_score: float = 0, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> float:
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    req_id = next(_REQUEST_COUNTER)
    _register_request(req_id, model_output, ground_truth)
    try:
        future = _get_process_pool().submit(_compute_score_inner, model_output, ground_truth_boxed, timeout_seconds, req_id)
        future.add_done_callback(lambda f, request_id=req_id: _on_future_done(request_id, f))
        wait_timeout_seconds = None
        if timeout_seconds is not None and timeout_seconds > 0:
            wait_timeout_seconds = timeout_seconds + _OUTER_TIMEOUT_GRACE_SECONDS
        return future.result(timeout=wait_timeout_seconds)
    except concurrent.futures.TimeoutError:
        _mark_request_timeout(req_id)
        logger.warning(
            "REWARD DEBUG math_verify.timeout req=%s timeout_s=%s inflight=%s queue=%s output_chars=%s has_boxed=%s "
            "has_tool=%s gt=%s preview=%s",
            req_id,
            timeout_seconds,
            _inflight_size(),
            _pool_queue_size(),
            len(model_output),
            "\\boxed{" in model_output,
            "<tool_call>" in model_output or "<tool_response>" in model_output,
            ground_truth,
            _preview_text(model_output),
        )
        return timeout_score
    except TimeoutException:
        _mark_request_timeout(req_id)
        logger.warning(
            "REWARD DEBUG math_verify.timeout_exception req=%s inflight=%s queue=%s output_chars=%s gt=%s preview=%s",
            req_id,
            _inflight_size(),
            _pool_queue_size(),
            len(model_output),
            ground_truth,
            _preview_text(model_output),
        )
        return timeout_score
    except Exception:
        logger.exception(
            "REWARD DEBUG math_verify.exception req=%s inflight=%s queue=%s output_chars=%s gt=%s preview=%s",
            req_id,
            _inflight_size(),
            _pool_queue_size(),
            len(model_output),
            ground_truth,
            _preview_text(model_output),
        )
        return 0.0
