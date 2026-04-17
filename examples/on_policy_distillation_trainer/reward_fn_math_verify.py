import logging
import os
import time

from verl.utils.reward_score.math_verify import compute_score as _compute_score_math_verify

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_REWARD_DEBUG = os.getenv("VERL_REWARD_DEBUG", "0") == "1"
_REWARD_DEBUG_SLOW_MS = float(os.getenv("VERL_REWARD_DEBUG_SLOW_MS", "200"))
_REWARD_DEBUG_PREVIEW_CHARS = int(os.getenv("VERL_REWARD_DEBUG_PREVIEW_CHARS", "160"))


def _preview_text(text: str) -> str:
    preview = text.replace("\n", " ")
    if len(preview) <= _REWARD_DEBUG_PREVIEW_CHARS:
        return preview
    return preview[: _REWARD_DEBUG_PREVIEW_CHARS] + "..."


def compute_score_math_verify(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
):
    start_time = time.perf_counter()
    score = None
    try:
        score = _compute_score_math_verify(
            model_output=solution_str,
            ground_truth=ground_truth,
        )
        return score
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if _REWARD_DEBUG or elapsed_ms >= _REWARD_DEBUG_SLOW_MS:
            extra_info_dict = extra_info if isinstance(extra_info, dict) else {}
            logger.warning(
                "REWARD DEBUG wrapper total_ms=%.2f data_source=%s num_turns=%s output_chars=%s has_boxed=%s has_tool=%s "
                "gt=%s score=%s preview=%s",
                elapsed_ms,
                data_source,
                extra_info_dict.get("num_turns"),
                len(solution_str),
                "\\boxed{" in solution_str,
                "<tool_call>" in solution_str or "<tool_response>" in solution_str,
                ground_truth,
                score,
                _preview_text(solution_str),
            )
