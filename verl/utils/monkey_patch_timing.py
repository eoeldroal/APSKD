import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _PYNVML_AVAILABLE = False


_MONKEY_PATCH_GPU_UTIL_ENABLED = os.getenv("VERL_MONKEY_PATCH_GPU_UTIL", "1") == "1"
_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS = float(os.getenv("VERL_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS", "200"))
_MONKEY_PATCH_GPU_UTIL_HISTORY_SEC = float(os.getenv("VERL_MONKEY_PATCH_GPU_UTIL_HISTORY_SEC", "900"))

_NVML_INIT_LOCK = threading.Lock()
_NVML_INITIALIZED = False


def _ensure_nvml_initialized() -> bool:
    global _NVML_INITIALIZED
    if not _PYNVML_AVAILABLE or not _MONKEY_PATCH_GPU_UTIL_ENABLED:
        return False
    if _NVML_INITIALIZED:
        return True
    with _NVML_INIT_LOCK:
        if _NVML_INITIALIZED:
            return True
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
    return True


def _resolve_gpu_indices() -> tuple[list[int], str]:
    if not _ensure_nvml_initialized():
        return [], "disabled"

    device_count = pynvml.nvmlDeviceGetCount()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return list(range(device_count)), "all"

    parsed_indices: list[int] = []
    parseable = True
    for item in visible_devices.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < device_count:
                parsed_indices.append(idx)
        else:
            parseable = False
            break

    if parseable and parsed_indices:
        return parsed_indices, ",".join(str(idx) for idx in parsed_indices)
    return list(range(device_count)), f"all_fallback_from_{visible_devices}"


@dataclass(frozen=True)
class MonkeyPatchTimingHandle:
    start_time: float
    capture_gpu: bool = False


@dataclass(frozen=True)
class _GpuUtilSummary:
    avg_gpu_util: float
    avg_mem_util: float
    sample_count: int
    device_scope: str


class _GlobalGpuUtilSampler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._indices, self._device_scope = _resolve_gpu_indices()
        self._handles = []
        if self._indices:
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in self._indices]
        max_samples = max(int((_MONKEY_PATCH_GPU_UTIL_HISTORY_SEC * 1000) / _MONKEY_PATCH_GPU_UTIL_SAMPLE_MS), 64)
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=max_samples)

    def _sample_once(self) -> None:
        if not self._handles:
            return
        gpu_utils = []
        mem_utils = []
        for handle in self._handles:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utils.append(float(util.gpu))
            mem_utils.append(float(util.memory))
        timestamp = time.perf_counter()
        with self._lock:
            self._samples.append(
                (
                    timestamp,
                    sum(gpu_utils) / len(gpu_utils),
                    sum(mem_utils) / len(mem_utils),
                )
            )

    def _loop(self) -> None:
        interval_s = max(_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS / 1000.0, 0.05)
        self._sample_once()
        while not self._stop_event.wait(interval_s):
            self._sample_once()

    def start(self) -> None:
        if self._started or not self._handles:
            return
        self._started = True
        self._thread = threading.Thread(target=self._loop, name="verl-monkey-patch-gpu-util", daemon=True)
        self._thread.start()

    def average_between(self, start_time: float, end_time: float) -> _GpuUtilSummary | None:
        if not self._handles:
            return None
        with self._lock:
            selected = [sample for sample in self._samples if start_time <= sample[0] <= end_time]
        if not selected:
            return None
        avg_gpu = sum(sample[1] for sample in selected) / len(selected)
        avg_mem = sum(sample[2] for sample in selected) / len(selected)
        return _GpuUtilSummary(
            avg_gpu_util=avg_gpu,
            avg_mem_util=avg_mem,
            sample_count=len(selected),
            device_scope=self._device_scope,
        )


_GPU_UTIL_SAMPLER: _GlobalGpuUtilSampler | None = None


def _get_gpu_util_sampler() -> _GlobalGpuUtilSampler | None:
    global _GPU_UTIL_SAMPLER
    if not _MONKEY_PATCH_GPU_UTIL_ENABLED:
        return None
    if _GPU_UTIL_SAMPLER is None:
        _GPU_UTIL_SAMPLER = _GlobalGpuUtilSampler()
        _GPU_UTIL_SAMPLER.start()
    return _GPU_UTIL_SAMPLER


def monkey_patch_timing_begin(capture_gpu: bool = False) -> MonkeyPatchTimingHandle:
    if capture_gpu:
        _get_gpu_util_sampler()
    return MonkeyPatchTimingHandle(start_time=time.perf_counter(), capture_gpu=capture_gpu)


def monkey_patch_log_timing(logger, name: str, timing: float | MonkeyPatchTimingHandle, enabled: bool, slow_ms: float, **extra: Any) -> None:
    if not enabled:
        return

    end_time = time.perf_counter()
    if isinstance(timing, MonkeyPatchTimingHandle):
        start_time = timing.start_time
        capture_gpu = timing.capture_gpu
    else:
        start_time = timing
        capture_gpu = False

    elapsed_ms = (end_time - start_time) * 1000.0
    if elapsed_ms < slow_ms:
        return

    if capture_gpu:
        sampler = _get_gpu_util_sampler()
        if sampler is not None:
            summary = sampler.average_between(start_time, end_time)
            if summary is not None:
                extra = dict(extra)
                extra["avg_gpu_util"] = f"{summary.avg_gpu_util:.1f}"
                extra["avg_gpu_mem_util"] = f"{summary.avg_mem_util:.1f}"
                extra["gpu_samples"] = summary.sample_count
                extra["gpu_scope"] = summary.device_scope

    extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
    if extra_str:
        logger.warning("MONKEY PATCH timing %s: %.2f ms (%s)", name, elapsed_ms, extra_str)
    else:
        logger.warning("MONKEY PATCH timing %s: %.2f ms", name, elapsed_ms)
