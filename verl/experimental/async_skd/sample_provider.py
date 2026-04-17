"""Sample reservation utilities for bounded asynchronous SKD."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class AsyncSkdSampleProvider:
    """Sample-level provider that sits above an existing dataloader/sampler.

    The provider deliberately does not alter sampler semantics.  It consumes
    items sequentially from the supplied iterable and adds the accounting needed
    by bounded lookahead: lookahead reservations consume future samples now, and
    promoted/carryover samples reduce the next step's fresh quota.
    """

    def __init__(self, samples: Iterable[Any], base_batch_size: int):
        if base_batch_size <= 0:
            raise ValueError(f"base_batch_size must be positive, got {base_batch_size}")
        self._samples = list(samples)
        self.base_batch_size = int(base_batch_size)
        self._cursor = 0
        self._next_fresh_quota = self.base_batch_size
        self._current_step: int | None = None
        self._lookahead_reserved_by_step: dict[int, list[Any]] = {}

    @property
    def cursor(self) -> int:
        return self._cursor

    def _take(self, count: int) -> list[Any]:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        end = self._cursor + count
        if end > len(self._samples):
            raise StopIteration(
                f"AsyncSkdSampleProvider exhausted: requested={count}, remaining={len(self._samples) - self._cursor}"
            )
        out = self._samples[self._cursor : end]
        self._cursor = end
        return out

    def next_base_samples(self, step_id: int) -> list[Any]:
        """Return fresh samples for ``step_id`` according to the current quota."""
        if self._current_step is not None and step_id <= self._current_step:
            raise ValueError(f"step_id must increase monotonically: prev={self._current_step}, got={step_id}")
        self._current_step = step_id
        quota = self._next_fresh_quota
        self._next_fresh_quota = self.base_batch_size
        return self._take(quota)

    def reserve_lookahead_samples(self, step_id: int, count: int) -> list[Any]:
        """Reserve future samples for ``step_id + 1`` during ``step_id``.

        The provider enforces one-step lookahead only.  The reserved samples are
        consumed from the same sequence as fresh samples, so they cannot appear
        again in a later fresh batch.
        """
        if self._current_step is None:
            raise RuntimeError("next_base_samples() must be called before reserving lookahead samples")
        if step_id != self._current_step:
            raise ValueError(f"can only reserve from current step={self._current_step}, got step_id={step_id}")
        target_step = step_id + 1
        reserved = self._take(count)
        self._lookahead_reserved_by_step.setdefault(target_step, []).extend(reserved)
        return reserved

    def set_next_step_accounting(self, *, promoted_count: int, carryover_count: int) -> None:
        """Set fresh quota for the next step after promotion/carryover accounting."""
        if promoted_count < 0 or carryover_count < 0:
            raise ValueError(
                f"counts must be non-negative, promoted={promoted_count}, carryover={carryover_count}"
            )
        quota = self.base_batch_size - promoted_count - carryover_count
        if quota < 0:
            raise ValueError(
                "promoted_count + carryover_count cannot exceed base_batch_size: "
                f"{promoted_count} + {carryover_count} > {self.base_batch_size}"
            )
        self._next_fresh_quota = quota

    def next_step_fresh_quota(self) -> int:
        return self._next_fresh_quota

    def lookahead_reserved_for_step(self, step_id: int) -> list[Any]:
        return list(self._lookahead_reserved_by_step.get(step_id, []))

    def state_dict(self) -> dict[str, Any]:
        return {
            "samples": list(self._samples),
            "base_batch_size": self.base_batch_size,
            "cursor": self._cursor,
            "next_fresh_quota": self._next_fresh_quota,
            "current_step": self._current_step,
            "lookahead_reserved_by_step": {
                int(step): list(samples) for step, samples in self._lookahead_reserved_by_step.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._samples = list(state["samples"])
        self.base_batch_size = int(state["base_batch_size"])
        self._cursor = int(state["cursor"])
        self._next_fresh_quota = int(state["next_fresh_quota"])
        self._current_step = state["current_step"]
        self._lookahead_reserved_by_step = {
            int(step): list(samples) for step, samples in state["lookahead_reserved_by_step"].items()
        }
