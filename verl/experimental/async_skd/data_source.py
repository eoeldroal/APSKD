"""Dataloader-backed sample source for bounded async SKD."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import copy
import uuid

import numpy as np

from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.protocol import DataProto


class AsyncSkdDataSource:
    """Convert collated dataloader batches into sample-level async-SKD units.

    The source keeps the existing StatefulDataLoader contract intact: it pulls
    one collated batch dict at a time, converts it with ``DataProto.from_single_dict``,
    and then exposes single-sample ``DataProto`` slices.
    """

    def __init__(self, batch_iterator: Iterator[dict], *, uid_fn: Callable[[], str] | None = None):
        self._batch_iterator = iter(batch_iterator)
        self._uid_fn = uid_fn or (lambda: str(uuid.uuid4()))
        self._fresh_buffer: DataProto | None = None
        self._fresh_cursor = 0
        self._carryover_partials: list[SkdPartialState] = []
        self._trained_reserved_sample_ids: set[str] = set()

    @property
    def trained_reserved_sample_ids(self) -> set[str]:
        return set(self._trained_reserved_sample_ids)

    def _ensure_uid(self, batch: DataProto) -> None:
        if "uid" in batch.non_tensor_batch:
            return
        batch.non_tensor_batch["uid"] = np.array([self._uid_fn() for _ in range(len(batch))], dtype=object)

    def _load_next_fresh_buffer(self) -> bool:
        for batch_dict in self._batch_iterator:
            batch = DataProto.from_single_dict(batch_dict)
            self._ensure_uid(batch)
            if len(batch) == 0:
                continue
            self._fresh_buffer = batch
            self._fresh_cursor = 0
            return True
        self._fresh_buffer = None
        self._fresh_cursor = 0
        return False

    def _ensure_fresh_available(self) -> bool:
        if self._fresh_buffer is not None and self._fresh_cursor < len(self._fresh_buffer):
            return True
        return self._load_next_fresh_buffer()

    def pop_fresh_sample(self) -> DataProto | None:
        """Return one fresh sample as a single-sample DataProto, or None if exhausted."""
        if not self._ensure_fresh_available():
            return None
        assert self._fresh_buffer is not None
        sample = self._fresh_buffer[self._fresh_cursor : self._fresh_cursor + 1]
        self._fresh_cursor += 1
        return sample

    def reserve_lookahead(self, logical_step: int) -> tuple[str, DataProto] | None:
        """Reserve one future sample for lookahead execution."""
        del logical_step
        sample = self.pop_fresh_sample()
        if sample is None:
            return None
        sample_id = str(sample.non_tensor_batch["uid"][0])
        return sample_id, sample

    def record_promoted(self, samples: list[AsyncSkdSample]) -> None:
        """Record promoted lookahead samples as already trained reserved samples."""
        for sample in samples:
            sample.validate()
            self._trained_reserved_sample_ids.add(sample.sample_id)

    def record_carryover(self, partials: list[SkdPartialState]) -> None:
        self._carryover_partials.extend(copy.deepcopy(partials))

    def pop_carryover(self) -> SkdPartialState | None:
        if not self._carryover_partials:
            return None
        return self._carryover_partials.pop(0)

    def next_fresh_quota(self, base_batch_size: int) -> int:
        """Fresh quota for the next current step. Promoted samples do not reduce it."""
        return max(0, base_batch_size - len(self._carryover_partials))

    def next_current_batch(self, base_batch_size: int) -> tuple[list[SkdPartialState], DataProto | None]:
        """Build current work as carryover first, then fresh samples up to base_batch_size."""
        if len(self._carryover_partials) > base_batch_size:
            raise ValueError(
                f"carryover_count={len(self._carryover_partials)} exceeds base_batch_size={base_batch_size}"
            )

        carryover = self._carryover_partials
        self._carryover_partials = []
        fresh_quota = base_batch_size - len(carryover)

        fresh_samples = []
        for _ in range(fresh_quota):
            sample = self.pop_fresh_sample()
            if sample is None:
                break
            fresh_samples.append(sample)

        fresh_batch = DataProto.concat(fresh_samples) if fresh_samples else None
        return carryover, fresh_batch

    def state_dict(self) -> dict:
        return {
            "fresh_buffer": self._fresh_buffer,
            "fresh_cursor": self._fresh_cursor,
            "carryover_partials": copy.deepcopy(self._carryover_partials),
            "trained_reserved_sample_ids": sorted(self._trained_reserved_sample_ids),
        }

    def load_state_dict(self, state: dict) -> None:
        self._fresh_buffer = state.get("fresh_buffer")
        self._fresh_cursor = int(state.get("fresh_cursor", 0))
        self._carryover_partials = copy.deepcopy(state.get("carryover_partials", []))
        self._trained_reserved_sample_ids = set(state.get("trained_reserved_sample_ids", []))
