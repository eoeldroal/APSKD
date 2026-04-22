"""Unit tests for async SKD dataloader-backed sample source."""

from __future__ import annotations

import numpy as np
import torch

from verl.experimental.async_skd.data_source import AsyncSkdDataSource
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.protocol import DataProto


class _UidFactory:
    def __init__(self):
        self._next = 0

    def __call__(self) -> str:
        value = f"uid-{self._next}"
        self._next += 1
        return value


class _BatchIterator:
    def __init__(self, batches: list[dict]):
        self._batches = list(batches)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._batches:
            raise StopIteration
        self.pulled += 1
        return self._batches.pop(0)


def _batch_dict(start: int, count: int) -> dict:
    values = list(range(start, start + count))
    return {
        "dummy_tensor": torch.tensor([[value] for value in values], dtype=torch.uint8),
        "input_pos": np.array(values, dtype=object),
        "raw_prompt": np.array([[{"role": "user", "content": f"q-{value}"}] for value in values], dtype=object),
        "reward_model": np.array([{"ground_truth": str(value)} for value in values], dtype=object),
    }


def _single_input_row(value: int, uid: str) -> DataProto:
    batch = DataProto.from_single_dict(_batch_dict(value, 1))
    batch.non_tensor_batch["uid"] = np.array([uid], dtype=object)
    return batch


def _partial(sample_id: str) -> SkdPartialState:
    return SkdPartialState(
        sample_id=sample_id,
        logical_step=4,
        source_type="lookahead",
        agent_state="generating",
        request_id=f"req-{sample_id}",
        response_ids=[1],
        response_mask=[1],
        extra_fields={
            "teacher_ids_list": [[1, 0, 0, 0]],
            "teacher_logprobs_list": [[-1.0, 0.0, 0.0, 0.0]],
        },
    )


def _completed(sample_id: str, batch: DataProto) -> AsyncSkdSample:
    return AsyncSkdSample.from_completed(
        sample_id=sample_id,
        logical_step=4,
        source_type="lookahead",
        batch=batch,
    )


def test_data_source_lazily_converts_batch_dicts_to_single_sample_dataproto():
    iterator = _BatchIterator([_batch_dict(0, 2), _batch_dict(2, 2)])
    source = AsyncSkdDataSource(iterator, uid_fn=_UidFactory())

    first = source.pop_fresh_sample()
    assert first is not None
    assert first.non_tensor_batch["input_pos"].tolist() == [0]
    assert first.non_tensor_batch["uid"].tolist() == ["uid-0"]
    assert first.batch["dummy_tensor"].shape == (1, 1)
    assert iterator.pulled == 1

    second = source.pop_fresh_sample()
    assert second is not None
    assert second.non_tensor_batch["input_pos"].tolist() == [1]
    assert second.non_tensor_batch["uid"].tolist() == ["uid-1"]
    assert iterator.pulled == 1

    third = source.pop_fresh_sample()
    assert third is not None
    assert third.non_tensor_batch["input_pos"].tolist() == [2]
    assert third.non_tensor_batch["uid"].tolist() == ["uid-2"]
    assert iterator.pulled == 2


def test_data_source_builds_current_batch_with_carryover_first_and_fresh_quota():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(10, 4)]), uid_fn=_UidFactory())
    source.record_carryover(
        [_partial("carry-0"), _partial("carry-1")],
        input_batches=[_single_input_row(100, "carry-0"), _single_input_row(101, "carry-1")],
    )

    carryover, fresh, current_input = source.next_current_batch(base_batch_size=4)

    assert [partial.sample_id for partial in carryover] == ["carry-0", "carry-1"]
    assert fresh is not None
    assert fresh.non_tensor_batch["input_pos"].tolist() == [10, 11]
    assert current_input is not None
    assert current_input.non_tensor_batch["input_pos"].tolist() == [100, 101, 10, 11]
    assert current_input.non_tensor_batch["uid"].tolist() == ["carry-0", "carry-1", "uid-0", "uid-1"]
    assert current_input.batch["dummy_tensor"].squeeze(-1).tolist() == [100, 101, 10, 11]
    assert source.next_fresh_quota(4) == 4


def test_data_source_uses_reserved_lookahead_input_rows_for_carryover_current_batch():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(0, 4)]), uid_fn=_UidFactory())

    reserved_0 = source.reserve_lookahead(logical_step=1)
    reserved_1 = source.reserve_lookahead(logical_step=1)
    assert reserved_0 is not None and reserved_1 is not None

    source.record_carryover([_partial("uid-0"), _partial("uid-1")])

    carryover, fresh, current_input = source.next_current_batch(base_batch_size=4)

    assert [partial.sample_id for partial in carryover] == ["uid-0", "uid-1"]
    assert fresh is not None
    assert fresh.non_tensor_batch["input_pos"].tolist() == [2, 3]
    assert current_input is not None
    assert current_input.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3]
    assert current_input.non_tensor_batch["uid"].tolist() == ["uid-0", "uid-1", "uid-2", "uid-3"]


def test_data_source_reserves_lookahead_and_records_promoted_without_reducing_fresh_quota():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(0, 4)]), uid_fn=_UidFactory())

    reserved = source.reserve_lookahead(logical_step=1)
    assert reserved is not None
    sample_id, sample = reserved
    assert sample_id == "uid-0"
    assert sample.non_tensor_batch["input_pos"].tolist() == [0]

    source.record_promoted([_completed(sample_id, sample)])
    source.record_carryover(
        [_partial("carry-0"), _partial("carry-1")],
        input_batches=[_single_input_row(100, "carry-0"), _single_input_row(101, "carry-1")],
    )

    assert source.next_fresh_quota(4) == 2
    carryover, fresh, current_input = source.next_current_batch(base_batch_size=4)
    assert [partial.sample_id for partial in carryover] == ["carry-0", "carry-1"]
    assert fresh is not None
    assert fresh.non_tensor_batch["input_pos"].tolist() == [1, 2]
    assert current_input is not None
    assert current_input.non_tensor_batch["input_pos"].tolist() == [100, 101, 1, 2]
    assert sample_id in source.trained_reserved_sample_ids


def test_data_source_returns_promoted_input_output_pairs_with_limit_in_order():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(0, 4)]), uid_fn=_UidFactory())

    reserved_0 = source.reserve_lookahead(logical_step=1)
    reserved_1 = source.reserve_lookahead(logical_step=1)
    assert reserved_0 is not None and reserved_1 is not None
    sample_id_0, sample_0 = reserved_0
    sample_id_1, sample_1 = reserved_1

    source.record_promoted([
        _completed(sample_id_0, sample_0),
        _completed(sample_id_1, sample_1),
    ])

    promoted_inputs, promoted_outputs = source.pop_promoted_pairs(max_count=1)

    assert [batch.non_tensor_batch["uid"].tolist()[0] for batch in promoted_inputs] == [sample_id_0]
    assert [batch.non_tensor_batch["input_pos"].tolist()[0] for batch in promoted_inputs] == [0]
    assert [batch.non_tensor_batch["uid"].tolist()[0] for batch in promoted_outputs] == [sample_id_0]
    assert [batch.non_tensor_batch["input_pos"].tolist()[0] for batch in promoted_outputs] == [0]

    promoted_inputs, promoted_outputs = source.pop_promoted_pairs(max_count=8)

    assert [batch.non_tensor_batch["uid"].tolist()[0] for batch in promoted_inputs] == [sample_id_1]
    assert [batch.non_tensor_batch["input_pos"].tolist()[0] for batch in promoted_outputs] == [1]
    assert source.pop_promoted_pairs(max_count=8) == ([], [])


def test_data_source_state_dict_restores_fresh_buffer_and_ledgers():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(0, 3)]), uid_fn=_UidFactory())
    first = source.pop_fresh_sample()
    assert first is not None
    source.record_promoted([_completed("uid-0", first)])
    source.record_carryover([_partial("carry-0")], input_batches=[_single_input_row(100, "carry-0")])

    restored = AsyncSkdDataSource(_BatchIterator([]), uid_fn=_UidFactory())
    restored.load_state_dict(source.state_dict())

    next_sample = restored.pop_fresh_sample()
    assert next_sample is not None
    assert next_sample.non_tensor_batch["input_pos"].tolist() == [1]
    assert restored.trained_reserved_sample_ids == {"uid-0"}

    carryover, fresh, current_input = restored.next_current_batch(base_batch_size=2)
    assert [partial.sample_id for partial in carryover] == ["carry-0"]
    assert fresh is not None
    assert fresh.non_tensor_batch["input_pos"].tolist() == [2]
    assert current_input is not None
    assert current_input.non_tensor_batch["input_pos"].tolist() == [100, 2]


def test_data_source_state_dict_restores_unconsumed_promoted_pairs():
    source = AsyncSkdDataSource(_BatchIterator([_batch_dict(0, 2)]), uid_fn=_UidFactory())
    reserved = source.reserve_lookahead(logical_step=1)
    assert reserved is not None
    sample_id, sample = reserved
    source.record_promoted([_completed(sample_id, sample)])

    restored = AsyncSkdDataSource(_BatchIterator([]), uid_fn=_UidFactory())
    restored.load_state_dict(source.state_dict())

    promoted_inputs, promoted_outputs = restored.pop_promoted_pairs(max_count=1)

    assert len(promoted_inputs) == 1
    assert len(promoted_outputs) == 1
    assert promoted_inputs[0].non_tensor_batch["uid"].tolist() == [sample_id]
    assert promoted_inputs[0].non_tensor_batch["input_pos"].tolist() == [0]
    assert promoted_outputs[0].non_tensor_batch["uid"].tolist() == [sample_id]
    assert promoted_outputs[0].non_tensor_batch["input_pos"].tolist() == [0]
