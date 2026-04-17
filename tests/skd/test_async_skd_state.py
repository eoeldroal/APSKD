"""Unit tests for async SKD state envelopes."""

from __future__ import annotations

import numpy as np
import pytest

from verl.experimental.async_skd import AsyncSkdSample
from verl.experimental.async_skd.state import SkdCommittedUnit, SkdPartialState
from verl.protocol import DataProto


def make_single_batch() -> DataProto:
    return DataProto.from_dict(
        non_tensors={
            "index": np.array([3], dtype=object),
            "rollout_birth_version": np.array([7], dtype=object),
            "rollout_min_version": np.array([7], dtype=object),
            "rollout_max_version": np.array([8], dtype=object),
            "skd_committed_gen_chunks": np.array([2], dtype=object),
            "skd_committed_env_units": np.array([1], dtype=object),
            "skd_committed_prefix_tokens": np.array([128], dtype=object),
        },
        meta_info={"metrics": [{"generate_sequences": 1.0, "tool_calls": 2.0}]},
    )


def make_partial() -> SkdPartialState:
    return SkdPartialState(
        sample_id="sample-partial",
        logical_step=5,
        source_type="lookahead_carryover",
        agent_state="generating",
        last_committed_unit=SkdCommittedUnit.ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT.value,
        request_id="req-partial",
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=8,
        committed_gen_chunks=2,
        committed_env_units=1,
        committed_prefix_tokens=128,
        metrics={"generate_sequences": 1.0},
        extra_fields={
            "teacher_ids_list": [[1, 2]],
            "teacher_logprobs_list": [[-1.0, -2.0]],
        },
    )


def test_async_skd_sample_from_completed_copies_metadata_and_requires_completed():
    batch = make_single_batch()
    sample = AsyncSkdSample.from_completed(
        sample_id="sample-completed",
        logical_step=4,
        source_type="base_current",
        batch=batch,
        train_consume_version=8,
    )

    assert sample.require_completed() is batch
    assert sample.rollout_birth_version == 7
    assert sample.rollout_min_version == 7
    assert sample.rollout_max_version == 8
    assert sample.train_consume_version == 8
    assert sample.committed_gen_chunks == 2
    assert sample.committed_env_units == 1
    assert sample.committed_prefix_tokens == 128
    assert sample.metrics == {"generate_sequences": 1.0, "tool_calls": 2.0}

    with pytest.raises(ValueError, match="not partial"):
        sample.require_partial()


def test_async_skd_sample_from_partial_copies_metadata_and_requires_partial():
    partial = make_partial()
    sample = AsyncSkdSample.from_partial(partial_state=partial)

    assert sample.require_partial() is partial
    assert sample.sample_id == partial.sample_id
    assert sample.source_type == partial.source_type
    assert sample.logical_step == partial.logical_step
    assert sample.rollout_birth_version == partial.rollout_birth_version
    assert sample.rollout_min_version == partial.rollout_min_version
    assert sample.rollout_max_version == partial.rollout_max_version
    assert sample.committed_gen_chunks == partial.committed_gen_chunks
    assert sample.committed_env_units == partial.committed_env_units
    assert sample.committed_prefix_tokens == partial.committed_prefix_tokens

    with pytest.raises(ValueError, match="not completed"):
        sample.require_completed()


def test_async_skd_sample_rejects_invalid_completed_payload_combinations():
    partial = make_partial()

    with pytest.raises(ValueError, match="must not carry partial_state"):
        AsyncSkdSample(
            sample_id="bad-completed",
            kind="completed",
            source_type="base_current",
            logical_step=0,
            batch=make_single_batch(),
            partial_state=partial,
        ).validate()

    with pytest.raises(ValueError, match="single-sample DataProto"):
        AsyncSkdSample.from_completed(
            sample_id="bad-batch",
            logical_step=0,
            source_type="base_current",
            batch=DataProto.from_dict(non_tensors={"index": np.array([0, 1], dtype=object)}),
        )


def test_async_skd_sample_rejects_invalid_partial_envelope_mismatch():
    partial = make_partial()
    sample = AsyncSkdSample.from_partial(partial_state=partial)
    sample.committed_prefix_tokens += 1

    with pytest.raises(ValueError, match="committed_prefix_tokens mismatch"):
        sample.validate()


def test_async_skd_sample_rejects_unknown_source_type_and_non_resumable_unit():
    with pytest.raises(ValueError, match="source_type"):
        AsyncSkdSample.from_completed(
            sample_id="bad-source",
            logical_step=0,
            source_type="unknown",
            batch=make_single_batch(),
        )

    partial = make_partial()
    partial.last_committed_unit = "NOT_RESUMABLE"
    with pytest.raises(ValueError, match="not resumable"):
        AsyncSkdSample.from_partial(partial_state=partial)
