"""Unit tests for async SKD sample reservation accounting."""

from __future__ import annotations

import pytest

from verl.experimental.async_skd import AsyncSkdSampleProvider


def make_samples(n: int = 20) -> list[str]:
    return [f"s{i}" for i in range(n)]


def test_provider_reserves_lookahead_and_reduces_next_fresh_quota():
    provider = AsyncSkdSampleProvider(make_samples(), base_batch_size=4)

    assert provider.next_base_samples(step_id=0) == ["s0", "s1", "s2", "s3"]
    assert provider.cursor == 4

    reserved = provider.reserve_lookahead_samples(step_id=0, count=2)
    assert reserved == ["s4", "s5"]
    assert provider.lookahead_reserved_for_step(1) == ["s4", "s5"]
    assert provider.cursor == 6

    provider.set_next_step_accounting(promoted_count=1, carryover_count=1)
    assert provider.next_step_fresh_quota() == 2

    assert provider.next_base_samples(step_id=1) == ["s6", "s7"]
    assert provider.cursor == 8

    provider.set_next_step_accounting(promoted_count=0, carryover_count=0)
    assert provider.next_base_samples(step_id=2) == ["s8", "s9", "s10", "s11"]


def test_provider_rejects_non_monotonic_step_and_non_current_lookahead():
    provider = AsyncSkdSampleProvider(make_samples(), base_batch_size=3)

    assert provider.next_base_samples(step_id=5) == ["s0", "s1", "s2"]

    with pytest.raises(ValueError, match="step_id must increase monotonically"):
        provider.next_base_samples(step_id=5)

    with pytest.raises(ValueError, match="can only reserve from current step"):
        provider.reserve_lookahead_samples(step_id=6, count=1)


def test_provider_requires_base_before_lookahead_reservation():
    provider = AsyncSkdSampleProvider(make_samples(), base_batch_size=4)

    with pytest.raises(RuntimeError, match="next_base_samples"):
        provider.reserve_lookahead_samples(step_id=0, count=1)


def test_provider_rejects_invalid_quota_accounting():
    provider = AsyncSkdSampleProvider(make_samples(), base_batch_size=4)

    with pytest.raises(ValueError, match="non-negative"):
        provider.set_next_step_accounting(promoted_count=-1, carryover_count=0)

    with pytest.raises(ValueError, match="cannot exceed base_batch_size"):
        provider.set_next_step_accounting(promoted_count=3, carryover_count=2)


def test_provider_state_dict_roundtrip_preserves_cursor_quota_and_reservations():
    provider = AsyncSkdSampleProvider(make_samples(), base_batch_size=4)

    assert provider.next_base_samples(step_id=0) == ["s0", "s1", "s2", "s3"]
    assert provider.reserve_lookahead_samples(step_id=0, count=1) == ["s4"]
    provider.set_next_step_accounting(promoted_count=1, carryover_count=0)

    restored = AsyncSkdSampleProvider([], base_batch_size=1)
    restored.load_state_dict(provider.state_dict())

    assert restored.cursor == 5
    assert restored.next_step_fresh_quota() == 3
    assert restored.lookahead_reserved_for_step(1) == ["s4"]
    assert restored.next_base_samples(step_id=1) == ["s5", "s6", "s7"]


def test_provider_exhaustion_is_explicit():
    provider = AsyncSkdSampleProvider(make_samples(3), base_batch_size=2)

    assert provider.next_base_samples(step_id=0) == ["s0", "s1"]
    with pytest.raises(StopIteration, match="exhausted"):
        provider.next_base_samples(step_id=1)
