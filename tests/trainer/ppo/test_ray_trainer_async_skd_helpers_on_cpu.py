from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.data_source import AsyncSkdDataSource
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeDataLoader:
    def __init__(self, batches: list[dict], sampler=None):
        self._batches = list(batches)
        self.sampler = sampler

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _UidFactory:
    def __init__(self):
        self._next = 0

    def __call__(self) -> str:
        value = f"uid-{self._next}"
        self._next += 1
        return value


class _FakeRolloutManager:
    def __init__(self):
        self.source = None

    def set_async_skd_data_source(self, source):
        self.source = source


def _batch_dict(start: int, count: int) -> dict:
    values = list(range(start, start + count))
    return {
        "dummy_tensor": torch.tensor([[value] for value in values], dtype=torch.uint8),
        "input_pos": np.array(values, dtype=object),
        "raw_prompt": np.array([[{"role": "user", "content": f"q-{value}"}] for value in values], dtype=object),
        "reward_model": np.array([{"ground_truth": str(value)} for value in values], dtype=object),
    }


def _completed(sample_id: str, batch: DataProto) -> AsyncSkdSample:
    return AsyncSkdSample.from_completed(
        sample_id=sample_id,
        logical_step=4,
        source_type="lookahead",
        batch=batch,
    )


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


def _single_input_row(value: int, uid: str) -> DataProto:
    batch = DataProto.from_single_dict(_batch_dict(value, 1))
    batch.non_tensor_batch["uid"] = np.array([uid], dtype=object)
    return batch


def _make_trainer(
    *,
    mode: str = "sync",
    rollout_n: int = 1,
    adv_estimator: str = "grpo",
    skip_enable: bool = False,
    batches: list[dict] | None = None,
):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "gen_batch_size": 2,
                "train_batch_size": 2,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "n": rollout_n,
                    "agent": {"async_skd_mode": mode},
                    "skip": {"enable": skip_enable},
                }
            },
            "algorithm": {"adv_estimator": adv_estimator},
            "trainer": {"total_epochs": 2},
        }
    )
    trainer.train_dataloader = _FakeDataLoader(batches or [_batch_dict(0, 2)])
    trainer.async_rollout_manager = _FakeRolloutManager()
    return trainer


def test_ensure_batch_uid_preserves_existing_uid_and_adds_missing_uid():
    trainer = _make_trainer()
    batch = DataProto.from_single_dict(_batch_dict(0, 2))
    batch.non_tensor_batch["uid"] = np.array(["keep-0", "keep-1"], dtype=object)

    trainer._ensure_batch_uid(batch)

    assert batch.non_tensor_batch["uid"].tolist() == ["keep-0", "keep-1"]

    missing_uid_batch = DataProto.from_single_dict(_batch_dict(2, 2))
    trainer._ensure_batch_uid(missing_uid_batch)

    assert "uid" in missing_uid_batch.non_tensor_batch
    assert len(missing_uid_batch.non_tensor_batch["uid"]) == 2
    assert all(isinstance(uid, str) for uid in missing_uid_batch.non_tensor_batch["uid"])


def test_iter_training_batches_uses_existing_dataloader_path_when_lookahead_disabled():
    trainer = _make_trainer(mode="sync", batches=[_batch_dict(0, 2)])

    [(carryover, fresh_batch, current_input_batch)] = list(trainer._iter_training_batches())

    assert carryover == []
    assert fresh_batch.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert current_input_batch.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert trainer.async_rollout_manager.source is None


def test_iter_training_batches_lookahead_uses_async_skd_source_and_binds_manager():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])

    [(carryover, fresh_batch, current_input_batch)] = list(trainer._iter_training_batches())

    assert carryover == []
    assert fresh_batch.non_tensor_batch["input_pos"].tolist() == [10, 11]
    assert current_input_batch.non_tensor_batch["input_pos"].tolist() == [10, 11]
    assert trainer.async_rollout_manager.source is trainer._async_skd_data_source


def test_iter_training_batches_lookahead_continues_across_epoch_calls():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])

    first_epoch_batches = list(trainer._iter_training_batches())
    second_epoch_batches = list(trainer._iter_training_batches())

    assert len(first_epoch_batches) == 1
    assert len(second_epoch_batches) == 1
    assert first_epoch_batches[0][2].non_tensor_batch["input_pos"].tolist() == [10, 11]
    assert second_epoch_batches[0][2].non_tensor_batch["input_pos"].tolist() == [10, 11]


def test_iter_training_batches_lookahead_preserves_carryover_when_epoch_iterator_advances():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])

    list(trainer._iter_training_batches())
    source = trainer._async_skd_data_source
    source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(100, "carry-0")],
    )

    [(carryover, fresh_batch, current_input_batch)] = list(trainer._iter_training_batches())

    assert [partial.sample_id for partial in carryover] == ["carry-0"]
    assert fresh_batch is not None
    assert fresh_batch.non_tensor_batch["input_pos"].tolist() == [10]
    assert current_input_batch.non_tensor_batch["input_pos"].tolist() == [100, 10]


def test_async_skd_checkpoint_payload_preserves_source_state():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(100, 2)])), uid_fn=_UidFactory())
    source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(200, "carry-0")],
    )
    trainer._async_skd_data_source = source

    payload = trainer._build_dataloader_checkpoint_state({"loader": "state"})
    dataloader_state, source_state = trainer._split_dataloader_checkpoint_state(payload)

    assert dataloader_state == {"loader": "state"}
    assert source_state is not None
    assert [partial.sample_id for partial in source_state["carryover_partials"]] == ["carry-0"]


def test_async_skd_source_state_pending_work_detection():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])
    empty_source = AsyncSkdDataSource(iter(_FakeDataLoader([])), uid_fn=_UidFactory())
    pending_source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(100, 2)])), uid_fn=_UidFactory())
    pending_source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(200, "carry-0")],
    )

    assert not trainer._async_skd_source_state_has_pending_work(empty_source.state_dict())
    assert trainer._async_skd_source_state_has_pending_work(pending_source.state_dict())


def test_ensure_async_skd_data_source_loads_pending_checkpoint_state():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(100, 2)])), uid_fn=_UidFactory())
    source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(200, "carry-0")],
    )
    _, source_state = trainer._split_dataloader_checkpoint_state(
        trainer._build_dataloader_checkpoint_state({"loader": "state"}, async_skd_source=source)
    )
    trainer._async_skd_data_source_state_to_load = source_state

    restored = trainer._ensure_async_skd_data_source()

    carryover, fresh_batch, current_input_batch = restored.next_current_batch(base_batch_size=2)
    assert [partial.sample_id for partial in carryover] == ["carry-0"]
    assert fresh_batch is not None
    assert current_input_batch.non_tensor_batch["input_pos"].tolist() == [200, 10]


def test_prepare_async_skd_current_input_batch_matches_fresh_generation_batch_keys():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(10, 4)])), uid_fn=_UidFactory())
    carryover, fresh_batch, current_input_batch = source.next_current_batch(base_batch_size=2)
    assert carryover == []
    assert fresh_batch is not None
    assert current_input_batch is not None
    current_input_batch.non_tensor_batch["extra_generation_key"] = np.array(["x", "y"], dtype=object)
    fresh_batch.non_tensor_batch["extra_generation_key"] = np.array(["x", "y"], dtype=object)

    gen_batch = trainer._get_gen_batch(fresh_batch)
    trainer._prepare_async_skd_current_input_batch(current_input_batch)

    assert "extra_generation_key" in gen_batch.non_tensor_batch
    assert "extra_generation_key" not in current_input_batch.non_tensor_batch
    assert current_input_batch.non_tensor_batch.keys() == {"reward_model", "uid"}


def test_prepare_async_skd_current_input_batch_handles_carryover_and_fresh_rows_together():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 2)])
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(10, 4)])), uid_fn=_UidFactory())
    source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(100, "carry-0")],
    )
    carryover, fresh_batch, current_input_batch = source.next_current_batch(base_batch_size=2)
    assert [partial.sample_id for partial in carryover] == ["carry-0"]
    assert fresh_batch is not None
    assert current_input_batch is not None
    current_input_batch.non_tensor_batch["extra_generation_key"] = np.array(["carry", "fresh"], dtype=object)
    fresh_batch.non_tensor_batch["extra_generation_key"] = np.array(["fresh"], dtype=object)

    gen_batch = trainer._get_gen_batch(fresh_batch)
    trainer._prepare_async_skd_current_input_batch(current_input_batch)

    assert gen_batch.non_tensor_batch["input_pos"].tolist() == [10]
    assert current_input_batch.non_tensor_batch["uid"].tolist() == ["carry-0", "uid-0"]
    assert current_input_batch.non_tensor_batch.keys() == {"reward_model", "uid"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rollout_n": 2}, "rollout.n == 1"),
        ({"adv_estimator": "remax"}, "REMAX"),
        ({"skip_enable": True}, "rollout skip"),
    ],
)
def test_iter_training_batches_lookahead_rejects_unsupported_modes(kwargs, match):
    trainer = _make_trainer(mode="lookahead", **kwargs)

    with pytest.raises(ValueError, match=match):
        list(trainer._iter_training_batches())


def test_append_async_skd_promoted_inputs_extends_training_batch_once():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(0, 2)])
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(100, 2)])), uid_fn=_UidFactory())
    trainer._async_skd_data_source = source

    reserved_0 = source.reserve_lookahead(logical_step=1)
    reserved_1 = source.reserve_lookahead(logical_step=1)
    assert reserved_0 is not None and reserved_1 is not None
    sample_id_0, sample_0 = reserved_0
    sample_id_1, sample_1 = reserved_1
    source.record_promoted([
        _completed(sample_id_0, sample_0),
        _completed(sample_id_1, sample_1),
    ])

    batch = DataProto.from_single_dict(_batch_dict(0, 2))
    batch.non_tensor_batch["uid"] = np.array(["base-0", "base-1"], dtype=object)
    trainer._get_gen_batch(batch)

    extended = trainer._append_async_skd_promoted_inputs(batch)

    assert len(extended) == 4
    assert trainer._async_skd_last_promoted_rows_appended == 2
    assert extended.non_tensor_batch["uid"].tolist() == ["base-0", "base-1", sample_id_0, sample_id_1]
    assert [item["ground_truth"] for item in extended.non_tensor_batch["reward_model"]] == [
        "0",
        "1",
        "100",
        "101",
    ]
    assert extended.batch["dummy_tensor"].squeeze(-1).tolist() == [0, 1, 100, 101]

    unchanged = trainer._append_async_skd_promoted_inputs(extended)

    assert len(unchanged) == 4
    assert trainer._async_skd_last_promoted_rows_appended == 0


def test_record_async_skd_current_batch_metrics_reports_quota_and_row_counts():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(10, 4)])
    trainer.global_steps = 7
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(10, 4)])), uid_fn=_UidFactory())
    trainer._async_skd_data_source = source
    source.record_carryover(
        [_partial("carry-0")],
        input_batches=[_single_input_row(100, "carry-0")],
    )
    carryover, fresh_batch, current_input_batch = source.next_current_batch(base_batch_size=2)
    assert fresh_batch is not None
    assert current_input_batch is not None
    metrics = {}

    trainer._record_async_skd_current_batch_metrics(
        metrics,
        carryover_partials=carryover,
        fresh_batch=fresh_batch,
        current_input_batch=current_input_batch,
    )

    assert metrics["async_skd/current_base_batch_size"] == 2
    assert metrics["async_skd/current_carryover_count"] == 1
    assert metrics["async_skd/current_fresh_quota"] == 1
    assert metrics["async_skd/current_fresh_count"] == 1
    assert metrics["async_skd/current_input_batch_size"] == 2


def test_record_async_skd_post_generation_and_union_metrics():
    trainer = _make_trainer(mode="lookahead", batches=[_batch_dict(0, 2)])
    trainer.global_steps = 8
    source = AsyncSkdDataSource(iter(_FakeDataLoader([_batch_dict(100, 2)])), uid_fn=_UidFactory())
    trainer._async_skd_data_source = source
    reserved_0 = source.reserve_lookahead(logical_step=1)
    reserved_1 = source.reserve_lookahead(logical_step=1)
    assert reserved_0 is not None and reserved_1 is not None
    sample_id_0, sample_0 = reserved_0
    sample_id_1, sample_1 = reserved_1
    source.record_promoted([
        _completed(sample_id_0, sample_0),
        _completed(sample_id_1, sample_1),
    ])
    batch = DataProto.from_single_dict(_batch_dict(0, 2))
    batch.non_tensor_batch["uid"] = np.array(["base-0", "base-1"], dtype=object)
    trainer._get_gen_batch(batch)
    gen_batch_output = DataProto.from_dict(
        tensors={"gen_tensor": torch.arange(4, dtype=torch.long).unsqueeze(-1)},
        meta_info={"async_skd_metrics": {"async_skd/lookahead_promoted_count": 2}},
    )
    metrics = {}

    train_before_union = trainer._record_async_skd_post_generation_metrics(
        metrics,
        batch=batch,
        gen_batch_output=gen_batch_output,
    )
    final_batch = train_before_union.union(gen_batch_output)
    trainer._record_async_skd_union_metrics(
        metrics,
        train_batch_before_union=train_before_union,
        gen_batch_output=gen_batch_output,
        final_batch=final_batch,
    )

    assert metrics["async_skd/lookahead_promoted_count"] == 2
    assert metrics["async_skd/gen_batch_output_size"] == 4
    assert metrics["async_skd/promoted_rows_appended"] == 2
    assert metrics["async_skd/train_batch_size_before_union"] == 4
    assert metrics["async_skd/union_row_delta"] == 0
    assert metrics["async_skd/final_train_batch_size"] == 4
