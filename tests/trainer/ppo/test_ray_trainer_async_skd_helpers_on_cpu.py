from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.data_source import AsyncSkdDataSource
from verl.experimental.async_skd.state import AsyncSkdSample
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
