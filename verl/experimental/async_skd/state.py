"""Serializable state containers for bounded asynchronous SKD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.protocol import DataProto


ASYNC_SKD_SAMPLE_KIND_COMPLETED = "completed"
ASYNC_SKD_SAMPLE_KIND_PARTIAL = "partial"
ASYNC_SKD_SAMPLE_KINDS: frozenset[str] = frozenset(
    {
        ASYNC_SKD_SAMPLE_KIND_COMPLETED,
        ASYNC_SKD_SAMPLE_KIND_PARTIAL,
    }
)
ASYNC_SKD_SAMPLE_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "base_current",
        "lookahead",
        "lookahead_promoted",
        "lookahead_carryover",
        "resumed_current",
    }
)


def _single_non_tensor_value(batch: DataProto, key: str) -> Any:
    if key not in batch.non_tensor_batch:
        return None
    values = batch.non_tensor_batch[key]
    if len(values) != 1:
        raise ValueError(f"Expected single-sample non_tensor_batch[{key!r}], got shape={values.shape}")
    return values[0]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass
class SkdPartialState:
    """Snapshot of an unfinished SKD trajectory at an exportable handler boundary.

    This is intentionally separate from ``AgentData``.  ``AgentData`` is a
    runtime object and can contain live handles or tool/interact objects.  This
    dataclass is the explicit state that an async scheduler can put into a
    queue and later restore from.
    """

    sample_id: str
    logical_step: int
    source_type: str
    agent_state: str

    request_id: str
    tools_kwargs: dict[str, Any] = field(default_factory=dict)

    messages: list[dict[str, Any]] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    teacher_prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)

    assistant_turns: int = 0
    user_turns: int = 0
    tool_rewards: list[float] = field(default_factory=list)
    turn_scores: list[float] = field(default_factory=list)

    rollout_birth_version: int | None = None
    rollout_min_version: int | None = None
    rollout_max_version: int | None = None
    committed_gen_chunks: int = 0
    committed_env_units: int = 0
    committed_prefix_tokens: int = 0

    metrics: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    # Kept as opaque references for now.  The first async-SKD implementation
    # should prefer text-only tools; multimodal serialization should be made
    # explicit before these are sent through a cross-process queue.
    image_data: Any = None
    video_data: Any = None


@dataclass
class AsyncSkdSample:
    """Single queue/scheduler envelope for completed and partial SKD samples.

    The payload is intentionally a strict union:
    - ``kind == "completed"`` carries a single-sample ``DataProto``.
    - ``kind == "partial"`` carries a resumable ``SkdPartialState``.

    Callers must use ``require_completed()`` or ``require_partial()`` instead
    of directly reading nullable payload fields.
    """

    sample_id: str
    kind: str
    source_type: str
    logical_step: int

    batch: DataProto | None = None
    partial_state: SkdPartialState | None = None

    rollout_birth_version: int | None = None
    rollout_min_version: int | None = None
    rollout_max_version: int | None = None
    train_consume_version: int | None = None

    committed_gen_chunks: int = 0
    committed_env_units: int = 0
    committed_prefix_tokens: int = 0

    drop_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_completed(
        cls,
        *,
        sample_id: str,
        logical_step: int,
        source_type: str,
        batch: DataProto,
        train_consume_version: int | None = None,
        drop_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> "AsyncSkdSample":
        """Build a completed-sample envelope from a single-sample DataProto."""
        batch_metrics = batch.meta_info.get("metrics")
        if metrics is None and isinstance(batch_metrics, list) and len(batch_metrics) == 1:
            metrics = dict(batch_metrics[0])

        sample = cls(
            sample_id=sample_id,
            kind=ASYNC_SKD_SAMPLE_KIND_COMPLETED,
            source_type=source_type,
            logical_step=logical_step,
            batch=batch,
            partial_state=None,
            rollout_birth_version=_optional_int(_single_non_tensor_value(batch, "rollout_birth_version")),
            rollout_min_version=_optional_int(_single_non_tensor_value(batch, "rollout_min_version")),
            rollout_max_version=_optional_int(_single_non_tensor_value(batch, "rollout_max_version")),
            train_consume_version=train_consume_version,
            committed_gen_chunks=int(_single_non_tensor_value(batch, "skd_committed_gen_chunks") or 0),
            committed_env_units=int(_single_non_tensor_value(batch, "skd_committed_env_units") or 0),
            committed_prefix_tokens=int(_single_non_tensor_value(batch, "skd_committed_prefix_tokens") or 0),
            drop_reason=drop_reason,
            metrics=metrics or {},
        )
        sample.validate()
        return sample

    @classmethod
    def from_partial(
        cls,
        *,
        partial_state: SkdPartialState,
        train_consume_version: int | None = None,
        drop_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> "AsyncSkdSample":
        """Build a partial-sample envelope from a resumable SKD snapshot."""
        sample = cls(
            sample_id=partial_state.sample_id,
            kind=ASYNC_SKD_SAMPLE_KIND_PARTIAL,
            source_type=partial_state.source_type,
            logical_step=partial_state.logical_step,
            batch=None,
            partial_state=partial_state,
            rollout_birth_version=partial_state.rollout_birth_version,
            rollout_min_version=partial_state.rollout_min_version,
            rollout_max_version=partial_state.rollout_max_version,
            train_consume_version=train_consume_version,
            committed_gen_chunks=partial_state.committed_gen_chunks,
            committed_env_units=partial_state.committed_env_units,
            committed_prefix_tokens=partial_state.committed_prefix_tokens,
            drop_reason=drop_reason,
            metrics=metrics or dict(partial_state.metrics),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("AsyncSkdSample.sample_id must be non-empty")
        if self.kind not in ASYNC_SKD_SAMPLE_KINDS:
            raise ValueError(f"Invalid AsyncSkdSample.kind={self.kind!r}")
        if self.source_type not in ASYNC_SKD_SAMPLE_SOURCE_TYPES:
            raise ValueError(f"Invalid AsyncSkdSample.source_type={self.source_type!r}")
        if self.logical_step < 0:
            raise ValueError(f"AsyncSkdSample.logical_step must be non-negative, got {self.logical_step}")
        for key, value in (
            ("committed_gen_chunks", self.committed_gen_chunks),
            ("committed_env_units", self.committed_env_units),
            ("committed_prefix_tokens", self.committed_prefix_tokens),
        ):
            if value < 0:
                raise ValueError(f"AsyncSkdSample.{key} must be non-negative, got {value}")

        if self.kind == ASYNC_SKD_SAMPLE_KIND_COMPLETED:
            if self.batch is None:
                raise ValueError("Completed AsyncSkdSample requires batch")
            if self.partial_state is not None:
                raise ValueError("Completed AsyncSkdSample must not carry partial_state")
            if len(self.batch) != 1:
                raise ValueError(f"Completed AsyncSkdSample requires single-sample DataProto, got {len(self.batch)}")
            return

        if self.batch is not None:
            raise ValueError("Partial AsyncSkdSample must not carry batch")
        if self.partial_state is None:
            raise ValueError("Partial AsyncSkdSample requires partial_state")
        if self.partial_state.sample_id != self.sample_id:
            raise ValueError(
                f"Partial AsyncSkdSample sample_id mismatch: envelope={self.sample_id!r}, "
                f"partial={self.partial_state.sample_id!r}"
            )
        if self.partial_state.logical_step != self.logical_step:
            raise ValueError(
                "Partial AsyncSkdSample logical_step mismatch: "
                f"envelope={self.logical_step}, partial={self.partial_state.logical_step}"
            )
        if self.partial_state.source_type != self.source_type:
            raise ValueError(
                f"Partial AsyncSkdSample source_type mismatch: envelope={self.source_type!r}, "
                f"partial={self.partial_state.source_type!r}"
            )

        for key in ("rollout_birth_version", "rollout_min_version", "rollout_max_version"):
            if getattr(self, key) != getattr(self.partial_state, key):
                raise ValueError(
                    f"Partial AsyncSkdSample {key} mismatch: "
                    f"envelope={getattr(self, key)!r}, partial={getattr(self.partial_state, key)!r}"
                )
        for key, envelope_value, partial_value in (
            ("committed_gen_chunks", self.committed_gen_chunks, self.partial_state.committed_gen_chunks),
            ("committed_env_units", self.committed_env_units, self.partial_state.committed_env_units),
            ("committed_prefix_tokens", self.committed_prefix_tokens, self.partial_state.committed_prefix_tokens),
        ):
            if envelope_value != partial_value:
                raise ValueError(
                    f"Partial AsyncSkdSample {key} mismatch: envelope={envelope_value}, partial={partial_value}"
                )

    def require_completed(self) -> DataProto:
        self.validate()
        if self.kind != ASYNC_SKD_SAMPLE_KIND_COMPLETED or self.batch is None:
            raise ValueError(f"AsyncSkdSample is not completed: kind={self.kind!r}")
        return self.batch

    def require_partial(self) -> SkdPartialState:
        self.validate()
        if self.kind != ASYNC_SKD_SAMPLE_KIND_PARTIAL or self.partial_state is None:
            raise ValueError(f"AsyncSkdSample is not partial: kind={self.kind!r}")
        return self.partial_state
