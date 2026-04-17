# Copyright 2025 DDAI Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Speculative Knowledge Distillation (SKD) Agent Loop.

Extends ToolAgentLoop by overriding the generation phase:
instead of generating a full sequence at once, generates in chunks,
verifies each chunk against the Teacher model's top-K,
and replaces rejected tokens with Teacher's top-1.

Teacher logprobs are accumulated during chunk verification,
eliminating the need for a separate teacher logprob computation in postprocessing.
"""

from copy import deepcopy
import logging
import os
from pathlib import Path
import time
import warnings
from typing import Any

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.async_skd.state import RESUMABLE_COMMITTED_UNITS, SkdPartialState
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# SKD debug logging: set VERL_SKD_DEBUG=1 to enable per-chunk diagnostics.
# VERL_SKD_DEBUG=2 for token-level alignment verification (first 3 samples per batch).
_SKD_DEBUG = int(os.getenv("VERL_SKD_DEBUG", "0"))
_SKD_PENDING_TURN_RESPONSE_IDS = "skd_pending_turn_response_ids"


@register("skd_agent")
class SkdAgentLoop(ToolAgentLoop):
    """Agent loop with Speculative Knowledge Distillation.

    Inherits ToolAgentLoop's state machine (PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED)
    and overrides only _handle_generating_state to implement SKD chunk-based generation with
    Teacher verification.
    """

    # Class-level counter for debug logging (only first N samples get token-level logs)
    _debug_sample_count = 0

    def __init__(self, *args, teacher_server_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_server_manager = teacher_server_manager

        # SKD config — read from distillation config if available, otherwise defaults
        distillation_config = self.config.get("distillation", {})
        skd_config = distillation_config.get("skd", {})
        self.skd_chunk_size = skd_config.get("chunk_size", 1024)
        self.skd_verify_top_k = skd_config.get("verify_top_k", 25)
        self.max_chunks_per_sample = skd_config.get("max_chunks_per_sample", 60)
        self.teacher_system_prompt_path = skd_config.get("teacher_system_prompt_path")

        # Loss top-K for teacher logprobs accumulation (from distillation_loss config)
        loss_config = distillation_config.get("distillation_loss", {})
        self.loss_top_k = loss_config.get("topk", 128)
        self.teacher_system_prompt = self._load_teacher_system_prompt()

        if self.teacher_server_manager is None:
            logger.warning(
                "SkdAgentLoop: teacher_server_manager is None. "
                "SKD verification will be skipped — falling back to standard generation."
            )

    def _load_teacher_system_prompt(self) -> str | None:
        """Load teacher-only system prompt once at worker initialization."""
        if not self.teacher_system_prompt_path:
            return None
        prompt_path = Path(self.teacher_system_prompt_path).expanduser()
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        return prompt_text or None

    def _build_teacher_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge teacher-only system guidance into the initial conversation."""
        teacher_messages = deepcopy(messages)
        if not self.teacher_system_prompt:
            return teacher_messages

        if teacher_messages and teacher_messages[0].get("role") == "system":
            content = teacher_messages[0].get("content")
            if isinstance(content, str):
                merged = content.rstrip()
                if merged:
                    merged = f"{merged}\n\n{self.teacher_system_prompt}"
                else:
                    merged = self.teacher_system_prompt
                teacher_messages[0]["content"] = merged
                return teacher_messages

        teacher_messages.insert(0, {"role": "system", "content": self.teacher_system_prompt})
        return teacher_messages

    def _append_student_prompt_delta_to_teacher_stream(self, agent_data: AgentData, prev_prompt_len: int) -> None:
        teacher_prompt_ids = agent_data.extra_fields.get("teacher_prompt_ids")
        if teacher_prompt_ids is None:
            return
        prompt_delta = agent_data.prompt_ids[prev_prompt_len:]
        if prompt_delta:
            teacher_prompt_ids.extend(prompt_delta)

    def _append_dummy_teacher_rows(self, agent_data: AgentData, count: int) -> None:
        """Keep SKD teacher targets aligned with response_mask for tool/user spans."""
        if count <= 0:
            return
        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            return
        assert self.loss_top_k is not None and self.loss_top_k > 0, "SKD dummy rows require distillation topk > 0"

        teacher_ids_list = agent_data.extra_fields["teacher_ids_list"]
        teacher_logprobs_list = agent_data.extra_fields["teacher_logprobs_list"]
        teacher_ids_list.extend([[0] * self.loss_top_k for _ in range(count)])
        teacher_logprobs_list.extend([[0.0] * self.loss_top_k for _ in range(count)])

    def _set_skd_last_committed_unit(self, agent_data: AgentData, unit: str) -> None:
        """Record the latest fully committed SKD atomic unit."""
        agent_data.extra_fields["skd_last_committed_unit"] = unit

    def _get_skd_last_committed_unit(self, agent_data: AgentData) -> str | None:
        """Return the latest fully committed SKD atomic unit, if recorded."""
        return agent_data.extra_fields.get("skd_last_committed_unit")

    def _increment_skd_prefix_stats(
        self,
        agent_data: AgentData,
        *,
        gen_chunks: int = 0,
        env_units: int = 0,
        tokens: int = 0,
    ) -> None:
        """Track committed prefix size without changing rollout semantics."""
        extra_fields = agent_data.extra_fields
        extra_fields["skd_committed_gen_chunks"] = extra_fields.get("skd_committed_gen_chunks", 0) + gen_chunks
        extra_fields["skd_committed_env_units"] = extra_fields.get("skd_committed_env_units", 0) + env_units
        extra_fields["skd_committed_prefix_tokens"] = extra_fields.get("skd_committed_prefix_tokens", 0) + tokens

    def _record_rollout_version_from_output(self, agent_data: AgentData, output: Any) -> None:
        """Track min/max rollout model versions reported by the inference engine."""
        extra = getattr(output, "extra_fields", None) or {}
        version_values = []
        for key in ("min_global_steps", "global_steps", "max_global_steps"):
            value = extra.get(key)
            if value is None:
                continue
            try:
                version_values.append(int(value))
            except (TypeError, ValueError):
                continue
        if not version_values:
            return

        current_min = agent_data.extra_fields.get("rollout_min_version")
        current_max = agent_data.extra_fields.get("rollout_max_version")
        new_min = min(version_values) if current_min is None else min(int(current_min), min(version_values))
        new_max = max(version_values) if current_max is None else max(int(current_max), max(version_values))
        agent_data.extra_fields["rollout_min_version"] = new_min
        agent_data.extra_fields["rollout_max_version"] = new_max
        agent_data.extra_fields.setdefault("rollout_birth_version", new_min)

    def _can_export_partial_state(self, agent_data: AgentData, next_state: AgentState) -> bool:
        """Return whether the current trajectory can be snapshotted for resume.

        The first async-SKD path is conservative: it only exports states whose
        next outer state is ``GENERATING``.  If the next state is
        ``PROCESSING_TOOLS`` or ``INTERACTING``, the assistant prefix is token
        aligned, but an environment result is still pending; the scheduler
        should advance through that macro-step first.
        """
        if next_state != AgentState.GENERATING:
            return False

        last_unit = self._get_skd_last_committed_unit(agent_data)
        if last_unit not in RESUMABLE_COMMITTED_UNITS:
            return False

        teacher_ids_list = agent_data.extra_fields.get("teacher_ids_list")
        teacher_logprobs_list = agent_data.extra_fields.get("teacher_logprobs_list")
        if teacher_ids_list is None or teacher_logprobs_list is None:
            return False
        return len(agent_data.response_mask) == len(teacher_ids_list) == len(teacher_logprobs_list)

    def _export_partial_state(
        self,
        agent_data: AgentData,
        next_state: AgentState,
        *,
        sample_id: str,
        logical_step: int,
        source_type: str,
    ) -> SkdPartialState:
        """Export an unfinished trajectory snapshot at a committed-unit boundary."""
        if not self._can_export_partial_state(agent_data, next_state):
            raise ValueError(
                "Cannot export SKD partial state: "
                f"next_state={next_state}, "
                f"last_committed_unit={self._get_skd_last_committed_unit(agent_data)}, "
                f"response_len={len(agent_data.response_mask)}, "
                f"teacher_rows={len(agent_data.extra_fields.get('teacher_ids_list', []))}"
            )
        self._assert_teacher_alignment(agent_data)

        extra_fields = deepcopy(agent_data.extra_fields)
        return SkdPartialState(
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            agent_state=next_state.value,
            last_committed_unit=extra_fields["skd_last_committed_unit"],
            request_id=agent_data.request_id,
            tools_kwargs=deepcopy(agent_data.tools_kwargs),
            messages=deepcopy(agent_data.messages),
            prompt_ids=list(agent_data.prompt_ids),
            teacher_prompt_ids=list(extra_fields.get("teacher_prompt_ids", [])),
            response_ids=list(agent_data.response_ids),
            response_mask=list(agent_data.response_mask),
            response_logprobs=list(agent_data.response_logprobs),
            assistant_turns=agent_data.assistant_turns,
            user_turns=agent_data.user_turns,
            tool_rewards=list(agent_data.tool_rewards),
            turn_scores=list(agent_data.turn_scores),
            rollout_birth_version=extra_fields.get("rollout_birth_version"),
            rollout_min_version=extra_fields.get("rollout_min_version"),
            rollout_max_version=extra_fields.get("rollout_max_version"),
            committed_gen_chunks=int(extra_fields.get("skd_committed_gen_chunks", 0)),
            committed_env_units=int(extra_fields.get("skd_committed_env_units", 0)),
            committed_prefix_tokens=int(extra_fields.get("skd_committed_prefix_tokens", 0)),
            metrics=deepcopy(agent_data.metrics),
            extra_fields=extra_fields,
            image_data=deepcopy(agent_data.image_data),
            video_data=deepcopy(agent_data.video_data),
        )

    def _restore_partial_state(self, partial_state: SkdPartialState) -> tuple[AgentData, AgentState]:
        """Restore a previously exported SKD partial snapshot."""
        try:
            next_state = AgentState(partial_state.agent_state)
        except ValueError as exc:
            raise ValueError(f"Invalid SKD partial state agent_state={partial_state.agent_state!r}") from exc

        if next_state != AgentState.GENERATING:
            raise ValueError(f"Cannot restore SKD partial state into unsupported next_state={next_state}")
        if partial_state.last_committed_unit not in RESUMABLE_COMMITTED_UNITS:
            raise ValueError(
                "Cannot restore SKD partial state with unsupported "
                f"last_committed_unit={partial_state.last_committed_unit!r}"
            )

        agent_data = AgentData(
            messages=deepcopy(partial_state.messages),
            image_data=deepcopy(partial_state.image_data),
            video_data=deepcopy(partial_state.video_data),
            metrics=deepcopy(partial_state.metrics),
            request_id=partial_state.request_id,
            tools_kwargs=deepcopy(partial_state.tools_kwargs),
        )
        agent_data.prompt_ids = list(partial_state.prompt_ids)
        agent_data.response_ids = list(partial_state.response_ids)
        agent_data.response_mask = list(partial_state.response_mask)
        agent_data.response_logprobs = list(partial_state.response_logprobs)
        agent_data.assistant_turns = partial_state.assistant_turns
        agent_data.user_turns = partial_state.user_turns
        agent_data.tool_rewards = list(partial_state.tool_rewards)
        agent_data.turn_scores = list(partial_state.turn_scores)
        agent_data.extra_fields = deepcopy(partial_state.extra_fields)

        # Keep the structured dataclass fields authoritative for restore.  The
        # same values are mirrored into extra_fields because the existing SKD
        # loss reconstruction path consumes them from there.
        agent_data.extra_fields["teacher_prompt_ids"] = list(partial_state.teacher_prompt_ids)
        agent_data.extra_fields["skd_last_committed_unit"] = partial_state.last_committed_unit
        if partial_state.rollout_birth_version is not None:
            agent_data.extra_fields["rollout_birth_version"] = partial_state.rollout_birth_version
        if partial_state.rollout_min_version is not None:
            agent_data.extra_fields["rollout_min_version"] = partial_state.rollout_min_version
        if partial_state.rollout_max_version is not None:
            agent_data.extra_fields["rollout_max_version"] = partial_state.rollout_max_version
        agent_data.extra_fields["skd_committed_gen_chunks"] = partial_state.committed_gen_chunks
        agent_data.extra_fields["skd_committed_env_units"] = partial_state.committed_env_units
        agent_data.extra_fields["skd_committed_prefix_tokens"] = partial_state.committed_prefix_tokens

        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            raise ValueError("Invalid SKD partial state: missing teacher row lists in extra_fields")
        self._assert_teacher_alignment(agent_data)
        return agent_data, next_state

    async def _run_until_exportable_boundary(
        self,
        agent_data: AgentData,
        state: AgentState,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        """Advance a trajectory until it is completed or safe to export.

        This is the cooperative pause driver for lookahead execution.  It never
        interrupts inside an SKD chunk, teacher verification, tool execution, or
        dummy-row append.  It returns only at ``TERMINATED`` or at a boundary
        accepted by ``_can_export_partial_state``.
        """
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
                continue
            if state == AgentState.GENERATING:
                state = await self._handle_generating_state(
                    agent_data,
                    sampling_params,
                    stop_after_committed_unit=True,
                )
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                raise ValueError(f"Invalid AgentState while advancing SKD boundary: {state}")

            if self._can_export_partial_state(agent_data, state):
                return state

        return state

    def _assert_teacher_alignment(self, agent_data: AgentData) -> None:
        """Validate that response_mask and teacher rows stay response-token aligned."""
        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            return

        teacher_ids_list = agent_data.extra_fields["teacher_ids_list"]
        teacher_logprobs_list = agent_data.extra_fields["teacher_logprobs_list"]
        response_len = len(agent_data.response_mask)
        assert len(teacher_ids_list) == response_len, (
            f"[SKD] teacher_ids_list length {len(teacher_ids_list)} != response_mask length {response_len}"
        )
        assert len(teacher_logprobs_list) == response_len, (
            f"[SKD] teacher_logprobs_list length {len(teacher_logprobs_list)} != response_mask length {response_len}"
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Initialize separate student and teacher prompt streams."""
        del sampling_params
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids

        teacher_messages = self._build_teacher_messages(agent_data.messages)
        if teacher_messages == agent_data.messages:
            teacher_prompt_ids = list(prompt_ids)
        else:
            teacher_prompt_ids = await self.apply_chat_template(
                teacher_messages,
                tools=self.tool_schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
        agent_data.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        prev_prompt_len = len(agent_data.prompt_ids)
        prev_response_len = len(agent_data.response_mask)
        next_state = await super()._handle_processing_tools_state(agent_data)
        self._append_student_prompt_delta_to_teacher_stream(agent_data, prev_prompt_len)
        appended_len = len(agent_data.response_mask) - prev_response_len
        self._append_dummy_teacher_rows(agent_data, appended_len)
        self._assert_teacher_alignment(agent_data)
        if appended_len > 0:
            self._increment_skd_prefix_stats(agent_data, env_units=1, tokens=appended_len)
            self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK_WITH_TOOL_RESULT")
        return next_state

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        prev_prompt_len = len(agent_data.prompt_ids)
        prev_response_len = len(agent_data.response_mask)
        next_state = await super()._handle_interacting_state(agent_data)
        self._append_student_prompt_delta_to_teacher_stream(agent_data, prev_prompt_len)
        appended_len = len(agent_data.response_mask) - prev_response_len
        self._append_dummy_teacher_rows(agent_data, appended_len)
        self._assert_teacher_alignment(agent_data)
        if appended_len > 0:
            self._increment_skd_prefix_stats(agent_data, env_units=1, tokens=appended_len)
            self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK_WITH_INTERACTION_RESULT")
        return next_state

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
        stop_after_committed_unit: bool = False,
    ) -> AgentState:
        """SKD chunk-based generation with Teacher verification.

        Replaces ToolAgentLoop's single-shot generation with:
        1. Student generates a chunk of tokens
        2. Teacher verifies each token against its top-K
        3. First rejected token is replaced with Teacher's top-1
        4. Repeat from the replacement point
        5. Teacher logprobs are accumulated during verification (no separate pass needed)
        """
        # Fallback to standard generation if no teacher
        if self.teacher_server_manager is None:
            if stop_after_committed_unit:
                raise ValueError("stop_after_committed_unit requires SKD teacher verification")
            return await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        sample_start_time = time.monotonic()

        # Initialize SKD metrics
        skd_metrics = agent_data.metrics.setdefault("skd", {
            "accept_count": 0,
            "reject_count": 0,
            "chunk_count": 0,
            "student_gen_ms": 0.0,
            "teacher_verify_ms": 0.0,
        })

        # Initialize teacher logprobs accumulation lists in extra_fields
        teacher_ids_list = agent_data.extra_fields.setdefault("teacher_ids_list", [])
        teacher_logprobs_list = agent_data.extra_fields.setdefault("teacher_logprobs_list", [])
        teacher_prompt_ids = agent_data.extra_fields.setdefault("teacher_prompt_ids", list(agent_data.prompt_ids))

        # Track response_ids for the current assistant turn.  In the normal
        # path this stays local until the turn finishes.  In pause-aware
        # lookahead mode it must survive export/restore across chunk boundaries.
        turn_response_ids = list(agent_data.extra_fields.get(_SKD_PENDING_TURN_RESPONSE_IDS, []))

        # Debug: token-level alignment logging for first N samples
        SkdAgentLoop._debug_sample_count += 1
        do_token_debug = _SKD_DEBUG >= 2 and SkdAgentLoop._debug_sample_count <= 3
        termination_reason = "unknown"
        chunk_output = None

        initial_prompt_len = len(agent_data.prompt_ids)

        # === SKD chunk loop ===
        with simple_timer("skd_generate", agent_data.metrics):
            while True:
                remaining_budget = self.response_length - len(agent_data.response_mask)
                if remaining_budget <= 0:
                    termination_reason = "budget_exhausted"
                    break
                actual_chunk_size = min(self.skd_chunk_size, remaining_budget)

                # 1. Student generates a chunk
                chunk_t0 = time.monotonic()
                with simple_timer("skd_student_chunk", agent_data.metrics):
                    chunk_output = await self.server_manager.generate(
                        request_id=agent_data.request_id,
                        prompt_ids=agent_data.prompt_ids,
                        sampling_params={**sampling_params, "max_tokens": actual_chunk_size},
                        image_data=agent_data.image_data,
                        video_data=agent_data.video_data,
                    )
                self._record_rollout_version_from_output(agent_data, chunk_output)
                chunk = chunk_output.token_ids
                student_ms = (time.monotonic() - chunk_t0) * 1000
                skd_metrics["student_gen_ms"] += student_ms

                if not chunk:
                    termination_reason = "empty_chunk"
                    break

                # Track num_preempted (same pattern as ToolAgentLoop)
                if agent_data.metrics.get("num_preempted") is None:
                    agent_data.metrics["num_preempted"] = (
                        chunk_output.num_preempted if chunk_output.num_preempted is not None else -1
                    )
                else:
                    agent_data.metrics["num_preempted"] += (
                        chunk_output.num_preempted if chunk_output.num_preempted is not None else 0
                    )

                # 2. Teacher verification
                teacher_t0 = time.monotonic()
                with simple_timer("skd_teacher_verify", agent_data.metrics):
                    verify_sequence = teacher_prompt_ids + chunk
                    logprob_start_len = max(len(teacher_prompt_ids) - 1, 0)
                    teacher_ids, teacher_logprobs = (
                        await self.teacher_server_manager.compute_teacher_logprobs_single(
                            request_id=agent_data.request_id,
                            sequence_ids=verify_sequence,
                            logprob_start_len=logprob_start_len,
                            multi_modal_data=agent_data.extra_fields.get("multi_modal_data"),
                        )
                    )
                teacher_ms = (time.monotonic() - teacher_t0) * 1000
                skd_metrics["teacher_verify_ms"] += teacher_ms
                # In SGLang delta mode, teacher_ids / teacher_logprobs only cover the
                # chunk suffix rows needed by SKD, aligned so local row k supervises
                # chunk token k.

                # 3. Accept/Reject
                chunk_start = len(agent_data.prompt_ids)
                rejection_pos = None
                for k in range(len(chunk)):
                    teacher_idx = k
                    if teacher_idx < 0 or teacher_idx >= teacher_ids.shape[0]:
                        break
                    # Check if student token is in teacher's top-K for verification
                    teacher_topk_at_pos = teacher_ids[teacher_idx, : self.skd_verify_top_k].tolist()
                    if chunk[k] not in teacher_topk_at_pos:
                        rejection_pos = k
                        break

                # 4. Accumulate tokens + teacher logprobs
                if rejection_pos is not None:
                    accepted_tokens = list(chunk[:rejection_pos])
                    teacher_replacement_idx = rejection_pos
                    teacher_replacement = int(teacher_ids[teacher_replacement_idx, 0].item())
                    new_tokens = accepted_tokens + [teacher_replacement]
                    skd_metrics["accept_count"] += len(accepted_tokens)
                    skd_metrics["reject_count"] += 1
                else:
                    new_tokens = list(chunk)
                    skd_metrics["accept_count"] += len(new_tokens)

                skd_metrics["chunk_count"] += 1

                # === Debug: token-level alignment verification ===
                if do_token_debug and skd_metrics["chunk_count"] <= 3:
                    self._log_token_alignment(
                        chunk, new_tokens, rejection_pos,
                        teacher_ids, chunk_start, agent_data.request_id,
                        skd_metrics["chunk_count"],
                    )

                # === Debug: per-chunk timing log ===
                if _SKD_DEBUG >= 1:
                    acc = len(new_tokens) - (1 if rejection_pos is not None else 0)
                    warnings.warn(
                        f"[SKD_DBG] chunk={skd_metrics['chunk_count']:>3} "
                        f"student={student_ms:>7.1f}ms teacher={teacher_ms:>7.1f}ms "
                        f"teacher_prefix_len={len(teacher_prompt_ids)} "
                        f"teacher_suffix_len={len(chunk)} "
                        f"teacher_seq_len={len(verify_sequence)} "
                        f"chunk_len={len(chunk)} accepted={acc} rejected={1 if rejection_pos is not None else 0} "
                        f"new_tokens={len(new_tokens)} "
                        f"prompt_len={chunk_start} total_resp={len(agent_data.response_mask)+len(new_tokens)} "
                        f"req={agent_data.request_id}",
                        stacklevel=1,
                    )

                # Update agent_data (same pattern as ToolAgentLoop L244-246)
                agent_data.prompt_ids += new_tokens
                teacher_prompt_ids += new_tokens
                agent_data.response_mask += [1] * len(new_tokens)
                turn_response_ids.extend(new_tokens)
                agent_data.response_ids = list(turn_response_ids)
                agent_data.extra_fields[_SKD_PENDING_TURN_RESPONSE_IDS] = list(turn_response_ids)

                # Accumulate teacher targets only for the committed rollout path.
                # In delta mode, local row k already corresponds to committed token k.
                for k in range(len(new_tokens)):
                    teacher_idx = k
                    assert 0 <= teacher_idx < teacher_ids.shape[0], (
                        f"[SKD] invalid teacher_idx={teacher_idx} for committed token {k} "
                        f"(chunk_start={chunk_start}, committed={len(new_tokens)}, teacher_len={teacher_ids.shape[0]})"
                    )
                    teacher_id_row = teacher_ids[teacher_idx].tolist()
                    teacher_logprob_row = teacher_logprobs[teacher_idx].tolist()
                    assert len(teacher_id_row) == self.loss_top_k, (
                        f"[SKD] teacher_ids row width {len(teacher_id_row)} != configured topk {self.loss_top_k}"
                    )
                    assert len(teacher_logprob_row) == self.loss_top_k, (
                        "[SKD] teacher_logprobs row width "
                        f"{len(teacher_logprob_row)} != configured topk {self.loss_top_k}"
                    )
                    teacher_ids_list.append(teacher_id_row)
                    teacher_logprobs_list.append(teacher_logprob_row)

                self._assert_teacher_alignment(agent_data)
                self._increment_skd_prefix_stats(agent_data, gen_chunks=1, tokens=len(new_tokens))
                self._set_skd_last_committed_unit(agent_data, "ASSISTANT_GEN_CHUNK")

                # 5. Termination checks within chunk loop
                # Note: stop_reason == "completed" covers both EOS and max_tokens,
                # so we check for actual EOS tokens instead.
                eos_token_id = self.tokenizer.eos_token_id
                eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                if any(t in new_tokens for t in eos_ids):
                    termination_reason = "eos"
                    break
                if skd_metrics["chunk_count"] >= self.max_chunks_per_sample:
                    termination_reason = "max_chunks"
                    break
                if stop_after_committed_unit:
                    termination_reason = "committed_unit_boundary"
                    agent_data.extra_fields["skd_termination_reason"] = termination_reason
                    if not agent_data.extra_fields.get("max_global_steps"):
                        agent_data.extra_fields.update(chunk_output.extra_fields)
                    if chunk_output.routed_experts is not None:
                        agent_data.routed_experts = chunk_output.routed_experts
                    return AgentState.GENERATING

        # === Post chunk loop: match ToolAgentLoop's post-generation logic ===

        sample_elapsed_ms = (time.monotonic() - sample_start_time) * 1000
        agent_data.extra_fields["skd_termination_reason"] = termination_reason

        # Store response_ids for this turn (used by tool_parser)
        agent_data.response_ids = turn_response_ids
        agent_data.extra_fields.pop(_SKD_PENDING_TURN_RESPONSE_IDS, None)

        # Update extra_fields (first time only, same pattern as ToolAgentLoop L235-241)
        if chunk_output is not None and not agent_data.extra_fields.get("max_global_steps"):
            agent_data.extra_fields.update(chunk_output.extra_fields)

        # Track routed_experts (same as ToolAgentLoop L250-251, for MoE models)
        if chunk_output is not None and chunk_output.routed_experts is not None:
            agent_data.routed_experts = chunk_output.routed_experts

        if turn_response_ids:
            agent_data.assistant_turns += 1

        # Compute accept rate metric
        total = skd_metrics["accept_count"] + skd_metrics["reject_count"]
        if total > 0:
            agent_data.metrics["skd_accept_rate"] = skd_metrics["accept_count"] / total

        # === Per-sample summary log (always emitted via warnings for visibility) ===
        accept_rate = skd_metrics["accept_count"] / total * 100 if total > 0 else 0
        avg_tokens_per_chunk = len(turn_response_ids) / max(skd_metrics["chunk_count"], 1)
        warnings.warn(
            f"[SKD] req={agent_data.request_id} "
            f"done={termination_reason} "
            f"chunks={skd_metrics['chunk_count']}/{self.max_chunks_per_sample} "
            f"resp_len={len(turn_response_ids)} "
            f"accept={skd_metrics['accept_count']} reject={skd_metrics['reject_count']} "
            f"rate={accept_rate:.1f}% "
            f"avg_tok/chunk={avg_tokens_per_chunk:.1f} "
            f"student={skd_metrics['student_gen_ms']:.0f}ms "
            f"teacher={skd_metrics['teacher_verify_ms']:.0f}ms "
            f"total={sample_elapsed_ms:.0f}ms "
            f"prompt={initial_prompt_len} "
            f"teacher_logprobs_accumulated={len(teacher_ids_list)}",
            stacklevel=1,
        )

        # Check termination conditions (same as ToolAgentLoop L253-259)
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (same as ToolAgentLoop L261-263)
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Handle interaction if needed (same as ToolAgentLoop L266-271)
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            agent_data.messages.append({"role": "assistant", "content": assistant_message})

        # Determine next state (same as ToolAgentLoop L274-279)
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    def _log_token_alignment(
        self,
        chunk: list[int],
        new_tokens: list[int],
        rejection_pos: int | None,
        teacher_ids: torch.Tensor,
        chunk_start: int,
        request_id: str,
        chunk_num: int,
    ):
        """Log token-level alignment details for verifying offset correctness.

        For each token in new_tokens, shows the local teacher row used for both
        verification and distillation after delta slicing.
        """
        try:
            lines = [f"[SKD_ALIGN] req={request_id} chunk={chunk_num} "
                     f"chunk_start={chunk_start} chunk_len={len(chunk)} "
                     f"rejection_pos={rejection_pos}"]
            # Show first 5 tokens of the chunk for brevity
            show_count = min(len(new_tokens), 5)
            for k in range(show_count):
                student_tok = chunk[k] if k < len(chunk) else -1
                final_tok = new_tokens[k]
                local_idx = k

                if 0 <= local_idx < teacher_ids.shape[0]:
                    local_top5 = teacher_ids[local_idx, :5].tolist()
                else:
                    local_top5 = "OOB"

                in_verify = "✓" if (isinstance(local_top5, list) and student_tok in local_top5) else "✗"
                replaced = " REPLACED" if (rejection_pos is not None and k == rejection_pos) else ""

                # Decode tokens for readability (catch errors for special tokens)
                try:
                    student_text = repr(self.tokenizer.decode([student_tok]))
                    final_text = repr(self.tokenizer.decode([final_tok]))
                except Exception:
                    student_text = f"id={student_tok}"
                    final_text = f"id={final_tok}"

                lines.append(
                    f"  k={k}: student={student_text}({student_tok}) {in_verify} "
                    f"teacher_top5={local_top5} | "
                    f"final={final_text}({final_tok}){replaced}"
                )

            warnings.warn("\n".join(lines), stacklevel=2)
        except Exception as e:
            warnings.warn(f"[SKD_ALIGN] logging error: {e}", stacklevel=2)
