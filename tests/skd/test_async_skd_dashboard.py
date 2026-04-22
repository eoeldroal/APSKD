"""Tests for async SKD dashboard event reduction."""

from __future__ import annotations

import json

from verl.experimental.async_skd.dashboard import DashboardStateStore, reduce_events


def test_reduce_events_tracks_replica_requests_scheduler_slots_and_lt_candidate():
    state = reduce_events(
        [
            {
                "event": "sample_launch",
                "ts": 10.0,
                "sample_id": "cur-0",
                "scheduler_worker_idx": 0,
                "source_type": "base_current",
                "barrier_role": "current",
                "worker_capacity": 2,
            },
            {
                "event": "sample_launch",
                "ts": 12.0,
                "sample_id": "look-0",
                "scheduler_worker_idx": 0,
                "source_type": "lookahead",
                "barrier_role": "lookahead",
                "worker_capacity": 2,
            },
            {
                "event": "replica_request_start",
                "ts": 13.0,
                "sample_id": "cur-0",
                "role": "student",
                "replica_id": "server-0",
            },
            {
                "event": "chunk_commit",
                "ts": 14.0,
                "sample_id": "cur-0",
                "chunk_idx": 1,
                "response_len": 31,
            },
            {
                "event": "replica_request_finish",
                "ts": 15.0,
                "sample_id": "cur-0",
                "role": "student",
                "replica_id": "server-0",
                "status": "ok",
            },
        ],
        now=20.0,
    )

    assert state["scheduler_workers"]["0"]["active_total"] == 2
    assert state["scheduler_workers"]["0"]["active_by_source"] == {"base_current": 1, "lookahead": 1}
    assert state["replicas"]["student:server-0"]["active_requests"] == 0
    assert state["samples"]["cur-0"]["chunk_idx"] == 1
    assert state["samples"]["cur-0"]["response_len"] == 31
    assert state["lt_candidate"]["sample_id"] == "cur-0"
    assert state["lt_candidate"]["elapsed_ms"] == 10000.0


def test_reduce_events_exposes_authoritative_worker_slots():
    state = reduce_events(
        [
            {
                "event": "sample_launch",
                "ts": 1.0,
                "sample_id": "base-0",
                "scheduler_worker_idx": 0,
                "source_type": "base_current",
                "barrier_role": "current",
                "worker_capacity": 2,
            },
            {
                "event": "sample_launch",
                "ts": 2.0,
                "sample_id": "look-0",
                "scheduler_worker_idx": 0,
                "source_type": "lookahead",
                "barrier_role": "lookahead",
                "worker_capacity": 2,
            },
            {
                "event": "sample_finish",
                "ts": 3.0,
                "sample_id": "base-0",
                "scheduler_worker_idx": 0,
                "source_type": "base_current",
                "status": "completed",
            },
        ],
        now=4.0,
    )

    worker = state["scheduler_workers"]["0"]
    assert worker["active_total"] == 1
    assert worker["active_by_source"] == {"lookahead": 1}
    assert worker["active_samples"] == [
        {
            "sample_id": "look-0",
            "source_type": "lookahead",
            "barrier_role": "lookahead",
            "chunk_idx": None,
            "response_len": None,
            "committed_gen_chunks": None,
            "committed_prefix_tokens": None,
            "elapsed_ms": 2000.0,
            "last_student_replica_id": None,
            "last_teacher_replica_id": None,
        }
    ]


def test_reduce_events_separates_student_and_teacher_request_queues():
    state = reduce_events(
        [
            {
                "event": "sample_launch",
                "ts": 1.0,
                "sample_id": "s0",
                "scheduler_worker_idx": 0,
                "source_type": "resumed_current",
                "barrier_role": "current",
                "worker_capacity": 1,
            },
            {
                "event": "replica_request_start",
                "ts": 2.0,
                "request_instance_id": "student-request-0",
                "request_id": "sticky-0",
                "sample_id": "s0",
                "role": "student",
                "replica_id": "student-0",
                "request_kind": "student_generate",
                "source_type": "resumed_current",
                "prompt_len": 100,
            },
            {
                "event": "replica_request_finish",
                "ts": 3.0,
                "request_instance_id": "student-request-0",
                "request_id": "sticky-0",
                "sample_id": "s0",
                "role": "student",
                "replica_id": "student-0",
                "request_kind": "student_generate",
                "duration_ms": 1000.0,
                "status": "ok",
            },
            {
                "event": "replica_request_start",
                "ts": 4.0,
                "request_instance_id": "teacher-request-0",
                "request_id": "sticky-0",
                "sample_id": "s0",
                "role": "teacher",
                "replica_id": "teacher-0",
                "request_kind": "teacher_verify",
                "source_type": "resumed_current",
                "prompt_len": 200,
            },
        ],
        now=5.0,
    )

    student = state["replicas"]["student:student-0"]
    teacher = state["replicas"]["teacher:teacher-0"]
    assert student["active_requests"] == 0
    assert student["recent_avg_ms"] == 1000.0
    assert teacher["active_requests"] == 1
    assert teacher["active_by_source"] == {"resumed_current": 1}
    assert teacher["active_by_kind"] == {"teacher_verify": 1}
    assert teacher["active_sample_count"] == 1
    assert teacher["active_requests_detail"][0]["sample_id"] == "s0"
    assert state["samples"]["s0"]["last_student_replica_id"] == "student-0"
    assert state["samples"]["s0"]["last_teacher_replica_id"] == "teacher-0"


def test_dashboard_state_store_reads_only_appended_events(tmp_path):
    event_log = tmp_path / "events.jsonl"
    first_event = {
        "event": "sample_launch",
        "ts": 1.0,
        "sample_id": "s0",
        "scheduler_worker_idx": 0,
        "source_type": "base_current",
        "barrier_role": "current",
        "worker_capacity": 1,
    }
    event_log.write_text(json.dumps(first_event) + "\n")
    store = DashboardStateStore(str(event_log))

    first = store.snapshot(now=2.0)
    assert first["scheduler_workers"]["0"]["active_total"] == 1
    first_offset = first["reader"]["offset"]

    with event_log.open("a", encoding="utf-8") as event_file:
        event_file.write(
            json.dumps(
                {
                    "event": "sample_finish",
                    "ts": 3.0,
                    "sample_id": "s0",
                    "scheduler_worker_idx": 0,
                    "source_type": "base_current",
                    "status": "completed",
                }
            )
            + "\n"
        )

    second = store.snapshot(now=4.0)
    assert second["scheduler_workers"]["0"]["active_total"] == 0
    assert second["reader"]["offset"] > first_offset
    assert second["reader"]["lines_read"] == 2
