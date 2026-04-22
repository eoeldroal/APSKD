"""Local web dashboard for async SKD JSONL events."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse


RECENT_LATENCY_LIMIT = 256


def _positive_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.items() if value > 0}


def _decrement(counter: Counter[str], key: str | None) -> None:
    if key is None:
        return
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def _request_kind(event: dict[str, Any], role: str) -> str:
    explicit = event.get("request_kind")
    if explicit:
        return str(explicit)
    return "teacher_verify" if role == "teacher" else "student_generate"


def _source_type(event: dict[str, Any], sample: dict[str, Any] | None = None) -> str:
    source = event.get("source_type")
    if source is None and sample is not None:
        source = sample.get("source_type")
    return str(source or "unknown")


def _barrier_role(event: dict[str, Any], sample: dict[str, Any] | None = None) -> str | None:
    role = event.get("barrier_role")
    if role is None and sample is not None:
        role = sample.get("barrier_role")
    return None if role is None else str(role)


def _worker_state(workers: dict[str, dict[str, Any]], worker_idx: Any) -> dict[str, Any]:
    key = str(worker_idx)
    return workers.setdefault(
        key,
        {
            "worker_idx": key,
            "capacity": None,
            "active_sample_ids": set(),
            "active_by_source": Counter(),
        },
    )


def _replica_state(replicas: dict[str, dict[str, Any]], role: str, replica_id: str) -> dict[str, Any]:
    key = f"{role}:{replica_id}"
    return replicas.setdefault(
        key,
        {
            "key": key,
            "role": role,
            "replica_id": replica_id,
            "active_request_keys": {},
            "active_by_source": Counter(),
            "active_by_kind": Counter(),
            "request_count": 0,
            "finished_request_count": 0,
            "error_request_count": 0,
            "recent_durations_ms": deque(maxlen=RECENT_LATENCY_LIMIT),
        },
    )


class DashboardReducer:
    """Incrementally reduces async SKD events into UI state."""

    def __init__(self) -> None:
        self.samples: dict[str, dict[str, Any]] = {}
        self.scheduler_workers: dict[str, dict[str, Any]] = {}
        self.replicas: dict[str, dict[str, Any]] = {}
        self.anomalies: list[dict[str, Any]] = []
        self.last_event_ts: float | None = None
        self._legacy_seq = 0
        self._legacy_request_queues: defaultdict[str, deque[str]] = defaultdict(deque)

    def apply(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        ts = float(event.get("ts", time.time()))
        self.last_event_ts = ts
        sample_id = event.get("sample_id")

        if event_name == "sample_launch" and sample_id is not None:
            self._apply_sample_launch(event, str(sample_id), ts)
        elif event_name == "sample_finish" and sample_id is not None:
            self._apply_sample_finish(event, str(sample_id), ts)
        elif event_name == "replica_request_start":
            self._apply_replica_request_start(event, ts)
        elif event_name == "replica_request_finish":
            self._apply_replica_request_finish(event, ts)
        elif event_name == "chunk_commit" and sample_id is not None:
            self._apply_chunk_commit(event, str(sample_id), ts)
        elif event_name == "drain_start":
            actual_lt_sample_id = event.get("actual_lt_sample_id")
            if actual_lt_sample_id is not None:
                self.samples.setdefault(str(actual_lt_sample_id), {"sample_id": str(actual_lt_sample_id)})[
                    "actual_lt"
                ] = True

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        scheduler_workers = self._snapshot_workers(now)
        replicas = self._snapshot_replicas(now)
        lt_candidate = self._snapshot_lt_candidate(now)
        anomalies = list(self.anomalies[-50:])

        for worker in scheduler_workers.values():
            capacity = worker.get("capacity")
            if capacity is not None and worker["active_total"] > int(capacity):
                anomalies.append(
                    {
                        "level": "critical",
                        "message": f"worker {worker['worker_idx']} exceeds capacity",
                        "worker_idx": worker["worker_idx"],
                    }
                )

        return {
            "last_event_ts": self.last_event_ts,
            "event_lag_s": None if self.last_event_ts is None else now - self.last_event_ts,
            "samples": {sample_id: dict(sample) for sample_id, sample in self.samples.items()},
            "scheduler_workers": scheduler_workers,
            "replicas": replicas,
            "lt_candidate": lt_candidate,
            "anomalies": anomalies,
        }

    def _apply_sample_launch(self, event: dict[str, Any], sample_id: str, ts: float) -> None:
        worker = _worker_state(self.scheduler_workers, event.get("scheduler_worker_idx", "unknown"))
        worker["capacity"] = event.get("worker_capacity", worker.get("capacity"))
        worker["active_sample_ids"].add(sample_id)
        source_type = _source_type(event)
        worker["active_by_source"][source_type] += 1
        self.samples[sample_id] = {
            **self.samples.get(sample_id, {}),
            "sample_id": sample_id,
            "status": "running",
            "launch_ts": ts,
            "scheduler_worker_idx": event.get("scheduler_worker_idx"),
            "order": event.get("order"),
            "source_type": source_type,
            "barrier_role": event.get("barrier_role"),
        }

    def _apply_sample_finish(self, event: dict[str, Any], sample_id: str, ts: float) -> None:
        sample = self.samples.setdefault(sample_id, {"sample_id": sample_id})
        sample.update(
            {
                "status": event.get("status", "finished"),
                "finish_ts": ts,
                "duration_ms": event.get("duration_ms"),
                "committed_gen_chunks": event.get("committed_gen_chunks", sample.get("committed_gen_chunks")),
                "committed_prefix_tokens": event.get(
                    "committed_prefix_tokens", sample.get("committed_prefix_tokens")
                ),
            }
        )
        worker = _worker_state(self.scheduler_workers, event.get("scheduler_worker_idx", "unknown"))
        if sample_id in worker["active_sample_ids"]:
            worker["active_sample_ids"].remove(sample_id)
            _decrement(worker["active_by_source"], _source_type(event, sample))

    def _apply_replica_request_start(self, event: dict[str, Any], ts: float) -> None:
        role = str(event.get("role", "unknown"))
        replica_id = str(event.get("replica_id", "unknown"))
        replica = _replica_state(self.replicas, role, replica_id)
        replica["request_count"] += 1

        sample_id = event.get("sample_id")
        sample = None
        if sample_id is not None:
            sample_id = str(sample_id)
            sample = self.samples.setdefault(sample_id, {"sample_id": sample_id})
            sample["last_replica_id"] = replica_id
            sample["last_replica_role"] = role
            sample[f"last_{role}_replica_id"] = replica_id

        request_key = self._start_request_key(event, role, replica_id, sample_id)
        source_type = _source_type(event, sample)
        request_kind = _request_kind(event, role)
        barrier_role = _barrier_role(event, sample)
        request = {
            "request_instance_id": event.get("request_instance_id"),
            "request_key": request_key,
            "request_id": event.get("request_id"),
            "sample_id": sample_id,
            "source_type": source_type,
            "barrier_role": barrier_role,
            "request_kind": request_kind,
            "prompt_len": event.get("prompt_len"),
            "start_ts": ts,
        }
        replica["active_request_keys"][request_key] = request
        replica["active_by_source"][source_type] += 1
        replica["active_by_kind"][request_kind] += 1

    def _apply_replica_request_finish(self, event: dict[str, Any], ts: float) -> None:
        role = str(event.get("role", "unknown"))
        replica_id = str(event.get("replica_id", "unknown"))
        replica = _replica_state(self.replicas, role, replica_id)
        replica["finished_request_count"] += 1
        if event.get("status") not in (None, "ok"):
            replica["error_request_count"] += 1

        request_key = self._finish_request_key(event, role, replica_id)
        request = replica["active_request_keys"].pop(request_key, None) if request_key is not None else None
        if request is None:
            self.anomalies.append(
                {
                    "level": "warning",
                    "message": "replica request finish without matching start",
                    "role": role,
                    "replica_id": replica_id,
                    "request_id": event.get("request_id"),
                    "request_instance_id": event.get("request_instance_id"),
                }
            )
            duration_ms = event.get("duration_ms")
            if duration_ms is not None:
                replica["recent_durations_ms"].append(float(duration_ms))
            return

        _decrement(replica["active_by_source"], request.get("source_type"))
        _decrement(replica["active_by_kind"], request.get("request_kind"))
        duration_ms = event.get("duration_ms")
        if duration_ms is None:
            duration_ms = (ts - float(request["start_ts"])) * 1000
        replica["recent_durations_ms"].append(float(duration_ms))

    def _apply_chunk_commit(self, event: dict[str, Any], sample_id: str, ts: float) -> None:
        sample = self.samples.setdefault(sample_id, {"sample_id": sample_id})
        sample.update(
            {
                "chunk_idx": event.get("chunk_idx"),
                "response_len": event.get("response_len"),
                "committed_gen_chunks": event.get("committed_gen_chunks"),
                "committed_prefix_tokens": event.get("committed_prefix_tokens"),
                "last_chunk_ts": ts,
            }
        )

    def _start_request_key(
        self, event: dict[str, Any], role: str, replica_id: str, sample_id: str | None
    ) -> str:
        request_instance_id = event.get("request_instance_id")
        if request_instance_id is not None:
            return str(request_instance_id)
        sticky_key = self._legacy_sticky_key(event, role, replica_id, sample_id)
        self._legacy_seq += 1
        request_key = f"legacy:{self._legacy_seq}"
        self._legacy_request_queues[sticky_key].append(request_key)
        return request_key

    def _finish_request_key(self, event: dict[str, Any], role: str, replica_id: str) -> str | None:
        request_instance_id = event.get("request_instance_id")
        if request_instance_id is not None:
            return str(request_instance_id)
        sample_id = event.get("sample_id")
        sticky_key = self._legacy_sticky_key(event, role, replica_id, None if sample_id is None else str(sample_id))
        if not self._legacy_request_queues[sticky_key]:
            return None
        return self._legacy_request_queues[sticky_key].popleft()

    @staticmethod
    def _legacy_sticky_key(
        event: dict[str, Any], role: str, replica_id: str, sample_id: str | None
    ) -> str:
        return "|".join([role, replica_id, str(event.get("request_id")), str(sample_id)])

    def _snapshot_workers(self, now: float) -> dict[str, dict[str, Any]]:
        output = {}
        for key, worker in self.scheduler_workers.items():
            active_sample_ids = sorted(
                worker["active_sample_ids"],
                key=lambda sample_id: (
                    float(self.samples.get(sample_id, {}).get("launch_ts", now)),
                    str(sample_id),
                ),
            )
            active_samples = [self._sample_summary(sample_id, now) for sample_id in active_sample_ids]
            output[key] = {
                "worker_idx": worker["worker_idx"],
                "capacity": worker.get("capacity"),
                "active_sample_ids": active_sample_ids,
                "active_samples": active_samples,
                "active_by_source": _positive_counter(worker["active_by_source"]),
                "active_total": len(active_sample_ids),
            }
        return output

    def _snapshot_replicas(self, now: float) -> dict[str, dict[str, Any]]:
        output = {}
        for key, replica in self.replicas.items():
            active_requests = sorted(
                replica["active_request_keys"].values(),
                key=lambda request: (float(request.get("start_ts", now)), str(request.get("request_key"))),
            )
            durations = [float(value) for value in replica["recent_durations_ms"]]
            active_sample_ids = {request.get("sample_id") for request in active_requests if request.get("sample_id")}
            output[key] = {
                "key": replica["key"],
                "role": replica["role"],
                "replica_id": replica["replica_id"],
                "active_requests": len(active_requests),
                "active_sample_count": len(active_sample_ids),
                "request_count": int(replica["request_count"]),
                "finished_request_count": int(replica["finished_request_count"]),
                "error_request_count": int(replica["error_request_count"]),
                "active_by_source": _positive_counter(replica["active_by_source"]),
                "active_by_kind": _positive_counter(replica["active_by_kind"]),
                "recent_avg_ms": None if not durations else sum(durations) / len(durations),
                "recent_p95_ms": _percentile(durations, 0.95),
                "active_requests_detail": [
                    self._request_summary(request, now) for request in active_requests[:64]
                ],
            }
        return output

    def _snapshot_lt_candidate(self, now: float) -> dict[str, Any] | None:
        current_running = [
            sample
            for sample in self.samples.values()
            if sample.get("status") == "running" and sample.get("barrier_role") == "current" and "launch_ts" in sample
        ]
        if not current_running:
            return None
        lt_candidate = max(current_running, key=lambda sample: now - float(sample["launch_ts"]))
        return {
            "sample_id": lt_candidate["sample_id"],
            "elapsed_ms": (now - float(lt_candidate["launch_ts"])) * 1000,
            "scheduler_worker_idx": lt_candidate.get("scheduler_worker_idx"),
            "source_type": lt_candidate.get("source_type"),
            "chunk_idx": lt_candidate.get("chunk_idx"),
            "response_len": lt_candidate.get("response_len"),
        }

    def _sample_summary(self, sample_id: str, now: float) -> dict[str, Any]:
        sample = self.samples.get(sample_id, {"sample_id": sample_id})
        launch_ts = sample.get("launch_ts")
        elapsed_ms = None if launch_ts is None else (now - float(launch_ts)) * 1000
        return {
            "sample_id": sample_id,
            "source_type": sample.get("source_type"),
            "barrier_role": sample.get("barrier_role"),
            "chunk_idx": sample.get("chunk_idx"),
            "response_len": sample.get("response_len"),
            "committed_gen_chunks": sample.get("committed_gen_chunks"),
            "committed_prefix_tokens": sample.get("committed_prefix_tokens"),
            "elapsed_ms": elapsed_ms,
            "last_student_replica_id": sample.get("last_student_replica_id"),
            "last_teacher_replica_id": sample.get("last_teacher_replica_id"),
        }

    @staticmethod
    def _request_summary(request: dict[str, Any], now: float) -> dict[str, Any]:
        start_ts = request.get("start_ts")
        return {
            "request_instance_id": request.get("request_instance_id"),
            "request_id": request.get("request_id"),
            "sample_id": request.get("sample_id"),
            "source_type": request.get("source_type"),
            "barrier_role": request.get("barrier_role"),
            "request_kind": request.get("request_kind"),
            "prompt_len": request.get("prompt_len"),
            "elapsed_ms": None if start_ts is None else (now - float(start_ts)) * 1000,
        }


def reduce_events(events: list[dict[str, Any]], *, now: float | None = None) -> dict[str, Any]:
    reducer = DashboardReducer()
    for event in events:
        reducer.apply(event)
    return reducer.snapshot(now=now)


class DashboardStateStore:
    """Tails the JSONL event log and exposes cached dashboard state."""

    def __init__(self, event_log: str):
        self.event_log = event_log
        self._lock = threading.RLock()
        self._reducer = DashboardReducer()
        self._offset = 0
        self._lines_read = 0
        self._last_refresh_ts: float | None = None

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        self.refresh()
        with self._lock:
            state = self._reducer.snapshot(now=now)
            file_size = os.path.getsize(self.event_log) if os.path.exists(self.event_log) else 0
            state["reader"] = {
                "event_log": self.event_log,
                "offset": self._offset,
                "file_size": file_size,
                "lines_read": self._lines_read,
                "last_refresh_ts": self._last_refresh_ts,
            }
            return state

    def refresh(self) -> None:
        with self._lock:
            if not os.path.exists(self.event_log):
                return
            file_size = os.path.getsize(self.event_log)
            if file_size < self._offset:
                self._reducer = DashboardReducer()
                self._offset = 0
                self._lines_read = 0
            with open(self.event_log, encoding="utf-8") as event_file:
                event_file.seek(self._offset)
                while True:
                    start = event_file.tell()
                    line = event_file.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        event_file.seek(start)
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        self._offset = event_file.tell()
                        continue
                    self._reducer.apply(event)
                    self._lines_read += 1
                    self._offset = event_file.tell()
            self._last_refresh_ts = time.time()


def _read_events(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as event_file:
        for line in event_file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _dashboard_html(event_log: str) -> str:
    html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Async SKD Live Dashboard</title>
<style>
:root {
  --bg:#f4ead9; --paper:#fffaf0; --ink:#17130d; --muted:#675d4c; --line:#d0bfa6;
  --base:#005f82; --look:#008f54; --carry:#9a5e19; --idle:#d8cebd; --lt:#b13f2d;
  --student:#174c5f; --teacher:#5c3d76; --warn:#b13f2d;
}
* { box-sizing:border-box; }
body {
  margin:0; padding:24px; color:var(--ink);
  background:radial-gradient(circle at top left,#fff8ea 0,#f4ead9 38%,#ead8bd 100%);
  font-family:Aptos,Segoe UI,sans-serif;
}
main { max-width:1680px; margin:auto; display:grid; gap:18px; }
header,.panel {
  background:var(--paper); border:2px solid var(--ink); border-radius:22px;
  box-shadow:7px 7px 0 rgba(23,19,13,.13); padding:18px;
}
h1 { margin:0; font:800 clamp(32px,4vw,58px)/.9 Georgia,serif; letter-spacing:-.055em; }
h2 { margin:0 0 10px; font:800 24px/.95 Georgia,serif; letter-spacing:-.035em; }
.muted { color:var(--muted); }
.small { font-size:12px; }
.top { display:grid; grid-template-columns:1.2fr .8fr; gap:18px; align-items:stretch; }
.status-grid { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; }
.status-card { border:1px solid var(--line); border-radius:16px; padding:12px; background:#fffdf7; }
.metric { font:800 38px/.9 Georgia,serif; letter-spacing:-.045em; }
.legend { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:12px; }
.legend span { display:inline-flex; gap:6px; align-items:center; }
.dot {
  width:20px; height:20px; border-radius:50%; border:2px solid #fff8ea;
  box-shadow:0 0 0 1px rgba(23,19,13,.24); background:var(--idle); display:inline-block;
}
.base { background:var(--base); }
.lookahead { background:var(--look); }
.resumed_current { background:var(--carry); }
.unknown { background:#777; }
.ring { outline:4px solid var(--lt); outline-offset:2px; }
.worker-grid { display:grid; grid-template-columns:repeat(2,minmax(420px,1fr)); gap:14px; }
.worker-card { border:1px solid var(--line); border-radius:18px; padding:14px; background:#fffdf7; }
.worker-head,.replica-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.slot-grid { display:grid; grid-template-columns:repeat(12,22px); gap:8px; margin-top:14px; }
.slot { width:22px; height:22px; border-radius:999px; background:var(--idle); border:2px solid #fff8ea; box-shadow:0 0 0 1px rgba(23,19,13,.22); }
.slot.base_current { background:var(--base); }
.slot.lookahead { background:var(--look); }
.slot.resumed_current { background:var(--carry); }
.slot.lt { outline:4px solid var(--lt); outline-offset:2px; }
.replica-section { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.replica-list { display:grid; gap:12px; }
.replica-card { border:1px solid var(--line); border-radius:18px; padding:14px; background:#fffdf7; }
.replica-card.student { border-left:10px solid var(--student); }
.replica-card.teacher { border-left:10px solid var(--teacher); }
.replica-card.hot { box-shadow:inset 0 0 0 3px color-mix(in srgb,var(--warn) 45%,transparent); }
.request-row { display:flex; justify-content:space-between; gap:10px; border-top:1px solid var(--line); padding-top:7px; margin-top:7px; }
.bar { height:14px; display:flex; border:1px solid var(--line); border-radius:999px; overflow:hidden; background:var(--idle); margin-top:10px; }
.bar span { height:100%; display:block; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip { border:1px solid var(--line); border-radius:999px; padding:4px 8px; background:#fff7e8; font-size:12px; }
.stream { max-height:120px; overflow:auto; white-space:pre-wrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
@media (max-width: 980px) {
  body { padding:12px; }
  .top,.replica-section,.worker-grid { grid-template-columns:1fr; }
  .status-grid { grid-template-columns:repeat(2,1fr); }
}
</style>
</head>
<body>
<main>
<header>
  <h1>Async SKD live control panel</h1>
  <p class="muted">한 화면에서 scheduler worker slot, student generation, teacher verification을 분리해서 본다. event log: <code>__EVENT_LOG__</code></p>
  <div class="legend">
    <span><i class="dot base"></i>base_current</span>
    <span><i class="dot lookahead"></i>lookahead</span>
    <span><i class="dot resumed_current"></i>resumed_current</span>
    <span><i class="dot ring"></i>LT candidate</span>
  </div>
</header>
<section class="top">
  <div class="panel" id="status"></div>
  <div class="panel"><h2>Event stream</h2><p class="muted small stream" id="stream">waiting...</p></div>
</section>
<section class="panel"><h2>Scheduler worker lanes</h2><div class="worker-grid" id="workers"></div></section>
<section class="replica-section">
  <div class="panel"><h2>Student generation replicas</h2><div class="replica-list" id="students"></div></div>
  <div class="panel"><h2>Teacher verification replicas</h2><div class="replica-list" id="teachers"></div></div>
</section>
</main>
<script>
let state = null;
let loading = false;
let pending = false;
let scheduled = null;
let lastLoadMs = 0;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function sourceClass(source) {
  if (source === 'lookahead') return 'lookahead';
  if (source === 'resumed_current') return 'resumed_current';
  if (source === 'base_current') return 'base_current';
  return 'unknown';
}
function fmtMs(value) {
  if (value === null || value === undefined) return '-';
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}
function countSource(map, key) { return Number((map || {})[key] || 0); }

async function loadState() {
  if (loading) { pending = true; return; }
  loading = true;
  try {
    const res = await fetch('/state', { cache: 'no-store' });
    state = await res.json();
    lastLoadMs = Date.now();
    render();
  } finally {
    loading = false;
    if (pending) {
      pending = false;
      scheduleLoad(250);
    }
  }
}
function scheduleLoad(delay = 0) {
  if (scheduled !== null) return;
  const minGap = 1000;
  const wait = Math.max(delay, minGap - (Date.now() - lastLoadMs));
  scheduled = setTimeout(() => {
    scheduled = null;
    loadState();
  }, wait);
}

function renderStatus() {
  const lag = state.event_lag_s;
  const samples = Object.values(state.samples || {});
  const running = samples.filter(s => s.status === 'running').length;
  const lt = state.lt_candidate;
  document.getElementById('status').innerHTML = `
    <h2>Live status</h2>
    <div class="status-grid">
      <div class="status-card"><div class="metric">${fmtMs((lag || 0) * 1000)}</div><p class="muted small">event lag</p></div>
      <div class="status-card"><div class="metric">${running}</div><p class="muted small">running samples</p></div>
      <div class="status-card"><div class="metric">${Object.keys(state.scheduler_workers || {}).length}</div><p class="muted small">scheduler workers</p></div>
      <div class="status-card"><div class="metric">${Object.keys(state.replicas || {}).length}</div><p class="muted small">replicas</p></div>
    </div>
    <p class="muted">LT candidate: ${lt ? `${esc(lt.sample_id)} · worker ${esc(lt.scheduler_worker_idx)} · ${esc(lt.source_type)} · chunk ${esc(lt.chunk_idx ?? '-')}` : 'none'}</p>`;
}

function renderWorkers() {
  const ltId = state.lt_candidate && state.lt_candidate.sample_id;
  const workers = Object.values(state.scheduler_workers || {}).sort((a, b) => String(a.worker_idx).localeCompare(String(b.worker_idx)));
  document.getElementById('workers').innerHTML = workers.map(w => {
    const cap = Number(w.capacity || Math.max(1, w.active_total || 1));
    const samples = w.active_samples || [];
    const slots = Array.from({ length: cap }, (_, idx) => {
      const sample = samples[idx];
      if (!sample) return '<i class="slot" title="idle"></i>';
      const cls = sourceClass(sample.source_type);
      const lt = sample.sample_id === ltId ? ' lt' : '';
      const title = `${sample.sample_id} | ${sample.source_type} | chunks=${sample.committed_gen_chunks ?? '-'} | resp=${sample.response_len ?? '-'}`;
      return `<i class="slot ${cls}${lt}" title="${esc(title)}"></i>`;
    }).join('');
    return `<article class="worker-card">
      <div class="worker-head"><h2>worker ${esc(w.worker_idx)}</h2><div class="metric">${w.active_total}/${cap}</div></div>
      <p class="muted small">base ${countSource(w.active_by_source,'base_current')} · lookahead ${countSource(w.active_by_source,'lookahead')} · resumed ${countSource(w.active_by_source,'resumed_current')}</p>
      <div class="slot-grid">${slots}</div>
    </article>`;
  }).join('');
}

function renderReplicaCard(rep) {
  const active = Number(rep.active_requests || 0);
  const hot = rep.role === 'teacher' && active >= 24 ? ' hot' : '';
  const source = rep.active_by_source || {};
  const total = Math.max(1, countSource(source,'base_current') + countSource(source,'lookahead') + countSource(source,'resumed_current') + countSource(source,'unknown'));
  const details = (rep.active_requests_detail || []).slice(0, 6).map(req => `
    <div class="request-row">
      <span>${esc((req.sample_id || '').slice(0, 8))} · ${esc(req.source_type)} · ${esc(req.request_kind)}</span>
      <span>${fmtMs(req.elapsed_ms)}</span>
    </div>`).join('');
  return `<article class="replica-card ${esc(rep.role)}${hot}">
    <div class="replica-head"><h2>${esc(rep.replica_id)}</h2><div class="metric">${active}</div></div>
    <p class="muted small">${rep.role === 'teacher' ? 'active verification requests' : 'active generation requests'} · samples ${rep.active_sample_count || 0}</p>
    <div class="bar">
      <span style="width:${countSource(source,'base_current') / total * 100}%;background:var(--base)"></span>
      <span style="width:${countSource(source,'lookahead') / total * 100}%;background:var(--look)"></span>
      <span style="width:${countSource(source,'resumed_current') / total * 100}%;background:var(--carry)"></span>
    </div>
    <div class="chips">
      <span class="chip">avg ${fmtMs(rep.recent_avg_ms)}</span>
      <span class="chip">p95 ${fmtMs(rep.recent_p95_ms)}</span>
      <span class="chip">requests ${rep.request_count}</span>
      <span class="chip">errors ${rep.error_request_count}</span>
    </div>
    ${details || '<p class="muted small">no active request detail</p>'}
  </article>`;
}

function renderReplicas() {
  const replicas = Object.values(state.replicas || {});
  document.getElementById('students').innerHTML = replicas.filter(r => r.role === 'student').map(renderReplicaCard).join('');
  document.getElementById('teachers').innerHTML = replicas.filter(r => r.role === 'teacher').map(renderReplicaCard).join('');
}

function render() {
  renderStatus();
  renderWorkers();
  renderReplicas();
}

loadState();
const es = new EventSource('/events');
es.onmessage = ev => {
  document.getElementById('stream').textContent = ev.data.slice(0, 360);
  scheduleLoad(200);
};
es.onerror = () => scheduleLoad(1000);
setInterval(() => scheduleLoad(0), 1500);
</script>
</body>
</html>"""
    return html.replace("__EVENT_LOG__", event_log)


class _DashboardHandler(BaseHTTPRequestHandler):
    event_log = ""
    state_store: DashboardStateStore | None = None

    def _send(self, status: int, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, _dashboard_html(self.event_log), "text/html; charset=utf-8")
            return
        if parsed.path == "/state":
            store = self.state_store
            if store is None:
                store = DashboardStateStore(self.event_log)
                self.__class__.state_store = store
            state = store.snapshot()
            self._send(200, json.dumps(state, ensure_ascii=False, default=str), "application/json")
            return
        if parsed.path == "/events":
            self._stream_events()
            return
        self._send(404, "not found", "text/plain")

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        offset = os.path.getsize(self.event_log) if os.path.exists(self.event_log) else 0
        while True:
            try:
                if os.path.exists(self.event_log):
                    with open(self.event_log, encoding="utf-8") as event_file:
                        event_file.seek(offset)
                        sent = False
                        while True:
                            start = event_file.tell()
                            line = event_file.readline()
                            if not line:
                                break
                            if not line.endswith("\n"):
                                event_file.seek(start)
                                break
                            self.wfile.write(f"data: {line.strip()}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            sent = True
                            offset = event_file.tell()
                        if not sent:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve async SKD event dashboard.")
    parser.add_argument("--event-log", default=os.getenv("VERL_ASYNC_SKD_EVENT_LOG", "logs/async_skd_events.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    _DashboardHandler.event_log = args.event_log
    _DashboardHandler.state_store = DashboardStateStore(args.event_log)
    server = ThreadingHTTPServer((args.host, args.port), _DashboardHandler)
    print(f"async SKD dashboard: http://{args.host}:{args.port} event_log={args.event_log}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
