# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import ray

from verl.tools.sandbox_fusion_tools import ExecutionWorker


def test_execution_worker_without_rate_limit_executes_function():
    worker = ExecutionWorker(enable_global_rate_limit=False)

    result = worker.execute(lambda a, b: a + b, 1, 2)

    assert result == 3


def test_execution_worker_with_rate_limit_acquires_and_releases(monkeypatch):
    calls = []

    class _RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self):
            calls.append(self.name)
            return self.name

    fake_rate_limit_worker = SimpleNamespace(
        acquire=_RemoteMethod("acquire"),
        release=_RemoteMethod("release"),
    )

    monkeypatch.setattr(ray, "get", lambda value: value)

    worker = ExecutionWorker(enable_global_rate_limit=False)
    worker.rate_limit_worker = fake_rate_limit_worker

    result = worker.execute(lambda: "ok")

    assert result == "ok"
    assert calls == ["acquire", "release"]
