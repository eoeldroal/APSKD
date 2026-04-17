# Copyright 2026 DDAI Research
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
Preprocess the Nemotron-Cascade-RL-Math dataset into a multi-turn, tool-using VERL parquet format.

Notes:
- Uses the full raw train split as training data.
- Keeps semantic answer-format constraints such as "as a decimal", "in base 10",
  or "without units", but removes redundant instructions that ask the model to
  place the final answer inside boxed/oxed markup. The global system prompt
  already standardizes final answers into \\boxed{}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs


SYSTEM_PROMPT = """You are a careful mathematical problem-solving assistant.

Guidelines:
1. Reason about the problem before calling the tool.
2. Use the `code_interpreter` tool when it helps you solve or verify the math problem more reliably.
3. Each tool call must contain complete Python code.
4. Do not rely on hidden state from previous tool calls.
5. Keep the final answer only in the last turn, and put it inside \\boxed{}.

A short example:
Question: How many integers n from 1 to 20 are divisible by 3 or 5?

Assistant: I can count this with Python to avoid mistakes.
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "count = sum(1 for n in range(1, 21) if n % 3 == 0 or n % 5 == 0)\\nprint(count)"}}
</tool_call>

Tool:
9

Assistant: The tool shows there are 9 such integers, so the final answer is \\boxed{9}."""


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")
_BOXING_SENTENCE_RE = re.compile(
    r"""
    (?:
        please\s+put\s+your\s+final\s+answer.*?(?:boxed|oxed)\s*\{[^}]*\}\.?
        |
        put\s+your\s+final\s+answer.*?(?:boxed|oxed)\s*\{[^}]*\}\.?
        |
        your\s+final\s+answer.*?(?:boxed|oxed)\s*\{[^}]*\}\.?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_task_type(problem: str) -> str:
    lower = problem.lower()
    if "[solution]" in lower or "<paragraph_" in lower or "review and critique" in lower or "earliest error" in lower:
        return "solution_critique"
    return "direct_math"


def normalize_problem(problem: str) -> str:
    """Remove only outer boxed-answer instructions while preserving task semantics."""
    cleaned = _CONTROL_CHAR_RE.sub("", problem)
    cleaned = _BOXING_SENTENCE_RE.sub("", cleaned)

    # Normalize blank lines introduced by sentence removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def load_raw_dataset(local_dataset_path: str | None):
    if local_dataset_path is not None:
        local_path = Path(local_dataset_path).expanduser()
        if local_path.is_dir():
            return datasets.load_from_disk(str(local_path))
        return datasets.load_dataset(str(local_path))
    return datasets.load_dataset("nvidia/Nemotron-Cascade-RL-Math")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Local raw dataset path or dataset script path.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/nemotron_cascade_rl_math_multiturn_w_tool",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    data_source = "nvidia/Nemotron-Cascade-RL-Math"
    print(f"Loading the {data_source} dataset...", flush=True)
    dataset = load_raw_dataset(args.local_dataset_path)
    train_dataset = dataset["train"]

    def process_fn(example, idx):
        problem = normalize_problem(example["problem"])
        answer = str(example["answer"]).strip()
        source = example.get("source", "")
        task_type = detect_task_type(example["problem"])
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "source": source,
                "task_type": task_type,
                "problem": problem,
                "answer": answer,
            },
        }

    train_dataset = train_dataset.map(function=process_fn, with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, "train.parquet")
    train_dataset.to_parquet(train_path)

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
