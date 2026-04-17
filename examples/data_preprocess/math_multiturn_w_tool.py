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
Preprocess the MATH-lighteval dataset to a multi-turn, tool-using VERL parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


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


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/math_multiturn_w_tool",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            answer = example.pop("solution")
            solution = extract_solution(answer)
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "problem": problem,
                    "solution": answer,
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
