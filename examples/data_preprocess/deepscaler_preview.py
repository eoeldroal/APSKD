"""
Preprocess the DeepScaleR Preview dataset to VERL parquet format for train-time distillation.
"""

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    end = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0 and i > idx:
                end = i
                break
        i += 1

    return None if end is None else text[idx : end + 1]


def remove_boxed(text: str | None) -> str | None:
    if text is None:
        return None
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text[len("\\boxed{") : -1]
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]
    return text


def normalize_for_match(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"^\(?[A-E]\)?\s+", "", text)
    text = re.sub(r"^\(?[A-E]\)?", "", text)
    text = text.replace(" ", "")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/deepscaler", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--data_source_name", default="deepscaler_preview")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    hf_data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(hf_data_source)

    train_dataset = dataset["train"]

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    seen_problems = set()
    stats = {
        "source_dataset": hf_data_source,
        "output_data_source": args.data_source_name,
        "total_rows": len(train_dataset),
        "kept_rows": 0,
        "dropped_missing_problem": 0,
        "dropped_missing_answer": 0,
        "dropped_duplicate_problem": 0,
        "boxed_solution_present": 0,
        "boxed_solution_matches_answer_simple": 0,
    }

    processed_rows = []
    for idx, example in enumerate(train_dataset):
        problem = example.get("problem")
        answer = example.get("answer")
        solution = example.get("solution")

        if problem is None or str(problem).strip() == "":
            stats["dropped_missing_problem"] += 1
            continue
        if answer is None or str(answer).strip() == "":
            stats["dropped_missing_answer"] += 1
            continue
        if problem in seen_problems:
            stats["dropped_duplicate_problem"] += 1
            continue
        seen_problems.add(problem)

        answer = str(answer).strip()
        solution = None if solution is None else str(solution)

        boxed_solution = remove_boxed(extract_last_boxed(solution))
        if boxed_solution is not None:
            stats["boxed_solution_present"] += 1
            if normalize_for_match(boxed_solution) == normalize_for_match(answer):
                stats["boxed_solution_matches_answer_simple"] += 1

        processed_rows.append(
            {
                "data_source": args.data_source_name,
                "prompt": [{"role": "user", "content": f"{problem} {instruction_following}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "original_data_source": hf_data_source,
                    "answer": answer,
                    "question": problem,
                    "solution": solution,
                    "boxed_solution": boxed_solution,
                },
            }
        )

    stats["kept_rows"] = len(processed_rows)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_output_path = os.path.join(local_dir, "train.parquet")
    datasets.Dataset.from_list(processed_rows).to_parquet(train_output_path)

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(processed_rows[0], f, indent=2)

    with open(os.path.join(local_dir, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))

    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
