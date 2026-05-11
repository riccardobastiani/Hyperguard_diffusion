import argparse
import csv
import json
import os
import random
import sys
import tempfile
import urllib.request


HF_DATASET = "tatsu-lab/alpaca"
DEFAULT_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/alpaca_data.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic random subset from tatsu-lab/alpaca."
    )
    parser.add_argument("--dataset-name", default=HF_DATASET)
    parser.add_argument("--source-json", default=None)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--sample-size", type=int, default=689)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def load_with_hf_datasets(dataset_name):
    cwd = os.getcwd()
    sys.path = [p for p in sys.path if p not in ("", cwd)]

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    return [dict(row) for row in ds]


def load_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def download_json(url):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        return load_from_json(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_alpaca(args):
    if args.source_json:
        return load_from_json(args.source_json)

    try:
        return load_with_hf_datasets(args.dataset_name)
    except Exception as exc:
        print(f"HF datasets load failed ({exc}); falling back to direct JSON download.")
        return download_json(args.source_url)


def build_goal(instruction, input_text):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def normalize_row(row, source_index):
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")
    goal = build_goal(instruction, input_text)
    return {
        "source": HF_DATASET,
        "source_index": source_index,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "goal": goal,
        "target": output,
    }


def main():
    args = parse_args()
    data = load_alpaca(args)

    if args.sample_size > len(data):
        raise ValueError(
            f"Requested {args.sample_size} rows, but dataset only has {len(data)} rows."
        )

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(data)), args.sample_size))
    subset = [normalize_row(data[i], i) for i in sampled_indices]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    fieldnames = [
        "source",
        "source_index",
        "instruction",
        "input",
        "output",
        "goal",
        "target",
    ]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)

    print(f"Saved {len(subset)} Alpaca rows to {args.output_json}")
    print(f"Saved CSV copy to {args.output_csv}")


if __name__ == "__main__":
    main()
