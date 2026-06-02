"""Prepare multimodal JSONL data for GPO-V probing.

Unsafe rows are harmful text prompts paired with the benign GPO-V image. The
probe runner later adds the GPO-V perturbation for these rows. Safe rows are
benign text prompts paired with the same image and are left unperturbed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


DATASET_NAME = "saralazza/llada-safety-dataset"
SAFE_SOURCE = "alpaca"
PROMPT_FIELDS = (
    "refined_behavior",
    "behavior",
    "goal",
    "prompt",
    "instruction",
    "question",
    "query",
    "redteam_query",
    "jailbreak_query",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multimodal GPO-V JSONL data.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("data/multimodal_gpo/llada_v_gpo_prompts.jsonl"))
    parser.add_argument("--image-path", type=Path, default=Path("gpo_v/assets/example_input_image.jpg"))
    parser.add_argument(
        "--unsafe-file",
        type=Path,
        default=None,
        help="Optional AdvBench/JailBreakBench-style CSV, JSON, or JSONL file.",
    )
    parser.add_argument(
        "--unsafe-field",
        default=None,
        help="Prompt column/key for --unsafe-file. If omitted, common prompt fields are tried.",
    )
    parser.add_argument("--unsafe-source-name", default=None)
    parser.add_argument(
        "--safe-file",
        type=Path,
        default=None,
        help="Optional Alpaca/XSTest-safe-style CSV, JSON, or JSONL file.",
    )
    parser.add_argument(
        "--safe-field",
        default=None,
        help="Prompt column/key for --safe-file. If omitted, common prompt fields are tried.",
    )
    parser.add_argument("--safe-source-name", default=None)
    parser.add_argument("--max-safe", type=int, default=None)
    parser.add_argument("--max-unsafe", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def prompt_from_row(row: object, explicit_field: str | None = None) -> str:
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return ""

    if explicit_field:
        return str(row.get(explicit_field) or "").strip()

    for field in PROMPT_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def read_json_rows(path: Path) -> list[object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, (dict, str))]
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "prompts"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, (dict, str))]
    raise ValueError(f"Could not find a list of rows in {path}")


def read_jsonl_rows(path: Path) -> list[object]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, (dict, str)):
                rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_local_rows(path: Path) -> list[object]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl_rows(path)
    if suffix == ".json":
        return read_json_rows(path)
    if suffix in {".csv", ".tsv"}:
        if suffix == ".tsv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle, delimiter="\t"))
        return read_csv_rows(path)
    raise ValueError(f"Unsupported prompt file extension for {path}")


def make_records(
    rows: Iterable[object],
    *,
    label: int,
    source_name: str,
    image_path: str,
    prompt_field: str | None,
    max_rows: int | None,
) -> list[dict]:
    records = []
    for idx, row in enumerate(rows):
        prompt = prompt_from_row(row, prompt_field)
        if not prompt:
            continue

        records.append(
            {
                "id": f"{source_name}-{idx}",
                "prompt": prompt,
                "label": label,
                "source": source_name,
                "image_path": image_path,
            }
        )

        if max_rows is not None and len(records) >= max_rows:
            break
    return records


def load_default_safety_dataset(args: argparse.Namespace) -> list[dict]:
    ds = load_dataset(args.dataset_name, split=args.split)
    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed)
    return list(ds)


def build_safe_rows(args: argparse.Namespace, dataset_rows: list[dict] | None, image_path: str) -> list[dict]:
    if args.safe_file:
        source_name = args.safe_source_name or args.safe_file.stem
        rows = read_local_rows(args.safe_file)
        if not args.no_shuffle:
            random.Random(args.seed).shuffle(rows)
        return make_records(
            rows,
            label=0,
            source_name=source_name,
            image_path=image_path,
            prompt_field=args.safe_field,
            max_rows=args.max_safe,
        )

    source_name = args.safe_source_name or SAFE_SOURCE
    safe_rows = (
        row
        for row in dataset_rows or []
        if str(row.get("source_dataset") or "").strip().lower() == SAFE_SOURCE
    )
    return make_records(
        safe_rows,
        label=0,
        source_name=source_name,
        image_path=image_path,
        prompt_field=None,
        max_rows=args.max_safe,
    )


def build_unsafe_rows(args: argparse.Namespace, dataset_rows: list[dict] | None, image_path: str) -> list[dict]:
    if args.unsafe_file:
        source_name = args.unsafe_source_name or args.unsafe_file.stem
        rows = read_local_rows(args.unsafe_file)
        if not args.no_shuffle:
            random.Random(args.seed + 1).shuffle(rows)
        return make_records(
            rows,
            label=1,
            source_name=source_name,
            image_path=image_path,
            prompt_field=args.unsafe_field,
            max_rows=args.max_unsafe,
        )

    source_name = args.unsafe_source_name or "dija_non_alpaca"
    unsafe_rows = (
        row
        for row in dataset_rows or []
        if str(row.get("source_dataset") or "").strip().lower() != SAFE_SOURCE
    )
    return make_records(
        unsafe_rows,
        label=1,
        source_name=source_name,
        image_path=image_path,
        prompt_field=None,
        max_rows=args.max_unsafe,
    )


def main() -> None:
    args = parse_args()
    image_path = str(args.image_path)
    needs_default_dataset = args.safe_file is None or args.unsafe_file is None
    dataset_rows = load_default_safety_dataset(args) if needs_default_dataset else None

    safe_rows = build_safe_rows(args, dataset_rows, image_path)
    unsafe_rows = build_unsafe_rows(args, dataset_rows, image_path)
    rows = safe_rows + unsafe_rows
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"safe={len(safe_rows)} unsafe={len(unsafe_rows)} image={image_path}")
    print("Unsafe rows receive GPO-V perturbations in run_gpo_v_lladav_probe.py.")
    print("Safe rows use the same image without perturbation.")


if __name__ == "__main__":
    main()
