"""Verify a completed full multimodal GPO-V probe run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final multimodal checkpoint probe artifacts.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--expected-safe-count", type=int, default=200)
    parser.add_argument("--expected-unsafe-count", type=int, default=200)
    parser.add_argument("--expected-unsafe-attack-steps", type=int, default=5)
    parser.add_argument("--expected-hidden-dim", type=int, default=4096)
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[1, 5, 10, 15])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_stats(arr: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(arr, axis=1)
    return {
        "row_norm_min": float(norms.min()),
        "row_norm_mean": float(norms.mean()),
        "row_norm_max": float(norms.max()),
        "element_min": float(arr.min()),
        "element_mean": float(arr.mean()),
        "element_std": float(arr.std()),
        "element_max": float(arr.max()),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    npz_path = checkpoint_dir / "checkpoint_probes.npz"
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    state_path = checkpoint_dir / "checkpoint_state.json"

    if not npz_path.exists():
        npz_path = checkpoint_dir / "probes.npz"
    if not metadata_path.exists():
        metadata_path = checkpoint_dir / "metadata.json"

    state = load_json(state_path) if state_path.exists() else {}
    metadata = load_json(metadata_path)
    data = np.load(npz_path)

    require(isinstance(metadata, list), "metadata must be a JSON list")
    require(len(metadata) == args.expected_count, f"metadata rows={len(metadata)} expected={args.expected_count}")
    if state:
        processed_count = int(state.get("processed_count", -1))
        require(
            processed_count == args.expected_count,
            f"checkpoint_state processed_count={processed_count} expected={args.expected_count}",
        )

    require("labels" in data.files, "checkpoint/probe npz is missing labels")
    labels = data["labels"].astype(np.int64)
    require(labels.shape == (args.expected_count,), f"labels shape={labels.shape} expected=({args.expected_count},)")
    safe_count = int((labels == 0).sum())
    unsafe_count = int((labels == 1).sum())
    require(safe_count == args.expected_safe_count, f"safe labels={safe_count} expected={args.expected_safe_count}")
    require(
        unsafe_count == args.expected_unsafe_count,
        f"unsafe labels={unsafe_count} expected={args.expected_unsafe_count}",
    )

    for idx, (label, row) in enumerate(zip(labels.tolist(), metadata, strict=True)):
        require(int(row.get("label")) == label, f"metadata label mismatch at row {idx}")
        if label == 0:
            require(row.get("label_name") == "safe", f"row {idx} expected label_name=safe")
            require(int(row.get("attack_steps_run", -1)) == 0, f"safe row {idx} has attack_steps_run != 0")
        elif label == 1:
            require(row.get("label_name") == "unsafe", f"row {idx} expected label_name=unsafe")
            require(
                int(row.get("attack_steps_run", -1)) == args.expected_unsafe_attack_steps,
                f"unsafe row {idx} has attack_steps_run != {args.expected_unsafe_attack_steps}",
            )
        else:
            raise AssertionError(f"Unexpected label at row {idx}: {label}")

    summary: dict[str, object] = {
        "checkpoint_dir": str(checkpoint_dir),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "state_path": str(state_path) if state_path.exists() else None,
        "samples": args.expected_count,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "probe_steps": args.probe_steps,
        "layers": args.layers,
        "stats": {},
    }

    stats = summary["stats"]
    assert isinstance(stats, dict)
    for step in args.probe_steps:
        step_stats = {}
        for layer in args.layers:
            key = f"step_{step}_layer_{layer}"
            require(key in data.files, f"missing probe key: {key}")
            arr = data[key]
            require(
                arr.shape == (args.expected_count, args.expected_hidden_dim),
                f"{key} shape={arr.shape} expected=({args.expected_count}, {args.expected_hidden_dim})",
            )
            require(np.isfinite(arr).all(), f"{key} contains NaN or Inf")
            safe_arr = arr[labels == 0]
            unsafe_arr = arr[labels == 1]
            require(
                safe_arr.shape == (args.expected_safe_count, args.expected_hidden_dim),
                f"{key} safe split shape={safe_arr.shape}",
            )
            require(
                unsafe_arr.shape == (args.expected_unsafe_count, args.expected_hidden_dim),
                f"{key} unsafe split shape={unsafe_arr.shape}",
            )
            step_stats[f"layer_{layer}"] = {
                "safe": finite_stats(safe_arr),
                "unsafe": finite_stats(unsafe_arr),
            }
        stats[f"step_{step}"] = step_stats

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Verified multimodal probe run: "
        f"samples={args.expected_count} safe={safe_count} unsafe={unsafe_count} "
        f"steps={args.probe_steps} layers={len(args.layers)}"
    )


if __name__ == "__main__":
    main()
