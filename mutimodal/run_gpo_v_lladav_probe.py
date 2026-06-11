"""Run GPO-V on LLaDA-V and save multimodal hidden-state probes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gpo_v.lladav import (
    ActivationRecorder,
    build_prompt,
    find_layer_stack,
    initialize_response_span,
    load_lladav_assets,
    model_dtype,
    optimize_image_with_gpo,
    prepare_image,
    validation_generate,
)


LOGGER = logging.getLogger("run_gpo_v_lladav_probe")

# Upstream GPO-V code can call breakpoint() in model loading paths.
os.environ.setdefault("PYTHONBREAKPOINT", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPO-V LLaDA-V multimodal activation probe runner.")
    parser.add_argument("--gpo-v-root", type=Path, required=True, help="Path to upstream GPO-V/LLaDA-V folder.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal_gpo/probes"))
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-V")
    parser.add_argument("--model-alias", default="llava_llada")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load-8bit", action="store_true", help="Load model with 8-bit quantization.")
    parser.add_argument("--load-4bit", action="store_true", help="Load model with 4-bit quantization.")
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[16, 23, 29])
    parser.add_argument("--all-layers", action="store_true", help="Capture activations for all transformer layers.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attack-steps", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--generation-steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--skip-attack", action="store_true", help="Smoke-test mode: do not optimize unsafe images.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Checkpoint every N processed samples (0 disables periodic checkpoints).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint files in output-dir.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_jsonl(path: Path, max_samples: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def append_feature(buffers: dict[str, list], step: int, layer_id: int, values: list[np.ndarray]) -> None:
    key = f"step_{step}_layer_{layer_id}"
    if not values:
        raise RuntimeError(f"No activation captured for {key}.")
    arr = np.concatenate(values, axis=0)
    buffers.setdefault(key, []).append(arr)


def _checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "npz": output_dir / "checkpoint_probes.npz",
        "meta": output_dir / "checkpoint_metadata.json",
        "state": output_dir / "checkpoint_state.json",
    }


def _atomic_write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as tmp:
        np.savez(tmp, **arrays)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _flatten_feature_buffers(feature_buffers: dict[str, list], labels: list[int]) -> dict[str, np.ndarray]:
    arrays = {key: np.concatenate(values, axis=0) for key, values in feature_buffers.items()}
    arrays["labels"] = np.asarray(labels, dtype=np.int64)
    return arrays


def _save_checkpoint(
    output_dir: Path,
    feature_buffers: dict[str, list],
    labels: list[int],
    metadata: list[dict],
    state: dict,
) -> None:
    paths = _checkpoint_paths(output_dir)
    arrays = _flatten_feature_buffers(feature_buffers, labels)
    _atomic_write_npz(paths["npz"], arrays)
    _atomic_write_json(paths["meta"], metadata)
    _atomic_write_json(paths["state"], state)


def _try_resume(
    output_dir: Path,
    feature_buffers: dict[str, list],
    metadata: list[dict],
) -> tuple[list[int], int, dict] | None:
    paths = _checkpoint_paths(output_dir)
    if not (paths["npz"].exists() and paths["meta"].exists() and paths["state"].exists()):
        return None

    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    checkpoint_meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    checkpoint_npz = np.load(paths["npz"])

    labels = checkpoint_npz["labels"].astype(np.int64).tolist()
    for key in checkpoint_npz.files:
        if key == "labels":
            continue
        feature_buffers[key] = [checkpoint_npz[key]]
    metadata.extend(checkpoint_meta)

    processed = int(state.get("processed_count", len(labels)))
    return labels, processed, state


def main() -> None:
    configure_logging()
    args = parse_args()
    rows = read_jsonl(args.input_jsonl, args.max_samples)
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    dtype = model_dtype(args.dtype)
    tokenizer, model, image_processor, _ = load_lladav_assets(
        gpo_v_root=args.gpo_v_root,
        model_name=args.model_name,
        model_alias=args.model_alias,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )

    if args.all_layers:
        layer_path, layers = find_layer_stack(model)
        args.layer_ids = list(range(len(layers)))
        LOGGER.info("Using all layers from %s: count=%d", layer_path, len(args.layer_ids))

    device = next(model.parameters()).device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_buffers: dict[str, list] = {}
    labels: list[int] = []
    metadata: list[dict] = []

    start_index = 0
    if args.resume:
        resumed = _try_resume(args.output_dir, feature_buffers, metadata)
        if resumed is not None:
            labels, start_index, state = resumed
            expected_probe_steps = state.get("probe_steps")
            expected_layer_ids = state.get("layer_ids")
            if expected_probe_steps is not None and list(expected_probe_steps) != list(args.probe_steps):
                raise ValueError("Checkpoint probe_steps do not match current --probe-steps.")
            if expected_layer_ids is not None and list(expected_layer_ids) != list(args.layer_ids):
                raise ValueError("Checkpoint layer_ids do not match current layer selection.")
            LOGGER.info("Resuming from checkpoint: processed=%d", start_index)
        else:
            LOGGER.info("No checkpoint found in %s; starting from scratch.", args.output_dir)

    if start_index >= len(rows):
        LOGGER.info("Nothing to do: checkpoint already covers all %d rows.", len(rows))
        arrays = _flatten_feature_buffers(feature_buffers, labels)
        _atomic_write_npz(args.output_dir / "probes.npz", arrays)
        _atomic_write_json(args.output_dir / "metadata.json", metadata)
        LOGGER.info("Saved probes to %s", args.output_dir / "probes.npz")
        return

    for row_index, row in enumerate(tqdm(rows[start_index:], desc="GPO-V probes"), start=start_index):
        input_ids = build_prompt(args.gpo_v_root, tokenizer, row["prompt"]).to(device)
        image_tensor, image_sizes = prepare_image(
            args.gpo_v_root,
            row["image_path"],
            image_processor,
            model,
            device,
            dtype=dtype,
        )

        label = int(row["label"])
        attack_history = []
        if label == 1 and not args.skip_attack:
            final_image, attack_history, span = optimize_image_with_gpo(
                model,
                input_ids,
                image_tensor,
                image_sizes,
                tokenizer,
                attack_steps=args.attack_steps,
                epsilon=args.epsilon,
                lr=args.lr,
                generation_steps=args.generation_steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
            )
        else:
            final_image = image_tensor
            span = initialize_response_span(
                model,
                input_ids,
                image_tensor.to(dtype=torch.float16),
                image_sizes,
                tokenizer,
                args.generation_steps,
                args.gen_length,
                args.block_length,
            )

        with ActivationRecorder(
            model=model,
            layer_ids=args.layer_ids,
            probe_steps=args.probe_steps,
            pool_start=span[0],
            pool_end=span[1],
        ) as recorder:
            _tokens, response = validation_generate(
                model,
                input_ids,
                final_image,
                image_sizes,
                tokenizer,
                generation_steps=args.generation_steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
            )

        for step in args.probe_steps:
            for layer_id in args.layer_ids:
                append_feature(feature_buffers, step, layer_id, recorder.features[step][layer_id])

        labels.append(label)
        metadata.append(
            {
                "id": row.get("id"),
                "label": label,
                "label_name": "unsafe" if label == 1 else "safe",
                "source": row.get("source"),
                "prompt": row.get("prompt"),
                "image_path": row.get("image_path"),
                "response": response,
                "attack_steps_run": len(attack_history),
                "span": [span[0], span[1]],
            }
        )

        processed_count = row_index + 1
        if args.save_every > 0 and processed_count % args.save_every == 0:
            checkpoint_state = {
                "processed_count": processed_count,
                "probe_steps": list(args.probe_steps),
                "layer_ids": list(args.layer_ids),
            }
            _save_checkpoint(args.output_dir, feature_buffers, labels, metadata, checkpoint_state)
            LOGGER.info("Checkpoint saved at sample %d", processed_count)

    arrays = _flatten_feature_buffers(feature_buffers, labels)
    _atomic_write_npz(args.output_dir / "probes.npz", arrays)
    _atomic_write_json(args.output_dir / "metadata.json", metadata)

    checkpoint_paths = _checkpoint_paths(args.output_dir)
    for p in checkpoint_paths.values():
        p.unlink(missing_ok=True)

    LOGGER.info("Saved probes to %s", args.output_dir / "probes.npz")


if __name__ == "__main__":
    main()
