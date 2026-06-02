"""Run GPO-V on LLaDA-V and save multimodal hidden-state probes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gpo_v.lladav import (
    ActivationRecorder,
    build_prompt,
    initialize_response_span,
    load_lladav_assets,
    model_dtype,
    optimize_image_with_gpo,
    prepare_image,
    validation_generate,
)


LOGGER = logging.getLogger("run_gpo_v_lladav_probe")


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
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[16, 23, 29])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attack-steps", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--generation-steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--skip-attack", action="store_true", help="Smoke-test mode: do not optimize unsafe images.")
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
    )
    device = next(model.parameters()).device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_buffers: dict[str, list] = {}
    labels = []
    metadata = []

    for row in tqdm(rows, desc="GPO-V probes"):
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
                "source": row.get("source"),
                "prompt": row.get("prompt"),
                "image_path": row.get("image_path"),
                "response": response,
                "attack_steps_run": len(attack_history),
                "span": [span[0], span[1]],
            }
        )

    arrays = {key: np.concatenate(values, axis=0) for key, values in feature_buffers.items()}
    arrays["labels"] = np.asarray(labels, dtype=np.int64)
    np.savez(args.output_dir / "probes.npz", **arrays)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved probes to %s", args.output_dir / "probes.npz")


if __name__ == "__main__":
    main()
