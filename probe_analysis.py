import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer

from dataset import load_balanced_prompt_dataset, validate_labels
from generate import generate_with_layer_probes
from visualization import save_best_layer_projection, save_layer_score_plot


LOGGER = logging.getLogger("probe_analysis")
DEFAULT_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise LLaDA safe/unsafe probing analysis.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model id or local model path.")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Number of safe and unsafe prompts to load.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size for generation/probing.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("probe_outputs"), help="Directory for output artifacts.")
    parser.add_argument("--steps", type=int, default=64, help="Total denoising steps.")
    parser.add_argument("--gen-length", type=int, default=64, help="Generated mask-token length.")
    parser.add_argument("--block-length", type=int, default=64, help="LLaDA denoising block length.")
    parser.add_argument("--probe-step", type=int, default=10, help="One-based denoising step to capture.")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument("--projection", default="pca", choices=["pca", "tsne"], help="Best-layer visualization method.")
    parser.add_argument("--unsafe-split", default="train", help="Unsafe dataset split.")
    parser.add_argument("--safe-split", default="train", help="Safe dataset split.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def model_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def format_prompts_for_instruct_model(tokenizer, prompts: Sequence[str]) -> List[str]:
    """Apply the model chat template when available."""
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
        return formatted
    return list(prompts)


def iter_batches(prompts: Sequence[str], labels: Sequence[int], batch_size: int):
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        yield start, prompts[start:end], labels[start:end]


def extract_layer_features(
    model,
    tokenizer,
    prompts: Sequence[str],
    labels: Sequence[int],
    batch_size: int,
    device: torch.device,
    max_prompt_length: int,
    steps: int,
    gen_length: int,
    block_length: int,
    probe_step: int,
) -> Dict[int, np.ndarray]:
    """Run LLaDA once per batch and collect pooled layer embeddings."""
    layer_buffers = None
    layer_ids = list(range(getattr(model.config, "num_hidden_layers", 32)))

    for batch_index, (start, batch_prompts, _) in enumerate(iter_batches(prompts, labels, batch_size), start=1):
        LOGGER.info("Extracting probes for batch %d (%d-%d/%d).", batch_index, start + 1, start + len(batch_prompts), len(prompts))
        formatted_prompts = format_prompts_for_instruct_model(tokenizer, batch_prompts)
        encoded = tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        _, probes = generate_with_layer_probes(
            model,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            probe_step=probe_step,
            layer_ids=layer_ids,
        )

        if layer_buffers is None:
            layer_buffers = {layer_id: [] for layer_id in layer_ids}

        for layer_id in layer_ids:
            layer_buffers[layer_id].append(probes[f"layer_{layer_id}"].numpy())

        del input_ids, attention_mask, encoded, probes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {layer_id: np.concatenate(chunks, axis=0) for layer_id, chunks in layer_buffers.items()}


def compute_layer_scores(features_by_layer: Dict[int, np.ndarray], labels: Sequence[int]) -> Dict[int, float]:
    """Compute silhouette scores for every layer."""
    labels_array = np.asarray(labels)
    scores = {}
    for layer_id, features in features_by_layer.items():
        LOGGER.info("Computing silhouette score for layer %d.", layer_id)
        scores[layer_id] = float(silhouette_score(features, labels_array))
    return scores


def save_scores(scores: Dict[int, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({str(layer): score for layer, score in sorted(scores.items())}, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    if not 50 <= args.samples_per_class <= 100:
        LOGGER.warning("The requested sample count is outside the recommended 50-100 range.")

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    prompts, labels = load_balanced_prompt_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        safe_split=args.safe_split,
        unsafe_split=args.unsafe_split,
    )
    validate_labels(labels)

    LOGGER.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == 126336:
        raise ValueError("Tokenizer pad_token_id equals LLaDA mask_id; probing would need a distinct pad token.")

    LOGGER.info("Loading model: %s", args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype(device),
    ).to(device).eval()

    features_by_layer = extract_layer_features(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        labels=labels,
        batch_size=args.batch_size,
        device=device,
        max_prompt_length=args.max_prompt_length,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        probe_step=args.probe_step,
    )

    layer_scores = compute_layer_scores(features_by_layer, labels)
    best_layer = max(layer_scores, key=layer_scores.get)
    LOGGER.info("Best layer: %d with silhouette score %.6f.", best_layer, layer_scores[best_layer])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_scores(layer_scores, args.output_dir / "layer_separability_scores.json")
    save_layer_score_plot(layer_scores, args.output_dir / "layer_score_plot.png")

    projection_path = args.output_dir / f"best_layer_{args.projection}.png"
    save_best_layer_projection(features_by_layer[best_layer], labels, projection_path, method=args.projection, seed=args.seed)

    LOGGER.info("Saved outputs to %s.", args.output_dir.resolve())


if __name__ == "__main__":
    main()
