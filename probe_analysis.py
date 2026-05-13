import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from dataset import load_balanced_prompt_dataset, validate_labels
from generate import ensure_llada_config_compat, generate_with_layer_probes, install_transformers_tied_weights_compat
from hyperbolic_projection import HyperbolicProjection
from svdd import HyperbolicSVDD, init_center, save_checkpoint, train_svdd


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
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[10], metavar="STEP", help="One-based denoising steps to capture (space-separated, e.g. --probe-steps 1 10 32 64).")
    parser.add_argument("--layer-id", type=int, default=15, help="Transformer layer index to extract hooks from.")
    parser.add_argument("--proj-dim", type=int, default=128, help="Output dimension of the linear projection before hyperbolic mapping.")
    parser.add_argument("--curvature", type=float, default=1.0, help="Curvature k of the Lorentz manifold.")
    parser.add_argument("--nu", type=float, default=0.1, help="SVDD soft-boundary parameter: fraction of training samples allowed outside the sphere.")
    parser.add_argument("--svdd-epochs", type=int, default=50, help="Number of SVDD training epochs.")
    parser.add_argument("--svdd-lr", type=float, default=1e-3, help="Learning rate for SVDD training.")
    parser.add_argument("--svdd-batch-size", type=int, default=32, help="Mini-batch size for SVDD training.")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument("--safe-local-path", type=Path, help="Optional local JSON/CSV file for safe prompts.")
    parser.add_argument("--safe-local-field", default="Refined_behavior", help="Field to read from --safe-local-path.")
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


def extract_layer_features(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int,
    device: torch.device,
    max_prompt_length: int,
    steps: int,
    gen_length: int,
    block_length: int,
    probe_steps: List[int],
    layer_id: int,
) -> Dict[int, np.ndarray]:
    """Run LLaDA and collect pooled hidden states for a single layer at each probe step."""
    step_buffers: Dict[int, list] = {s: [] for s in probe_steps}

    for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch_prompts = prompts[start:start + batch_size]
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
            probe_steps=probe_steps,
            layer_ids=[layer_id],
        )

        for step in probe_steps:
            step_buffers[step].append(probes[step][f"layer_{layer_id}"].numpy())

        del input_ids, attention_mask, encoded, probes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {step: np.concatenate(chunks, axis=0) for step, chunks in step_buffers.items()}


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
        safe_local_path=str(args.safe_local_path) if args.safe_local_path else None,
        safe_local_field=args.safe_local_field,
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
    install_transformers_tied_weights_compat()
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype(device),
    ).to(device).eval()
    ensure_llada_config_compat(model)

    features_by_step = extract_layer_features(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        device=device,
        max_prompt_length=args.max_prompt_length,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        probe_steps=args.probe_steps,
        layer_id=args.layer_id,
    )
    LOGGER.info(
        "Extracted hooks for layer %d at steps %s — shapes: %s",
        args.layer_id,
        args.probe_steps,
        {s: arr.shape for s, arr in features_by_step.items()},
    )

    labels_array = np.asarray(labels)
    safe_mask = labels_array == 0
    unsafe_mask = labels_array == 1
    safe_features: Dict[int, np.ndarray] = {s: arr[safe_mask] for s, arr in features_by_step.items()}
    unsafe_features: Dict[int, np.ndarray] = {s: arr[unsafe_mask] for s, arr in features_by_step.items()}
    LOGGER.info(
        "Safe samples: %d, unsafe samples: %d.",
        safe_mask.sum(),
        unsafe_mask.sum(),
    )

    in_dim = getattr(model.config, "hidden_size", 4096)
    projector = HyperbolicProjection(
        in_dim=in_dim,
        proj_dim=args.proj_dim,
        curvature=args.curvature,
    ).to(device)
    LOGGER.info(
        "HyperbolicProjection: Linear(%d \u2192 %d) + Lorentz expmap0 (k=%.2f). Output dim: %d.",
        in_dim, args.proj_dim, args.curvature, args.proj_dim + 1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step in args.probe_steps:
        LOGGER.info("--- Training SVDD for denoising step %d ---", step)

        safe_tensor = torch.from_numpy(safe_features[step]).float().to(device)

        # 1. Initialize center from safe embeddings
        center = init_center(projector, safe_tensor)

        # 2. Build SVDD
        svdd = HyperbolicSVDD(
            projector=projector,
            center=center,
            nu=args.nu,
        )

        # 3. Train
        train_svdd(
            svdd=svdd,
            safe_features=safe_tensor,
            epochs=args.svdd_epochs,
            lr=args.svdd_lr,
            batch_size=args.svdd_batch_size,
            device=device,
        )

        # 4. Checkpoint
        ckpt_path = args.output_dir / f"svdd_step_{step}.pt"
        save_checkpoint(svdd, ckpt_path)

    LOGGER.info("All checkpoints saved to %s.", args.output_dir.resolve())


if __name__ == "__main__":
    main()
