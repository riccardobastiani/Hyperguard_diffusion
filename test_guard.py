"""
Evaluate trained SVDD detectors using either:
  - fresh activations extracted on-the-fly from the HuggingFace test set at
    saralazza/llada-safety-dataset (default), or
  - pre-extracted probe activations (probe_outputs/layer_23/safe_probes.npz /
    unsafe_probes.npz).

Pass --cached-probes to use cached probe files instead. All other flags are
compatible with both modes.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers

from svdd import load_checkpoint, save_checkpoint

LOGGER = logging.getLogger("test_guard")

PROBE_STEPS = [5]
CKPT_DIR = Path("probe_outputs/layer_23")
DEFAULT_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SAFETY_DATASET = "saralazza/llada-safety-dataset"
SAFE_SOURCE = "alpaca"


def _rate(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                        help="Directory containing svdd_step_*.pt and *_probes.npz files.")
    parser.add_argument("--probe-steps", type=int, nargs="+", default=PROBE_STEPS,
                        metavar="STEP")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which sample to pick from safe/unsafe arrays (0-based).")
    parser.add_argument(
        "--eval-all-unsafe",
        action="store_true",
        default=True,
        help="Evaluate all unsafe samples and report block rate and distance stats.",
    )
    parser.add_argument(
        "--eval-all-safe",
        action="store_true",
        default=True,
        help="Evaluate all safe samples and report block rate and distance stats.",
    )
    parser.add_argument(
        "--single-sample",
        action="store_true",
        help="Evaluate only --sample-idx from each class instead of all samples.",
    )
    parser.add_argument(
        "--calibrate-quantile",
        type=float,
        default=None,
        help="Set R to this quantile of safe distances (e.g., 0.95).",
    )
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Overwrite svdd_step_*.pt with calibrated R (default saves new files).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (default: 'cuda' if available, else 'cpu').")

    # ------------------------------------------------------------------ #
    # HuggingFace live-inference mode                                      #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--hf-test",
        action="store_true",
        default=True,
        help=(
            "Download test samples from saralazza/llada-safety-dataset, run "
            "the LLaDA model and evaluate on fresh probe features instead of "
            "the cached .npz files."
        ),
    )
    parser.add_argument(
        "--cached-probes",
        action="store_false",
        dest="hf_test",
        help="Use cached safe_probes.npz / unsafe_probes.npz instead of HuggingFace test data.",
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="Dataset split to use in HF mode (default: 'test').",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Number of safe and unsafe samples to draw in HF mode (default: all available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for HF-mode dataset shuffling.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id used for probe extraction in HF mode.",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=23,
        help="Transformer layer index to extract hooks from (must match checkpoint).",
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompt batch size for LLaDA inference in HF mode.")
    parser.add_argument("--steps", type=int, default=64,
                        help="Total denoising steps for LLaDA in HF mode.")
    parser.add_argument("--gen-length", type=int, default=64,
                        help="Generated mask-token length in HF mode.")
    parser.add_argument("--block-length", type=int, default=64,
                        help="LLaDA denoising block length in HF mode.")
    parser.add_argument("--max-prompt-length", type=int, default=512,
                        help="Tokenizer truncation length in HF mode.")
    return parser.parse_args()


def _resolve_split(dataset_dict, preferred: str):
    """Return the preferred split or the first available one."""
    if preferred in dataset_dict:
        return dataset_dict[preferred]
    fallback = next(iter(dataset_dict.keys()))
    LOGGER.warning("Split '%s' not found; falling back to '%s'.", preferred, fallback)
    return dataset_dict[fallback]


def _collect(dataset, text_fn, n: Optional[int]) -> List[str]:
    prompts: List[str] = []
    for row in dataset:
        text = (text_fn(row) or "").strip()
        if text:
            prompts.append(text)
        if n is not None and len(prompts) == n:
            break
    if n is not None and len(prompts) < n:
        LOGGER.warning("Only %d/%d prompts available.", len(prompts), n)
    return prompts


def load_hf_test_prompts(
    split: str,
    samples_per_class: Optional[int],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Return (safe_prompts, unsafe_prompts) from the HuggingFace test set."""
    from datasets import load_dataset as hf_load

    LOGGER.info("Downloading dataset %s …", SAFETY_DATASET)
    ds_dict = hf_load(SAFETY_DATASET)
    ds = _resolve_split(ds_dict, split).shuffle(seed=seed)

    safe_ds = ds.filter(
        lambda r: (r.get("source_dataset") or "").strip().lower() == SAFE_SOURCE
    )
    unsafe_ds = ds.filter(
        lambda r: (r.get("source_dataset") or "").strip().lower() != SAFE_SOURCE
    )

    def _safe_text(row) -> str:
        for field in ("refined_behavior", "behavior", "instruction"):
            v = (row.get(field) or "").strip()
            if v:
                return v
        inp = (row.get("input") or "").strip()
        return inp

    def _unsafe_text(row) -> str:
        for field in ("refined_behavior", "prompt", "behavior", "instruction"):
            v = (row.get(field) or "").strip()
            if v:
                return v
        return (row.get("input") or "").strip()

    safe_prompts   = _collect(safe_ds,   _safe_text,   samples_per_class)
    unsafe_prompts = _collect(unsafe_ds, _unsafe_text, samples_per_class)

    LOGGER.info("Collected %d safe / %d unsafe prompts from split '%s'.",
                len(safe_prompts), len(unsafe_prompts), split)
    return safe_prompts, unsafe_prompts


def model_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32
def apply_transformers_attribute_patch():
    """
    Fixes AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
    by patching the transformers utility function before it's called.
    """
    from transformers.modeling_utils import get_total_byte_count
    
    # Check if already patched
    if hasattr(transformers.modeling_utils, "_is_patched_for_llada"):
        return

    original_get_total_byte_count = transformers.modeling_utils.get_total_byte_count

    def patched_get_total_byte_count(model, accelerator_device_map, hf_quantizer):
        if not hasattr(model, "all_tied_weights_keys"):
            # LLaDA doesn't implement this, so we provide an empty fallback
            # so the library doesn't crash.
            model.all_tied_weights_keys = {}
        return original_get_total_byte_count(model, accelerator_device_map, hf_quantizer)

    transformers.modeling_utils.get_total_byte_count = patched_get_total_byte_count
    transformers.modeling_utils._is_patched_for_llada = True
    LOGGER.info("Applied transformers tied_weights attribute patch.")

def load_llada_assets(model_name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer
    from generate import ensure_llada_config_compat, install_transformers_tied_weights_compat

    # 1. Apply both patches
    install_transformers_tied_weights_compat()
    apply_transformers_attribute_patch()

    LOGGER.info("Loading tokenizer from %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == 126336:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model from %s …", model_name)
    
    # 2. Use device_map to avoid RAM spike, but keep it simple
    # to avoid complex sharding logic that might trigger more bugs
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype(device),
        "low_cpu_mem_usage": True,
    }
    
    if device.type == "cuda":
        # Using a fixed map instead of "auto" often bypasses 
        # some of the heavy 'warmup' logic that causes the crash.
        load_kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    
    # 3. Final model setup
    model.eval()
    ensure_llada_config_compat(model)
    return model, tokenizer


def extract_features_from_prompts(
    prompts: Sequence[str],
    model,
    tokenizer,
    layer_id: int,
    probe_steps: List[int],
    batch_size: int,
    steps: int,
    gen_length: int,
    block_length: int,
    max_prompt_length: int,
    device: torch.device,
    l2_normalize: bool = True,  # Added this parameter
) -> Dict[int, np.ndarray]:
    """Return mean-pooled hidden states per probe step, optionally L2 normalized."""
    from generate import generate_with_layer_probes

    # Format with chat template when available
    formatted: List[str] = []
    if hasattr(tokenizer, "apply_chat_template"):
        for p in prompts:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
    else:
        formatted = list(prompts)

    step_buffers: Dict[int, list] = {s: [] for s in probe_steps}

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        LOGGER.info("Extracting probes batch %d-%d / %d …",
                    i + 1, i + len(batch), len(formatted))
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
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

        for s in probe_steps:
            # Move to CPU and convert to numpy immediately to save GPU RAM
            step_buffers[s].append(probes[s][f"layer_{layer_id}"].cpu().numpy())

        del input_ids, attention_mask, encoded, probes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- NEW NORMALIZATION LOGIC ---
    results = {}
    for s in probe_steps:
        # 1. Concatenate all batches for this step
        arr = np.concatenate(step_buffers[s], axis=0)
        
        # 2. Apply L2 normalization if requested
        if l2_normalize:
            # Calculate Euclidean norm per sample (axis=1)
            # Add a small epsilon (1e-6) to prevent division by zero
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norms + 1e-6)
            
        results[s] = arr

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.single_sample:
        args.eval_all_safe = False
        args.eval_all_unsafe = False
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Acquire probe features — either from .npz cache or live HF inference
    # ------------------------------------------------------------------
    if args.hf_test:
        safe_prompts, unsafe_prompts = load_hf_test_prompts(
            split=args.hf_split,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )

        model, tokenizer = load_llada_assets(args.model_name, device)
        common_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            layer_id=args.layer_id,
            probe_steps=args.probe_steps,
            batch_size=args.batch_size,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            max_prompt_length=args.max_prompt_length,
            device=device,
        )

        LOGGER.info("Extracting probe features for %d safe prompts …", len(safe_prompts))
        safe_features = extract_features_from_prompts(safe_prompts, **common_kwargs)

        LOGGER.info("Extracting probe features for %d unsafe prompts …", len(unsafe_prompts))
        unsafe_features = extract_features_from_prompts(unsafe_prompts, **common_kwargs)

        # Wrap as {key: array} dicts keyed by "step_N" to reuse the loop below
        safe_data   = {f"step_{s}": safe_features[s]   for s in args.probe_steps}
        unsafe_data = {f"step_{s}": unsafe_features[s] for s in args.probe_steps}

        print(f"\nUsing HuggingFace test data from '{SAFETY_DATASET}' (split='{args.hf_split}')")
        print(f"  safe samples : {len(safe_prompts)}")
        print(f"  unsafe samples: {len(unsafe_prompts)}\n")

    else:
        safe_path   = args.ckpt_dir / "safe_probes.npz"
        unsafe_path = args.ckpt_dir / "unsafe_probes.npz"

        if not safe_path.exists() or not unsafe_path.exists():
            raise FileNotFoundError(
                f"Probe cache not found in {args.ckpt_dir}. "
                "Run probe_analysis.py first to extract and save activations, "
                "or pass --hf-test to run live inference."
            )

        safe_data   = np.load(safe_path)
        unsafe_data = np.load(unsafe_path)

        print(f"Loaded probe cache from {args.ckpt_dir}")
        print(f"  safe  keys: {list(safe_data.keys())}")
        print(f"  unsafe keys: {list(unsafe_data.keys())}\n")

    overall_correct = 0
    overall_total   = 0
    unsafe_blocked_total = 0  # TP: unsafe prompt correctly stopped
    unsafe_allowed_total = 0  # FN: unsafe prompt allowed to pass
    unsafe_total = 0
    safe_blocked_total = 0    # FP: safe prompt incorrectly stopped
    safe_allowed_total = 0    # TN: safe prompt correctly allowed
    safe_total = 0

    for step in args.probe_steps:
        ckpt = args.ckpt_dir / f"svdd_step_{step}.pt"
        if not ckpt.exists():
            print(f"[step {step}] checkpoint not found at {ckpt}, skipping.")
            continue

        svdd = load_checkpoint(ckpt, device=device)
        key  = f"step_{step}"

        if key not in safe_data or key not in unsafe_data:
            print(f"[step {step}] key '{key}' missing in probe cache, skipping.")
            continue

        safe_arr   = safe_data[key]
        unsafe_arr = unsafe_data[key]

        calibrated = False
        if args.calibrate_quantile is not None:
            if not 0.0 < args.calibrate_quantile < 1.0:
                raise ValueError("--calibrate-quantile must be between 0 and 1.")
            safe_feat = torch.from_numpy(safe_arr).float().to(device)
            safe_dist = svdd.predict(safe_feat).cpu().numpy()
            new_R = float(np.quantile(safe_dist, args.calibrate_quantile))
            svdd.set_radius(new_R)
            out_path = ckpt if args.overwrite_checkpoints else ckpt.with_name(f"svdd_step_{step}_calib.pt")
            save_checkpoint(svdd, out_path)
            print(f"[step {step}] Calibrated R to {new_R:.6f} (quantile={args.calibrate_quantile}) -> {out_path}")
            calibrated = True

        R = svdd.R.item()
        if not calibrated:
            print(f"[step {step}] Loaded SVDD checkpoint from {ckpt} with R={R:.4f}")
        else:
            print(f"[step {step}] Using calibrated R={R:.4f} for evaluation")

        if args.eval_all_unsafe or args.eval_all_safe:
            safe_feat = torch.from_numpy(safe_arr).float().to(device)
            safe_dist = svdd.predict(safe_feat).cpu().numpy()
            safe_blocked = safe_dist > R
            fp_count = int(safe_blocked.sum())
            tn_count = int((~safe_blocked).sum())
            safe_count = safe_blocked.size

            unsafe_feat = torch.from_numpy(unsafe_arr).float().to(device)
            unsafe_dist = svdd.predict(unsafe_feat).cpu().numpy()
            unsafe_blocked = unsafe_dist > R
            tp_count = int(unsafe_blocked.sum())
            fn_count = int((~unsafe_blocked).sum())
            unsafe_count = unsafe_blocked.size

            safe_blocked_total += fp_count
            safe_allowed_total += tn_count
            safe_total += safe_count
            unsafe_blocked_total += tp_count
            unsafe_allowed_total += fn_count
            unsafe_total += unsafe_count

            print(f"=== Step {step} | R={R:.4f} ===")
            print(f"  TP unsafe blocked: {tp_count}/{unsafe_count} ({_rate(tp_count, unsafe_count):.1f}%)")
            print(f"  FN unsafe allowed: {fn_count}/{unsafe_count} ({_rate(fn_count, unsafe_count):.1f}%)")
            print(f"  TN safe allowed:   {tn_count}/{safe_count} ({_rate(tn_count, safe_count):.1f}%)")
            print(f"  FP safe blocked:   {fp_count}/{safe_count} ({_rate(fp_count, safe_count):.1f}%)")
            print(
                f"  rates: TPR={_rate(tp_count, unsafe_count):.1f}% "
                f"FNR={_rate(fn_count, unsafe_count):.1f}% "
                f"TNR={_rate(tn_count, safe_count):.1f}% "
                f"FPR={_rate(fp_count, safe_count):.1f}%"
            )
            if args.eval_all_safe:
                print(
                    f"  safe dist stats: min={safe_dist.min():.4f} "
                    f"mean={safe_dist.mean():.4f} max={safe_dist.max():.4f}"
                )
            if args.eval_all_unsafe:
                print(
                    f"  unsafe dist stats: min={unsafe_dist.min():.4f} "
                    f"mean={unsafe_dist.mean():.4f} max={unsafe_dist.max():.4f}"
                )
            print()
            continue

        idx = min(args.sample_idx, safe_arr.shape[0] - 1, unsafe_arr.shape[0] - 1)
        safe_feat   = torch.from_numpy(safe_arr[idx:idx+1]).float().to(device)
        unsafe_feat = torch.from_numpy(unsafe_arr[idx:idx+1]).float().to(device)

        safe_dist   = svdd.predict(safe_feat).item()
        unsafe_dist = svdd.predict(unsafe_feat).item()
        safe_blocked   = safe_dist   > R
        unsafe_blocked = unsafe_dist > R

        safe_result   = "STOP (FP!)"       if safe_blocked   else "CONTINUE (correct)"
        unsafe_result = "STOP (correct)"   if unsafe_blocked else "CONTINUE (FN!)"

        print(f"=== Step {step} | R={R:.4f} ===")
        print(f"  safe   sample: dist={safe_dist:.4f}   → {safe_result}")
        print(f"  unsafe sample: dist={unsafe_dist:.4f}   → {unsafe_result}")
        print()

        overall_correct += int(not safe_blocked) + int(unsafe_blocked)
        overall_total   += 2

    if (args.eval_all_safe or args.eval_all_unsafe) and safe_total > 0 and unsafe_total > 0:
        print("Confusion matrix across steps:")
        print(f"  TP unsafe blocked: {unsafe_blocked_total}/{unsafe_total} ({_rate(unsafe_blocked_total, unsafe_total):.1f}%)")
        print(f"  FN unsafe allowed: {unsafe_allowed_total}/{unsafe_total} ({_rate(unsafe_allowed_total, unsafe_total):.1f}%)")
        print(f"  TN safe allowed:   {safe_allowed_total}/{safe_total} ({_rate(safe_allowed_total, safe_total):.1f}%)")
        print(f"  FP safe blocked:   {safe_blocked_total}/{safe_total} ({_rate(safe_blocked_total, safe_total):.1f}%)")
        print(
            f"  rates: TPR={_rate(unsafe_blocked_total, unsafe_total):.1f}% "
            f"FNR={_rate(unsafe_allowed_total, unsafe_total):.1f}% "
            f"TNR={_rate(safe_allowed_total, safe_total):.1f}% "
            f"FPR={_rate(safe_blocked_total, safe_total):.1f}%"
        )
    if overall_total > 0:
        acc = 100 * overall_correct / overall_total
        print(f"Accuracy across all tested steps: {overall_correct}/{overall_total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
