"""
Evaluate trained SVDD detectors using the already-extracted probe activations
(probe_outputs/layer_23/safe_probes.npz and unsafe_probes.npz).
No model loading or dataset download required.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from svdd import load_checkpoint

PROBE_STEPS = [5, 10, 15]
CKPT_DIR = Path("probe_outputs/layer_23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                        help="Directory containing svdd_step_*.pt and *_probes.npz files.")
    parser.add_argument("--probe-steps", type=int, nargs="+", default=PROBE_STEPS,
                        metavar="STEP")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which sample to pick from safe/unsafe arrays (0-based).")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    safe_path   = args.ckpt_dir / "safe_probes.npz"
    unsafe_path = args.ckpt_dir / "unsafe_probes.npz"

    if not safe_path.exists() or not unsafe_path.exists():
        raise FileNotFoundError(
            f"Probe cache not found in {args.ckpt_dir}. "
            "Run probe_analysis.py first to extract and save activations."
        )

    safe_data   = np.load(safe_path)
    unsafe_data = np.load(unsafe_path)

    print(f"Loaded probe cache from {args.ckpt_dir}")
    print(f"  safe  keys: {list(safe_data.keys())}")
    print(f"  unsafe keys: {list(unsafe_data.keys())}\n")

    overall_correct = 0
    overall_total   = 0

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

        idx = min(args.sample_idx, safe_arr.shape[0] - 1, unsafe_arr.shape[0] - 1)

        safe_feat   = torch.from_numpy(safe_arr[idx:idx+1]).float().to(device)
        unsafe_feat = torch.from_numpy(unsafe_arr[idx:idx+1]).float().to(device)

        R           = svdd.R.item()
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

    if overall_total > 0:
        acc = 100 * overall_correct / overall_total
        print(f"Accuracy across all tested steps: {overall_correct}/{overall_total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()