import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from svdd import fit_euclidean_svdd, save_euclidean_checkpoint


LOGGER = logging.getLogger("euclidean_baseline")
DEFAULT_PROBE_DIR = Path("probe_outputs/layer_23")
DEFAULT_PROBE_STEPS = [5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Euclidean SVDD baselines from cached LLaDA probe activations."
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=DEFAULT_PROBE_DIR,
        help="Directory containing safe_probes.npz and unsafe_probes.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for euclidean_step_*.pt checkpoints (defaults to --probe-dir).",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        nargs="+",
        default=DEFAULT_PROBE_STEPS,
        metavar="STEP",
        help="One-based denoising steps to fit (e.g. --probe-steps 5 10 15).",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.01,
        help="Fraction of safe samples allowed outside the Euclidean sphere.",
    )
    parser.add_argument(
        "--radius-quantile",
        type=float,
        default=None,
        help="Optional explicit safe-distance quantile for R. Overrides 1 - nu.",
    )
    parser.add_argument(
        "--l2-normalize-probes",
        action="store_true",
        help="Apply per-sample L2 normalization to loaded probe features before fitting.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for fitting distances (default: cpu).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_probe_cache(path: Path, probe_steps: Sequence[int]) -> Dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing probe cache: {path}")

    data = np.load(path)
    loaded: Dict[int, np.ndarray] = {}
    for step in probe_steps:
        key = f"step_{step}"
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}. Available keys: {list(data.keys())}")
        arr = data[key]
        if not np.isfinite(arr).all():
            raise ValueError(f"Probe cache {path}:{key} contains NaN/Inf.")
        loaded[step] = arr
    return loaded


def _l2_normalize_features(features: Dict[int, np.ndarray], eps: float = 1e-6) -> Dict[int, np.ndarray]:
    normalized: Dict[int, np.ndarray] = {}
    for step, arr in features.items():
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        normalized[step] = arr / (norms + eps)
    return normalized


def main() -> None:
    args = parse_args()
    configure_logging()

    output_dir = args.output_dir or args.probe_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_path = args.probe_dir / "safe_probes.npz"
    unsafe_path = args.probe_dir / "unsafe_probes.npz"
    safe_features = _load_probe_cache(safe_path, args.probe_steps)

    # Loading unsafe probes is not needed to fit the one-class baseline, but it
    # catches mismatched probe-step caches before the user starts evaluation.
    _load_probe_cache(unsafe_path, args.probe_steps)

    if args.l2_normalize_probes:
        safe_features = _l2_normalize_features(safe_features)

    device = torch.device(args.device)
    for step in args.probe_steps:
        safe_tensor = torch.from_numpy(safe_features[step]).float().to(device)
        detector = fit_euclidean_svdd(
            safe_tensor,
            nu=args.nu,
            radius_quantile=args.radius_quantile,
        )
        checkpoint_path = output_dir / f"euclidean_step_{step}.pt"
        save_euclidean_checkpoint(detector, checkpoint_path)
        LOGGER.info(
            "Saved Euclidean baseline for step %d with R=%.6f to %s.",
            step,
            float(detector.R.detach()),
            checkpoint_path,
        )


if __name__ == "__main__":
    main()
