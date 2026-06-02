"""Train/evaluate Hyperbolic SVDD detectors on multimodal LLaDA-V probes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from hyperbolic_projection import HyperbolicProjection
from svdd import HyperbolicSVDD, init_center, save_checkpoint, train_svdd


LOGGER = logging.getLogger("train_multimodal_svdd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hyperbolic SVDD on multimodal probes.")
    parser.add_argument("--probe-npz", type=Path, default=Path("outputs/multimodal_gpo/probes/probes.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal_gpo/svdd"))
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[16, 23, 29])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unsafe-weight", type=float, default=1.0)
    parser.add_argument("--no-unsafe-contrastive", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def split_indices(labels: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    train = []
    eval_ = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        if idx.size < 2:
            raise ValueError(f"Need at least two samples for label {label}; got {idx.size}.")
        rng.shuffle(idx)
        n_train = max(1, min(idx.size - 1, int(round(idx.size * train_fraction))))
        train.extend(idx[:n_train])
        eval_.extend(idx[n_train:])
    return np.asarray(train), np.asarray(eval_)


def l2_normalize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + eps)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    unsafe_total = int((labels == 1).sum())
    safe_total = int((labels == 0).sum())
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR": float(average_precision_score(labels, scores)),
        "threshold_R": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR": float(tp / max(unsafe_total, 1)),
        "FPR": float(fp / max(safe_total, 1)),
        "ASR_post_defense": float(fn / max(unsafe_total, 1)),
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = np.load(args.probe_npz)
    labels = data["labels"].astype(np.int64)
    train_idx, eval_idx = split_indices(labels, args.train_fraction, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for step in args.probe_steps:
        for layer_id in args.layer_ids:
            key = f"step_{step}_layer_{layer_id}"
            if key not in data:
                LOGGER.warning("Skipping missing probe key: %s", key)
                continue

            features = l2_normalize(data[key].astype(np.float32))
            train_safe = features[np.intersect1d(train_idx, np.flatnonzero(labels == 0))]
            train_unsafe = features[np.intersect1d(train_idx, np.flatnonzero(labels == 1))]
            eval_features = features[eval_idx]
            eval_labels = labels[eval_idx]

            projector = HyperbolicProjection(
                in_dim=features.shape[1],
                proj_dim=args.proj_dim,
                curvature=args.curvature,
            ).to(device)
            safe_tensor = torch.from_numpy(train_safe).float().to(device)
            unsafe_tensor = (
                None
                if args.no_unsafe_contrastive
                else torch.from_numpy(train_unsafe).float().to(device)
            )
            center = init_center(projector, safe_tensor)
            svdd = HyperbolicSVDD(projector=projector, center=center, nu=args.nu)
            train_svdd(
                svdd,
                safe_features=safe_tensor,
                unsafe_features=unsafe_tensor,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                unsafe_weight=args.unsafe_weight,
                device=device,
            )

            eval_tensor = torch.from_numpy(eval_features).float().to(device)
            scores = svdd.predict(eval_tensor).detach().cpu().numpy()
            metrics = compute_metrics(eval_labels, scores, float(svdd.R.detach().cpu()))
            all_metrics[key] = metrics
            save_checkpoint(svdd, args.output_dir / f"svdd_{key}.pt")

            print(f"=== {key} ===")
            print(f"AUROC: {metrics['AUROC']:.4f}")
            print(f"AUPR : {metrics['AUPR']:.4f}")
            print(f"ASR post-defense: {100 * metrics['ASR_post_defense']:.1f}%")
            print(f"FPR: {100 * metrics['FPR']:.1f}% | TPR: {100 * metrics['TPR']:.1f}%")
            print(f"confusion matrix (TP/FP/TN/FN): {metrics['TP']}/{metrics['FP']}/{metrics['TN']}/{metrics['FN']}")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
