import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from svdd import load_euclidean_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Euclidean detectors with a multi-step OR rule.")
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.98, 0.99])
    parser.add_argument("--output-root", type=Path, default=Path("probe_outputs"))
    parser.add_argument("--best-layers-file", type=Path, default=Path("best_layers.json"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("logs/exp_02_multistep_or_rule"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _auc_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = tp / positives
    fpr = fp / negatives
    return _auc_integral(np.concatenate([[0.0], tpr]), np.concatenate([[0.0], fpr]))


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    recall = tp / positives
    precision = tp / (tp + fp)
    return _auc_integral(np.concatenate([[1.0], precision]), np.concatenate([[0.0], recall]))


def _load_best_layers(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): int(v) for k, v in raw.items()}


def _load_step_distances(step: int, layer: int, output_root: Path, device: torch.device):
    probe_dir = output_root / f"step_{step}" / f"layer_{layer}"
    ckpt_path = probe_dir / f"euclidean_step_{step}.pt"
    safe_path = probe_dir / "safe_probes.npz"
    unsafe_path = probe_dir / "unsafe_probes.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Euclidean checkpoint: {ckpt_path}")
    if not safe_path.exists() or not unsafe_path.exists():
        raise FileNotFoundError(f"Missing probe cache in {probe_dir}")

    detector = load_euclidean_checkpoint(ckpt_path, device=device)
    key = f"step_{step}"
    safe_arr = np.load(safe_path)[key]
    unsafe_arr = np.load(unsafe_path)[key]

    safe_feat = torch.from_numpy(safe_arr).float().to(device)
    unsafe_feat = torch.from_numpy(unsafe_arr).float().to(device)
    with torch.no_grad():
        safe_dist = detector.predict(safe_feat).cpu().numpy()
        unsafe_dist = detector.predict(unsafe_feat).cpu().numpy()
    return safe_dist, unsafe_dist


def _metrics_from_blocks(safe_blocked: np.ndarray, unsafe_blocked: np.ndarray, safe_score: np.ndarray, unsafe_score: np.ndarray):
    tp = int(unsafe_blocked.sum())
    fn = int((~unsafe_blocked).sum())
    fp = int(safe_blocked.sum())
    tn = int((~safe_blocked).sum())
    scores = np.concatenate([safe_score, unsafe_score])
    labels = np.concatenate([np.zeros_like(safe_score, dtype=np.int32), np.ones_like(unsafe_score, dtype=np.int32)])
    return {
        "safe_blocked": fp,
        "safe_total": int(safe_blocked.size),
        "safe_block_rate": 100.0 * fp / safe_blocked.size,
        "unsafe_blocked": tp,
        "unsafe_total": int(unsafe_blocked.size),
        "unsafe_block_rate": 100.0 * tp / unsafe_blocked.size,
        "ASR_post": 100.0 * fn / unsafe_blocked.size,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "AUROC_margin": _roc_auc(scores, labels),
        "AUPR_margin": _pr_auc(scores, labels),
    }


def _write_csv(rows: List[dict], path: Path) -> None:
    fields = [
        "quantile", "steps", "num_steps", "safe_blocked", "safe_total", "safe_block_rate",
        "unsafe_blocked", "unsafe_total", "unsafe_block_rate", "ASR_post",
        "TP", "FP", "TN", "FN", "AUROC_margin", "AUPR_margin",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: List[dict], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: r["quantile"])
    q = [r["quantile"] for r in rows]
    unsafe = [r["unsafe_block_rate"] for r in rows]
    safe = [r["safe_block_rate"] for r in rows]
    asr = [r["ASR_post"] for r in rows]
    auroc = [r["AUROC_margin"] for r in rows]
    aupr = [r["AUPR_margin"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(q, unsafe, marker="o", linewidth=2.5, label="Unsafe blocked")
    ax.plot(q, safe, marker="s", linewidth=2.5, label="Safe blocked")
    ax.plot(q, asr, marker="^", linewidth=2.5, label="ASR post-defense")
    ax.set_title("Multi-step OR Euclidean Guard")
    ax.set_xlabel("Safe-distance quantile used as per-step radius")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "multistep_or_rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(q, auroc, marker="o", linewidth=2.5, label="AUROC of max normalized margin")
    ax.plot(q, aupr, marker="s", linewidth=2.5, label="AUPR of max normalized margin")
    ax.set_title("Multi-step OR Margin Ranking")
    ax.set_xlabel("Safe-distance quantile used as per-step radius")
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.0)
    ax.invert_xaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "multistep_or_margin_auc.png", dpi=180)
    plt.close(fig)


def _write_analysis(rows: List[dict], out: Path) -> None:
    best_attack = max(rows, key=lambda r: (r["unsafe_block_rate"], -r["safe_block_rate"]))
    low_fpr = [r for r in rows if r["safe_block_rate"] <= 5.0]
    best_low_fpr = max(low_fpr, key=lambda r: r["unsafe_block_rate"]) if low_fpr else None
    lines = [
        "# exp_02: Multi-step OR Euclidean Baseline",
        "",
        "## Setup",
        "",
        "This experiment evaluates a trajectory-style Euclidean guard. For each quantile, every selected step gets its own radius calibrated from safe distances. A sample is blocked if any selected step exceeds its step-specific radius.",
        "",
        "This assumes cached safe/unsafe probe arrays are sample-aligned across steps because they were generated with the same dataset seed and split procedure.",
        "",
        "## Key Findings",
        "",
        f"Highest unsafe blocking: q={best_attack['quantile']:.2f}, unsafe blocked={best_attack['unsafe_block_rate']:.1f}%, safe blocked={best_attack['safe_block_rate']:.1f}%, ASR post-defense={best_attack['ASR_post']:.1f}%.",
    ]
    if best_low_fpr is not None:
        lines.append(f"Best low-FPR setting (safe block <= 5%): q={best_low_fpr['quantile']:.2f}, unsafe blocked={best_low_fpr['unsafe_block_rate']:.1f}%, safe blocked={best_low_fpr['safe_block_rate']:.1f}%, ASR post-defense={best_low_fpr['ASR_post']:.1f}%.")
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `multistep_or_summary.csv`",
        "- `multistep_or_rates.png`",
        "- `multistep_or_margin_auc.png`",
    ])
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    best_layers = _load_best_layers(args.best_layers_file)

    safe_dists = []
    unsafe_dists = []
    step_info = []
    for step in args.probe_steps:
        if step not in best_layers:
            raise KeyError(f"Missing best layer for step {step} in {args.best_layers_file}")
        layer = best_layers[step]
        safe_dist, unsafe_dist = _load_step_distances(step, layer, args.output_root, device)
        safe_dists.append(safe_dist)
        unsafe_dists.append(unsafe_dist)
        step_info.append((step, layer))

    safe_matrix = np.stack(safe_dists, axis=1)
    unsafe_matrix = np.stack(unsafe_dists, axis=1)
    if safe_matrix.shape[0] != unsafe_matrix.shape[0]:
        raise ValueError("Safe and unsafe sample counts differ; OR metrics still possible but this script expects balanced cached probes.")

    rows = []
    for q in args.quantiles:
        if not 0.0 < q <= 1.0:
            raise ValueError(f"Invalid quantile: {q}")
        radii = np.quantile(safe_matrix, q, axis=0)
        safe_margin = safe_matrix / radii.reshape(1, -1) - 1.0
        unsafe_margin = unsafe_matrix / radii.reshape(1, -1) - 1.0
        safe_blocked = (safe_margin > 0.0).any(axis=1)
        unsafe_blocked = (unsafe_margin > 0.0).any(axis=1)
        safe_score = safe_margin.max(axis=1)
        unsafe_score = unsafe_margin.max(axis=1)
        metrics = _metrics_from_blocks(safe_blocked, unsafe_blocked, safe_score, unsafe_score)
        metrics.update({
            "quantile": q,
            "steps": " ".join(str(s) for s, _ in step_info),
            "num_steps": len(step_info),
        })
        rows.append(metrics)

    rows.sort(key=lambda r: r["quantile"])
    _write_csv(rows, args.experiment_dir / "multistep_or_summary.csv")
    _plot(rows, args.experiment_dir)
    _write_analysis(rows, args.experiment_dir / "ANALYSIS.md")

    print("Multi-step OR Euclidean results")
    print(f"steps: {', '.join(f'{s}/layer{l}' for s, l in step_info)}")
    for row in rows:
        print(
            f"q={row['quantile']:.2f} | unsafe blocked={row['unsafe_blocked']}/{row['unsafe_total']} "
            f"({row['unsafe_block_rate']:.1f}%) | safe blocked={row['safe_blocked']}/{row['safe_total']} "
            f"({row['safe_block_rate']:.1f}%) | ASR post={row['ASR_post']:.1f}% | "
            f"AUROC_margin={row['AUROC_margin']:.4f} | AUPR_margin={row['AUPR_margin']:.4f}"
        )
    print(f"Wrote artifacts to {args.experiment_dir}")


if __name__ == "__main__":
    main()
