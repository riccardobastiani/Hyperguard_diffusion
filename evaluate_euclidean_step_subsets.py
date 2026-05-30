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


DEFAULT_SUBSETS = {
    "early": [1, 5, 10],
    "critical": [5, 10, 15],
    "late": [20, 25, 30],
    "best_pair": [5, 30],
    "all": [1, 5, 10, 15, 20, 25, 30],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Euclidean multi-step OR over step subsets.")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.95, 0.98])
    parser.add_argument("--output-root", type=Path, default=Path("probe_outputs"))
    parser.add_argument("--best-layers-file", type=Path, default=Path("best_layers.json"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("logs/exp_03_step_subset_or_ablation"))
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
    detector = load_euclidean_checkpoint(probe_dir / f"euclidean_step_{step}.pt", device=device)
    key = f"step_{step}"
    safe_arr = np.load(probe_dir / "safe_probes.npz")[key]
    unsafe_arr = np.load(probe_dir / "unsafe_probes.npz")[key]
    with torch.no_grad():
        safe_dist = detector.predict(torch.from_numpy(safe_arr).float().to(device)).cpu().numpy()
        unsafe_dist = detector.predict(torch.from_numpy(unsafe_arr).float().to(device)).cpu().numpy()
    return safe_dist, unsafe_dist


def _metrics(safe_blocked: np.ndarray, unsafe_blocked: np.ndarray, safe_score: np.ndarray, unsafe_score: np.ndarray):
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
        "subset", "quantile", "steps", "num_steps", "safe_blocked", "safe_total", "safe_block_rate",
        "unsafe_blocked", "unsafe_total", "unsafe_block_rate", "ASR_post",
        "TP", "FP", "TN", "FN", "AUROC_margin", "AUPR_margin",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: List[dict], out_dir: Path) -> None:
    subsets = list(DEFAULT_SUBSETS.keys())
    quantiles = sorted({r["quantile"] for r in rows})
    width = 0.8 / len(quantiles)
    x = np.arange(len(subsets))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for i, q in enumerate(quantiles):
        q_rows = {r["subset"]: r for r in rows if r["quantile"] == q}
        offset = (i - (len(quantiles) - 1) / 2) * width
        axes[0].bar(x + offset, [q_rows[s]["unsafe_block_rate"] for s in subsets], width=width, label=f"q={q:.2f}")
        axes[1].bar(x + offset, [q_rows[s]["safe_block_rate"] for s in subsets], width=width, label=f"q={q:.2f}")

    axes[0].set_title("Step-subset OR: Unsafe Block Rate")
    axes[0].set_ylabel("Unsafe blocked (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="best")
    axes[1].set_title("Step-subset OR: Safe Block Rate")
    axes[1].set_ylabel("Safe blocked (%)")
    axes[1].set_ylim(0, 30)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subsets)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "step_subset_or_rates.png", dpi=180)
    plt.close(fig)


def _write_analysis(rows: List[dict], out: Path) -> None:
    best_attack = max(rows, key=lambda r: (r["unsafe_block_rate"], -r["safe_block_rate"], -r["num_steps"]))
    low_fpr = [r for r in rows if r["safe_block_rate"] <= 5.0]
    best_low_fpr = max(low_fpr, key=lambda r: (r["unsafe_block_rate"], -r["num_steps"])) if low_fpr else None
    efficient = sorted(rows, key=lambda r: (-r["unsafe_block_rate"], r["safe_block_rate"], r["num_steps"]))[0]
    lines = [
        "# exp_03: Step-subset OR Ablation",
        "",
        "## Setup",
        "",
        "This experiment evaluates multi-step OR guards over predefined subsets of denoising steps. The goal is to see whether a smaller trajectory subset gives a better unsafe-blocking / false-positive tradeoff than using all steps.",
        "",
        "Subsets:",
        "",
    ]
    for name, steps in DEFAULT_SUBSETS.items():
        lines.append(f"- `{name}`: {' '.join(map(str, steps))}")
    lines.extend([
        "",
        "## Key Findings",
        "",
        f"Highest unsafe blocking: subset={best_attack['subset']}, q={best_attack['quantile']:.2f}, unsafe blocked={best_attack['unsafe_block_rate']:.1f}%, safe blocked={best_attack['safe_block_rate']:.1f}%.",
    ])
    if best_low_fpr is not None:
        lines.append(f"Best low-FPR setting (safe block <= 5%): subset={best_low_fpr['subset']}, q={best_low_fpr['quantile']:.2f}, unsafe blocked={best_low_fpr['unsafe_block_rate']:.1f}%, safe blocked={best_low_fpr['safe_block_rate']:.1f}%.")
    lines.append(f"Most efficient top setting by unsafe/FPR/step count ordering: subset={efficient['subset']}, q={efficient['quantile']:.2f}, unsafe blocked={efficient['unsafe_block_rate']:.1f}%, safe blocked={efficient['safe_block_rate']:.1f}%, steps={efficient['num_steps']}.")
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `step_subset_or_summary.csv`",
        "- `step_subset_or_rates.png`",
    ])
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    best_layers = _load_best_layers(args.best_layers_file)

    all_steps = sorted({step for steps in DEFAULT_SUBSETS.values() for step in steps})
    distances = {}
    for step in all_steps:
        layer = best_layers[step]
        distances[step] = _load_step_distances(step, layer, args.output_root, device)

    rows = []
    for subset_name, steps in DEFAULT_SUBSETS.items():
        safe_matrix = np.stack([distances[s][0] for s in steps], axis=1)
        unsafe_matrix = np.stack([distances[s][1] for s in steps], axis=1)
        for q in args.quantiles:
            radii = np.quantile(safe_matrix, q, axis=0)
            safe_margin = safe_matrix / radii.reshape(1, -1) - 1.0
            unsafe_margin = unsafe_matrix / radii.reshape(1, -1) - 1.0
            safe_blocked = (safe_margin > 0.0).any(axis=1)
            unsafe_blocked = (unsafe_margin > 0.0).any(axis=1)
            row = _metrics(safe_blocked, unsafe_blocked, safe_margin.max(axis=1), unsafe_margin.max(axis=1))
            row.update({
                "subset": subset_name,
                "quantile": q,
                "steps": " ".join(map(str, steps)),
                "num_steps": len(steps),
            })
            rows.append(row)

    rows.sort(key=lambda r: (r["quantile"], r["subset"]))
    _write_csv(rows, args.experiment_dir / "step_subset_or_summary.csv")
    _plot(rows, args.experiment_dir)
    _write_analysis(rows, args.experiment_dir / "ANALYSIS.md")

    print("Step-subset OR Euclidean results")
    for row in rows:
        print(
            f"subset={row['subset']} | q={row['quantile']:.2f} | steps={row['steps']} | "
            f"unsafe={row['unsafe_blocked']}/{row['unsafe_total']} ({row['unsafe_block_rate']:.1f}%) | "
            f"safe={row['safe_blocked']}/{row['safe_total']} ({row['safe_block_rate']:.1f}%) | "
            f"ASR post={row['ASR_post']:.1f}%"
        )
    print(f"Wrote artifacts to {args.experiment_dir}")


if __name__ == "__main__":
    main()
