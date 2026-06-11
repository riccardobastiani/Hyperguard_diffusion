import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Euclidean threshold-sweep logs.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("logs/exp_01_threshold_sweep_quantiles"),
        help="Directory containing quantile_*.log files.",
    )
    return parser.parse_args()


def _parse_log(path: Path):
    m = re.search(r"quantile_(\d+p\d+)", path.name)
    quantile = float(m.group(1).replace("p", ".")) if m else float("nan")
    text = path.read_text()
    blocks = re.split(r"(?=Evaluating Euclidean baseline: step )", text)
    rows = []
    for block in blocks:
        if not block.startswith("Evaluating Euclidean baseline: step "):
            continue
        step_layer = re.search(r"step (\d+), layer (\d+)", block)
        radius = re.search(r"(?:Using calibrated|with) R=([0-9.]+)", block)
        safe_block = re.search(r"safe block rate: (\d+)/(\d+) \(([0-9.]+)%\)", block)
        unsafe_block = re.search(r"unsafe block rate: (\d+)/(\d+) \(([0-9.]+)%\)", block)
        auroc = re.search(r"AUROC: ([0-9.]+)", block)
        aupr = re.search(r"AUPR : ([0-9.]+)", block)
        asr_post = re.search(r"ASR post-defense: ([0-9.]+)%", block)
        conf = re.search(r"confusion matrix \(TP/FP/TN/FN\): (\d+)/(\d+)/(\d+)/(\d+)", block)
        if not all([step_layer, radius, safe_block, unsafe_block, auroc, aupr, asr_post, conf]):
            continue
        rows.append({
            "quantile": quantile,
            "step": int(step_layer.group(1)),
            "layer": int(step_layer.group(2)),
            "R": float(radius.group(1)),
            "safe_blocked": int(safe_block.group(1)),
            "safe_total": int(safe_block.group(2)),
            "safe_block_rate": float(safe_block.group(3)),
            "unsafe_blocked": int(unsafe_block.group(1)),
            "unsafe_total": int(unsafe_block.group(2)),
            "unsafe_block_rate": float(unsafe_block.group(3)),
            "AUROC": float(auroc.group(1)),
            "AUPR": float(aupr.group(1)),
            "ASR_post": float(asr_post.group(1)),
            "TP": int(conf.group(1)),
            "FP": int(conf.group(2)),
            "TN": int(conf.group(3)),
            "FN": int(conf.group(4)),
        })
    return rows


def _write_csv(rows, path: Path) -> None:
    fields = [
        "quantile", "step", "layer", "R", "safe_blocked", "safe_total", "safe_block_rate",
        "unsafe_blocked", "unsafe_total", "unsafe_block_rate", "AUROC", "AUPR", "ASR_post",
        "TP", "FP", "TN", "FN",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_by_step(rows, out: Path) -> None:
    quantiles = sorted({r["quantile"] for r in rows})
    steps = sorted({r["step"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for q in quantiles:
        q_rows = sorted([r for r in rows if r["quantile"] == q], key=lambda r: r["step"])
        axes[0].plot([r["step"] for r in q_rows], [r["unsafe_block_rate"] for r in q_rows], marker="o", label=f"q={q:.2f}")
        axes[1].plot([r["step"] for r in q_rows], [r["safe_block_rate"] for r in q_rows], marker="o", label=f"q={q:.2f}")
    axes[0].set_title("Threshold Sweep: Unsafe Block Rate by Step")
    axes[0].set_ylabel("Unsafe blocked (%)")
    axes[0].legend(loc="best")
    axes[1].set_title("Threshold Sweep: Safe Block Rate by Step")
    axes[1].set_xlabel("Denoising step")
    axes[1].set_ylabel("Safe blocked (%)")
    axes[1].set_xticks(steps)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_step5_tradeoff(rows, out: Path) -> None:
    step5 = sorted([r for r in rows if r["step"] == 5], key=lambda r: r["quantile"])
    if not step5:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([r["safe_block_rate"] for r in step5], [r["unsafe_block_rate"] for r in step5], marker="o", linewidth=2.5)
    for r in step5:
        ax.annotate(f"q={r['quantile']:.2f}", (r["safe_block_rate"], r["unsafe_block_rate"]), textcoords="offset points", xytext=(6, 5))
    ax.set_title("Step 5 Threshold Tradeoff")
    ax.set_xlabel("Safe blocked / FPR (%)")
    ax.set_ylabel("Unsafe blocked / TPR (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _write_analysis(rows, out: Path) -> None:
    best = max(rows, key=lambda r: (r["unsafe_block_rate"], -r["safe_block_rate"]))
    low_fpr = [r for r in rows if r["safe_block_rate"] <= 5.0]
    best_low_fpr = max(low_fpr, key=lambda r: r["unsafe_block_rate"]) if low_fpr else None
    lines = [
        "# exp_01: Euclidean Threshold Sweep Analysis",
        "",
        "## Setup",
        "",
        "This experiment recalibrates the Euclidean radius `R` using several safe-distance quantiles while keeping the same Euclidean centroid and distance score.",
        "",
        "## Key Findings",
        "",
        f"Best raw unsafe block rate: q={best['quantile']:.2f}, step={best['step']}, unsafe blocked={best['unsafe_block_rate']:.1f}%, safe blocked={best['safe_block_rate']:.1f}%.",
    ]
    if best_low_fpr is not None:
        lines.append(
            f"Best configuration with safe block rate <= 5%: q={best_low_fpr['quantile']:.2f}, step={best_low_fpr['step']}, unsafe blocked={best_low_fpr['unsafe_block_rate']:.1f}%, safe blocked={best_low_fpr['safe_block_rate']:.1f}%."
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `threshold_sweep_summary.csv`",
        "- `threshold_sweep_by_step.png`",
        "- `threshold_sweep_step5_tradeoff.png`",
    ])
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.experiment_dir.glob("quantile_*.log")):
        rows.extend(_parse_log(path))
    if not rows:
        raise SystemExit(f"No threshold-sweep rows found in {args.experiment_dir}")
    rows.sort(key=lambda r: (r["quantile"], r["step"]))
    _write_csv(rows, args.experiment_dir / "threshold_sweep_summary.csv")
    _plot_by_step(rows, args.experiment_dir / "threshold_sweep_by_step.png")
    _plot_step5_tradeoff(rows, args.experiment_dir / "threshold_sweep_step5_tradeoff.png")
    _write_analysis(rows, args.experiment_dir / "ANALYSIS.md")
    print(f"Wrote threshold sweep summary to {args.experiment_dir}")


if __name__ == "__main__":
    main()
