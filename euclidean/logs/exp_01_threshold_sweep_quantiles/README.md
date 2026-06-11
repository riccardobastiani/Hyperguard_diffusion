# exp_01: Euclidean Threshold Sweep

## Goal

Evaluate whether the minimal Euclidean baseline is limited by representation quality or by the conservative radius threshold. The sweep changes only the safe-distance quantile used as radius `R`; the underlying center and score remain the same.

## Planned Quantiles

```text
0.90 0.95 0.98 0.99
```

Lower quantiles should block more unsafe prompts but may increase safe false positives.

## Run

From the repository root:

```bash
source .venv312/bin/activate

PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
DEVICE=cpu \
QUANTILES="0.90 0.95 0.98 0.99" \
./run_euclidean_threshold_sweep.sh
```

## Outputs

The runner automatically calls `summarize_threshold_sweep.py` after all quantile logs finish. Expected generated files:

- `threshold_sweep_summary.csv`
- `threshold_sweep_by_step.png`
- `threshold_sweep_step5_tradeoff.png`
- `ANALYSIS.md`
