# exp_02: Multi-step OR Euclidean Baseline

## Goal

Evaluate a trajectory-style Euclidean guard: block a sample if any selected denoising step exceeds its Euclidean radius. This tests whether combining per-step Euclidean detectors improves unsafe blocking compared with independent single-step evaluation.

## Run

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion
source .venv312/bin/activate

PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
DEVICE=cpu \
QUANTILES="0.90 0.95 0.98 0.99" \
./run_euclidean_multistep_or.sh
```

## Outputs

- `multistep_or.log`
- `multistep_or_summary.csv`
- `multistep_or_rates.png`
- `multistep_or_margin_auc.png`
- `ANALYSIS.md`
