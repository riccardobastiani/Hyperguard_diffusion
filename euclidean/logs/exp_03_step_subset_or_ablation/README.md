# exp_03: Step-subset OR Ablation

## Goal

Evaluate whether smaller Euclidean OR subsets can recover most of the multi-step benefit with fewer false positives.

Subsets:

- `early`: 1 5 10
- `critical`: 5 10 15
- `late`: 20 25 30
- `best_pair`: 5 30
- `all`: 1 5 10 15 20 25 30

## Run

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion
source .venv312/bin/activate

PYTHON=.venv312/bin/python \
DEVICE=cpu \
QUANTILES="0.95 0.98" \
./run_euclidean_step_subsets.sh
```

## Outputs

- `step_subset_or.log`
- `step_subset_or_summary.csv`
- `step_subset_or_rates.png`
- `ANALYSIS.md`
