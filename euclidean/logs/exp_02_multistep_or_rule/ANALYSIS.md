# exp_02: Multi-step OR Euclidean Baseline

## Setup

This experiment evaluates a trajectory-style Euclidean guard. For each quantile, every selected step gets its own radius calibrated from safe distances. A sample is blocked if any selected step exceeds its step-specific radius.

This assumes cached safe/unsafe probe arrays are sample-aligned across steps because they were generated with the same dataset seed and split procedure.

## Key Findings

Highest unsafe blocking: q=0.90, unsafe blocked=88.0%, safe blocked=18.0%, ASR post-defense=12.0%.
Best low-FPR setting (safe block <= 5%): q=0.98, unsafe blocked=78.0%, safe blocked=4.0%, ASR post-defense=22.0%.

## Artifacts

- `multistep_or_summary.csv`
- `multistep_or_rates.png`
- `multistep_or_margin_auc.png`
