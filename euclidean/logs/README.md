# Experiment Logs

This directory is organized by experiment. Each experiment subfolder should contain the raw terminal log, parsed summary files, generated plots, and a short analysis/notes file.

## Index

| Folder | Experiment | Status | Notes |
| --- | --- | --- | --- |
| `exp_00_euclidean_minimal_svdd_nu001_cached_eval/` | Minimal Euclidean SVDD baseline, `NU=0.01`, cached probes, independent per-step evaluation | Complete | Establishes the current baseline result. |
| `exp_01_threshold_sweep_quantiles/` | Euclidean radius threshold sweep over safe-distance quantiles | Complete | Shows threshold calibration improves the Euclidean baseline. |
| `exp_02_multistep_or_rule/` | Multi-step OR Euclidean guard across probe steps | Complete | Shows the trajectory OR tradeoff. |
| `exp_03_step_subset_or_ablation/` | Step-subset OR ablation | Complete | Tests smaller OR subsets to reduce false positives. |
| `exp_04_hf_heldout_eval/` | Held-out HF test evaluation | Complete | Evaluates best Euclidean settings on fresh HF test prompts. |
| `exp_05_mahalanobis_euclidean/` | PCA + Ledoit-Wolf Mahalanobis Euclidean baseline | Ready to run | Stronger covariance-aware Euclidean held-out baseline. |
| `manual_runs/` | Ad hoc logs from direct script runs | Ongoing | Used when no experiment-specific `LOG_FILE` is provided. |
