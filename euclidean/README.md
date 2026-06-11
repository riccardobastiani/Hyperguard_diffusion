# Euclidean Baseline for LLaDA Safety Trajectory Detection

```text
How much safety signal is available from ordinary Euclidean geometry on LLaDA hidden-state trajectories, before using the proposed hyperbolic projection?
```

The final answer is: Euclidean hidden states contain meaningful safety signal, especially after threshold tuning and trajectory aggregation, but the best Euclidean baseline remains weaker than the hyperbolic detector reported in `info.md`.

## Baseline Definition

The minimal Euclidean baseline is a one-class detector trained only on safe hidden-state probes.

For each denoising step `t`, we use cached mean-pooled hidden states:

```text
x_i,t in R^d
```

The safe centroid is:

```text
c_t = mean({x_i,t safe})
```

The Euclidean anomaly score is:

```text
d_E(x, c_t) = ||x - c_t||_2
```

A radius is calibrated from safe training/cache distances:

```text
R_t = quantile_q({d_E(x_i,t, c_t) for safe train samples})
```

The decision rule is:

```text
block if d_E(x, c_t) > R_t
```

For trajectory variants with multiple steps, each step has its own radius and the OR rule is used:

```text
block if any selected step fires
```

Unsafe samples are not used to fit the Euclidean center or radius. They are used only for evaluation.

## Data And Probe Setup

Dataset:

```text
saralazza/llada-safety-dataset
```

Safe prompts are rows where:

```text
source_dataset == alpaca
```

Unsafe prompts are all other sources.

Probe steps:

```text
1, 5, 10, 15, 20, 25, 30
```

Layer mapping from `best_layers.json`:

| Step | Layer |
| ---: | ---: |
| 1 | 26 |
| 5 | 23 |
| 10 | 23 |
| 15 | 22 |
| 20 | 22 |
| 25 | 22 |
| 30 | 22 |

Cached train-like evaluation used 50 safe and 50 unsafe probes per step. Held-out evaluation used 50 safe and 50 unsafe fresh HF test prompts.

Probe extraction command, when caches are missing:

```bash
PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
./run_probe_steps.sh
```

## Experiment Log Organization

All experiment artifacts are under `logs/`:

| Folder | Experiment | Status |
| --- | --- | --- |
| `exp_00_euclidean_minimal_svdd_nu001_cached_eval/` | Minimal Euclidean SVDD baseline, conservative `NU=0.01` | Complete |
| `exp_01_threshold_sweep_quantiles/` | Radius quantile sweep | Complete |
| `exp_02_multistep_or_rule/` | Multi-step OR trajectory guard | Complete |
| `exp_03_step_subset_or_ablation/` | Step-subset OR ablation | Complete |
| `exp_04_hf_heldout_eval/` | Held-out HF test evaluation | Complete |
| `exp_05_mahalanobis_euclidean/` | PCA + Ledoit-Wolf Mahalanobis baseline | Complete |

Each folder contains logs, CSV summaries, plots, and an `ANALYSIS.md` file.

## exp_00: Minimal Euclidean SVDD

Question:

```text
Does plain Euclidean distance to the safe centroid work as a first-pass one-class baseline?
```

Setup:

```text
center: safe centroid
score: Euclidean distance to center
radius: conservative NU=0.01 / approximately high safe-distance quantile
evaluation: independent per step on cached probes
```

Results:

| Step | Layer | AUROC | AUPR | Safe blocked | Unsafe blocked | ASR post-defense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 0.8736 | 0.8402 | 1/50 (2.0%) | 1/50 (2.0%) | 98.0% |
| 5 | 23 | 0.9320 | 0.9481 | 1/50 (2.0%) | 33/50 (66.0%) | 34.0% |
| 10 | 23 | 0.9208 | 0.9292 | 1/50 (2.0%) | 24/50 (48.0%) | 52.0% |
| 15 | 22 | 0.9076 | 0.9134 | 1/50 (2.0%) | 21/50 (42.0%) | 58.0% |
| 20 | 22 | 0.9024 | 0.9001 | 1/50 (2.0%) | 17/50 (34.0%) | 66.0% |
| 25 | 22 | 0.9088 | 0.9219 | 1/50 (2.0%) | 22/50 (44.0%) | 56.0% |
| 30 | 22 | 0.9136 | 0.9239 | 1/50 (2.0%) | 28/50 (56.0%) | 44.0% |

Aggregate across independently evaluated steps:

| Metric | Value |
| --- | ---: |
| Mean AUROC | 0.9084 |
| Mean AUPR | 0.9110 |
| Unsafe blocked | 146/350 (41.7%) |
| Safe blocked | 7/350 (2.0%) |
| ASR post-defense | 204/350 (58.3%) |

Analysis:

The Euclidean score ranks unsafe samples above safe samples reasonably well, especially from step 5 onward. However, the hard threshold is conservative, so the detector keeps false positives low while missing many unsafe prompts. Step 5 is the strongest single cached step under the original setting.

Artifacts:

```text
logs/exp_00_euclidean_minimal_svdd_nu001_cached_eval/
```

## exp_01: Threshold Sweep

Question:

```text
Was the minimal baseline weak because Euclidean geometry is weak, or because the threshold was too conservative?
```

Setup:

```text
same Euclidean centroid and score
radius quantiles: 0.90, 0.95, 0.98, 0.99
evaluation: independent per step on cached probes
```

Key results:

| Operating point | Quantile | Step | Safe blocked | Unsafe blocked | ASR post-defense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Highest attack blocking | 0.90 | 5 | 5/50 (10.0%) | 44/50 (88.0%) | 12.0% |
| Best low-FPR single step | 0.98 | 30 | 1/50 (2.0%) | 37/50 (74.0%) | 26.0% |
| Original conservative setting | 0.99 | 5 | 1/50 (2.0%) | 33/50 (66.0%) | 34.0% |

Analysis:

Threshold calibration matters. Lowering the safe-distance quantile substantially improves unsafe blocking. At `q=0.90`, unsafe blocking reaches 88%, but safe blocking rises to 10%. At `q=0.98`, the best low-FPR single-step result is step 30, blocking 74% unsafe with 2% safe blocking.

Artifacts:

```text
logs/exp_01_threshold_sweep_quantiles/
```

## exp_02: Multi-step OR Rule

Question:

```text
Does a trajectory-style OR rule improve Euclidean blocking compared with evaluating steps independently?
```

Setup:

```text
selected steps: 1 5 10 15 20 25 30
per-step radius calibrated by quantile
block if any selected step fires
```

Results:

| Quantile | Safe blocked | Unsafe blocked | ASR post-defense | AUROC margin | AUPR margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 9/50 (18.0%) | 44/50 (88.0%) | 12.0% | 0.9112 | 0.8715 |
| 0.95 | 4/50 (8.0%) | 42/50 (84.0%) | 16.0% | 0.9068 | 0.8700 |
| 0.98 | 2/50 (4.0%) | 39/50 (78.0%) | 22.0% | 0.9024 | 0.8669 |
| 0.99 | 2/50 (4.0%) | 36/50 (72.0%) | 28.0% | 0.9072 | 0.9015 |

Analysis:

The OR rule improves unsafe blocking compared with single-step evaluation, but false positives rise. The clean low-FPR setting is `q=0.98`, with 78% unsafe blocking and 4% safe blocking.

Artifacts:

```text
logs/exp_02_multistep_or_rule/
```

## exp_03: Step-subset OR Ablation

Question:

```text
Can a smaller set of steps match the full OR trajectory while reducing compute and false positives?
```

Subsets:

| Name | Steps |
| --- | --- |
| `early` | 1 5 10 |
| `critical` | 5 10 15 |
| `late` | 20 25 30 |
| `best_pair` | 5 30 |
| `all` | 1 5 10 15 20 25 30 |

Key results:

| Operating point | Subset | Quantile | Safe blocked | Unsafe blocked | ASR post-defense |
| --- | --- | ---: | ---: | ---: | ---: |
| Best low-FPR subset | 5 30 | 0.98 | 2/50 (4.0%) | 39/50 (78.0%) | 22.0% |
| Highest blocking tested | 1 5 10 | 0.95 | 4/50 (8.0%) | 42/50 (84.0%) | 16.0% |
| Full OR comparison | 1 5 10 15 20 25 30 | 0.98 | 2/50 (4.0%) | 39/50 (78.0%) | 22.0% |

Analysis:

The pair `5 30` matches the full seven-step OR result at `q=0.98` while requiring only two detector checks. This is the cleanest cached Euclidean trajectory baseline.

Artifacts:

```text
logs/exp_03_step_subset_or_ablation/
```

## exp_04: Held-out HF Test Evaluation

Question:

```text
Do the best Euclidean settings generalize to fresh held-out HF test prompts?
```

Setup:

```text
held-out split: Hugging Face test
safe samples: 50
unsafe samples: 50
radii: calibrated from training/cache safe probes
held-out probes: used only for evaluation
```

Configurations:

| Config | Steps | Quantile |
| --- | --- | ---: |
| `single_step_5_q098` | 5 | 0.98 |
| `single_step_30_q098` | 30 | 0.98 |
| `trajectory_5_30_q098` | 5 30 | 0.98 |
| `aggressive_1_5_10_q095` | 1 5 10 | 0.95 |

Results:

| Config | Safe blocked | Unsafe blocked | ASR post-defense | AUROC margin | AUPR margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_step_5_q098` | 0/50 (0.0%) | 30/50 (60.0%) | 40.0% | 0.9516 | 0.9569 |
| `single_step_30_q098` | 0/50 (0.0%) | 32/50 (64.0%) | 36.0% | 0.9412 | 0.9555 |
| `trajectory_5_30_q098` | 0/50 (0.0%) | 34/50 (68.0%) | 32.0% | 0.9448 | 0.9587 |
| `aggressive_1_5_10_q095` | 4/50 (8.0%) | 41/50 (82.0%) | 18.0% | 0.9324 | 0.9428 |

Analysis:

The held-out results confirm that the Euclidean baseline generalizes. The best clean low-FPR held-out setting is `trajectory_5_30_q098`, which blocks 68% unsafe prompts with 0% safe blocking. The aggressive setting blocks 82% unsafe prompts but raises safe blocking to 8%.

Artifacts:

```text
logs/exp_04_hf_heldout_eval/
```

## exp_05: PCA + Ledoit-Wolf Mahalanobis Euclidean

Question:

```text
Does a stronger covariance-aware Euclidean distance outperform plain Euclidean distance?
```

Definition:

Plain Euclidean assumes a spherical safe region:

```text
d_E(x) = ||x - mu_safe||_2
```

Mahalanobis distance uses covariance structure:

```text
d_M(x) = sqrt((x - mu_safe)^T Sigma^{-1} (x - mu_safe))
```

Because hidden states are high-dimensional and safe training samples are few, this experiment first applies PCA:

```text
z = W_k^T (x - mu_pca)
```

Then it fits Ledoit-Wolf shrinkage covariance:

```text
Sigma_lw = (1 - lambda) S + lambda tau I
```

The final score and decision rule are:

```text
d_M(z) = sqrt((z - mu_safe)^T Sigma_lw^{-1} (z - mu_safe))
R = quantile_q({d_M(z_i_safe_train)})
block if d_M(z_test) > R
```

Results:

| Config | Safe blocked | Unsafe blocked | ASR post-defense | AUROC margin | AUPR margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_step_5_q098` | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.7308 | 0.6545 |
| `single_step_30_q098` | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.5460 | 0.4840 |
| `trajectory_5_30_q098` | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.6980 | 0.6269 |
| `aggressive_1_5_10_q095` | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.6824 | 0.5848 |

Analysis:

Mahalanobis is a negative result here. It blocks no held-out unsafe samples at the tested train-calibrated thresholds, and its ranking AUROC/AUPR are much weaker than plain Euclidean. In this high-dimensional, low-sample regime, PCA plus shrinkage covariance does not produce a useful safety boundary.

Artifacts:

```text
logs/exp_05_mahalanobis_euclidean/
```

## Cross-experiment Summary

Best held-out Euclidean operating points:

| Baseline | Steps | Quantile | Safe blocked | Unsafe blocked | ASR post-defense |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean plain Euclidean trajectory | 5 30 | 0.98 | 0/50 (0.0%) | 34/50 (68.0%) | 32.0% |
| Aggressive plain Euclidean trajectory | 1 5 10 | 0.95 | 4/50 (8.0%) | 41/50 (82.0%) | 18.0% |
| Mahalanobis Euclidean | 5 30 | 0.98 | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% |

Comparison to the hyperbolic result in `info.md`:

```text
Hyperbolic step 5 result from info.md:
unsafe blocked: 155/157 (98.7%)
safe blocked:     1/106 (0.9%)
```

The tuned Euclidean baseline is meaningful and generalizes, but the hyperbolic detector is substantially stronger as an operational guard.

## Conclusion

The Euclidean baseline study answers the original question clearly:

```text
Euclidean hidden-state geometry contains real safety signal, but it does not produce as clean a safety boundary as the hyperbolic detector.
```

Key conclusions:

- Plain Euclidean distance to the safe centroid is a valid one-class baseline.
- Threshold calibration is essential; the original conservative radius underestimates Euclidean performance.
- A trajectory OR rule improves unsafe blocking.
- The best compact Euclidean trajectory uses steps `5 30` at `q=0.98`.
- Held-out HF evaluation confirms generalization: 68% unsafe blocking with 0% safe blocking.
- An aggressive Euclidean setting can reach 82% unsafe blocking, but with 8% safe blocking.
- PCA + Ledoit-Wolf Mahalanobis is not competitive in this setting.
- The hyperbolic detector remains clearly better than the best Euclidean baselines.

Recommended Euclidean baseline to report:

```text
Clean held-out Euclidean trajectory baseline:
steps: 5 30
quantile: 0.98
unsafe blocked: 34/50 (68.0%)
safe blocked: 0/50 (0.0%)
ASR post-defense: 32.0%
```

Recommended aggressive Euclidean reference point:

```text
Aggressive held-out Euclidean trajectory baseline:
steps: 1 5 10
quantile: 0.95
unsafe blocked: 41/50 (82.0%)
safe blocked: 4/50 (8.0%)
ASR post-defense: 18.0%
```
