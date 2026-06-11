# exp_05: PCA + Ledoit-Wolf Mahalanobis Euclidean Baseline

## Definition

Given safe hidden-state features `x`, this baseline first reduces dimensionality with PCA and then fits a shrinkage covariance model on safe examples.

Plain Euclidean distance assumes a spherical safe region:

```text
d_E(x) = ||x - mu_safe||_2
```

PCA projection:

```text
z = W_k^T (x - mu_pca)
```

Ledoit-Wolf shrinkage covariance:

```text
Sigma_lw = (1 - lambda) S + lambda tau I
```

Mahalanobis distance:

```text
d_M(z) = sqrt((z - mu_safe)^T Sigma_lw^{-1} (z - mu_safe))
```

Decision rule:

```text
R = quantile_q({d_M(z_i_safe_train)})
block if d_M(z_test) > R
```

PCA dimension cap: 32

## Results

| Config | Steps | q | Safe blocked | Unsafe blocked | ASR post-defense | AUROC margin | AUPR margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_step_5_q098` | 5 | 0.98 | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.7308 | 0.6545 |
| `single_step_30_q098` | 30 | 0.98 | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.5460 | 0.4840 |
| `trajectory_5_30_q098` | 5 30 | 0.98 | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.6980 | 0.6269 |
| `aggressive_1_5_10_q095` | 1 5 10 | 0.95 | 0/50 (0.0%) | 0/50 (0.0%) | 100.0% | 0.6824 | 0.5848 |

## Key Findings

All tested Mahalanobis configurations blocked 0/50 unsafe samples and 0/50 safe samples on held-out HF evaluation. This means the train-calibrated Mahalanobis radii were too loose for held-out blocking, and the score did not separate the held-out unsafe prompts strongly enough.

The ranking quality is also much weaker than plain Euclidean. The best Mahalanobis AUROC is 0.7308 at step 5, while the plain Euclidean held-out configs had AUROC around 0.94-0.95.

## Interpretation

This is a useful negative result. Although Mahalanobis is a stronger covariance-aware Euclidean distance in principle, it does not help here under the stable PCA + Ledoit-Wolf setup. The likely reason is the difficult high-dimensional / low-sample regime: each step has only about 50 safe training examples, so even shrinkage covariance after PCA may estimate a safe ellipsoid that is too broad or poorly aligned for unsafe detection.

Compared with exp_04 plain Euclidean:

```text
Plain Euclidean trajectory 5 30 q=0.98:
unsafe blocked: 34/50 (68.0%)
safe blocked:    0/50 (0.0%)
AUROC margin: 0.9448

Mahalanobis trajectory 5 30 q=0.98:
unsafe blocked: 0/50 (0.0%)
safe blocked:   0/50 (0.0%)
AUROC margin: 0.6980
```

Conclusion: the covariance-aware Euclidean baseline is not competitive with the tuned plain Euclidean baseline, and both remain weaker than the hyperbolic detector reported in `info.md`.

## Optional Follow-up

A broader Mahalanobis sweep could test lower quantiles or PCA dimensions such as 8, 16, and 48. However, given the weak AUROC/AUPR here, this is lower priority than reporting the negative result.

## Artifacts

- `mahalanobis_heldout.log`
- `mahalanobis_heldout_summary.csv`
- `mahalanobis_heldout_rates.png`
