# exp_03: Step-subset OR Ablation

## Setup

This experiment evaluates multi-step OR guards over predefined subsets of denoising steps. The goal is to see whether a smaller trajectory subset gives a better unsafe-blocking / false-positive tradeoff than using all steps.

Subsets:

- `early`: 1 5 10
- `critical`: 5 10 15
- `late`: 20 25 30
- `best_pair`: 5 30
- `all`: 1 5 10 15 20 25 30

Quantiles:

```text
0.95, 0.98
```

## Key Findings

Highest unsafe blocking:

```text
subset=early, q=0.95
unsafe blocked: 42/50 (84.0%)
safe blocked:    4/50 (8.0%)
ASR post-defense: 16.0%
```

Best low-FPR setting:

```text
subset=best_pair, q=0.98
steps: 5 30
unsafe blocked: 39/50 (78.0%)
safe blocked:    2/50 (4.0%)
ASR post-defense: 22.0%
AUROC/AUPR margin: 0.9140 / 0.9288
```

Important comparison:

```text
all steps, q=0.98:       39/50 unsafe blocked, 2/50 safe blocked
best_pair 5 30, q=0.98: 39/50 unsafe blocked, 2/50 safe blocked
```

The two-step `best_pair` subset exactly matches the full seven-step OR result at `q=0.98`, while requiring fewer checkpoints and fewer guard evaluations. This is the cleanest low-FPR Euclidean trajectory baseline found so far.

At `q=0.95`, the full/early subsets reach 84% unsafe blocking but with 8% safe blocking. The `best_pair` and `critical` subsets are close behind at 82% unsafe blocking with 6% safe blocking.

## Interpretation

Using every probe step is not necessary for the best low-FPR Euclidean OR result. The step pair `5 30` captures the same low-FPR blocking behavior as the full seven-step trajectory at `q=0.98`. This suggests that Euclidean anomaly evidence appears both early and late, and a small two-point trajectory can be enough for this baseline.

For reporting, the recommended Euclidean OR operating point is:

```text
steps: 5 30
quantile: 0.98
unsafe blocked: 78.0%
safe blocked: 4.0%
ASR post-defense: 22.0%
```

If higher false positives are acceptable, `early` at `q=0.95` gives stronger blocking:

```text
steps: 1 5 10
quantile: 0.95
unsafe blocked: 84.0%
safe blocked: 8.0%
ASR post-defense: 16.0%
```

## Artifacts

- `step_subset_or_summary.csv`
- `step_subset_or_rates.png`
