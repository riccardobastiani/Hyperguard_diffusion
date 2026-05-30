# exp_01: Euclidean Threshold Sweep Analysis

## Setup

This experiment recalibrates the Euclidean radius `R` using several safe-distance quantiles while keeping the same Euclidean centroid and distance score.

Quantiles evaluated:

```text
0.90, 0.95, 0.98, 0.99
```

Each row evaluates 50 safe and 50 unsafe cached probes at the selected denoising step/layer.

## Key Findings

Best raw unsafe block rate:

```text
q=0.90, step=5, layer=23
unsafe blocked: 44/50 (88.0%)
safe blocked:    5/50 (10.0%)
ASR post-defense: 12.0%
AUROC/AUPR: 0.9320 / 0.9481
```

Best configuration with safe block rate <= 5%:

```text
q=0.98, step=30, layer=22
unsafe blocked: 37/50 (74.0%)
safe blocked:    1/50 (2.0%)
ASR post-defense: 26.0%
AUROC/AUPR: 0.9136 / 0.9239
```

Strong low-FPR alternative:

```text
q=0.98, step=5, layer=23
unsafe blocked: 33/50 (66.0%)
safe blocked:    1/50 (2.0%)
ASR post-defense: 34.0%
AUROC/AUPR: 0.9320 / 0.9481
```

## Interpretation

The original `NU=0.01` baseline was equivalent to a very conservative high quantile, close to `q=0.99`. The threshold sweep shows that the Euclidean score is more useful than the initial hard-threshold result suggested.

Lowering the radius to `q=0.90` gives strong attack blocking: most steps block 84-88% of unsafe prompts. The cost is a 10% safe block rate. This may be too high for a production guard, but it is useful as an upper-bound operating point for the Euclidean baseline.

At `q=0.95`, the baseline is more balanced: safe block rate is 6%, while unsafe block rate ranges from 68-80%. This is probably the best mid-point if a small false-positive increase is acceptable.

At `q=0.98`, the baseline keeps false positives low at 2% while improving over the original result at several steps. Step 30 becomes the best low-FPR setting, blocking 74% unsafe at 2% FPR.

At `q=0.99`, results match the original conservative baseline and miss many unsafe samples.

Overall, the experiment shows threshold calibration matters. The Euclidean baseline is not as weak as the original `NU=0.01` result alone implied, but even after tuning it still remains below the hyperbolic step-5 result reported in `info.md`.

## Artifacts

- `threshold_sweep_summary.csv`
- `threshold_sweep_by_step.png`
- `threshold_sweep_step5_tradeoff.png`
