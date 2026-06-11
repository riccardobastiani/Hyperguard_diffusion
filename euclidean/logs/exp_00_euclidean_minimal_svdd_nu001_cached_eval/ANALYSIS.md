# exp_00: Minimal Euclidean SVDD Baseline

## Setup

- Detector: one-class Euclidean SVDD baseline.
- Center: mean of safe hidden-state probes.
- Score: Euclidean distance to the safe centroid.
- Radius: safe-distance quantile induced by `NU=0.01`, approximately the 99th percentile of safe distances.
- Evaluation: cached 50 safe / 50 unsafe probes, evaluated independently at steps 1, 5, 10, 15, 20, 25, and 30.

## Artifacts

- Raw log: `euclidean_eval_20260529_214337.log`
- Parsed CSV: `euclidean_eval_20260529_214337_summary.csv`
- Metrics plot: `euclidean_eval_20260529_214337_metrics.png`
- Distance plot: `euclidean_eval_20260529_214337_distances.png`

## Key Result

Best checkpoint is step 5 / layer 23:

```text
AUROC: 0.9320
AUPR: 0.9481
Unsafe blocked: 33/50 (66.0%)
Safe blocked: 1/50 (2.0%)
ASR post-defense: 34.0%
```

Aggregate across independently evaluated steps:

```text
Mean AUROC: 0.9084
Mean AUPR: 0.9110
Unsafe blocked: 146/350 (41.7%)
Safe blocked: 7/350 (2.0%)
ASR post-defense: 204/350 (58.3%)
Aggregate confusion TP/FP/TN/FN: 146/7/343/204
```

## Interpretation

The Euclidean score ranks unsafe samples above safe samples reasonably well, especially from step 5 onward. However, the hard radius threshold is conservative, keeping false positives low while missing many unsafe prompts. This is a useful baseline because it shows raw Euclidean hidden states contain signal, but the hyperbolic detector has a much cleaner operational safety boundary.
