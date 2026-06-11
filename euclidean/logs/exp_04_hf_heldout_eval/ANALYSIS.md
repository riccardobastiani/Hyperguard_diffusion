# exp_04: Held-out HF Test Evaluation

## Setup

This experiment evaluates selected Euclidean baseline configurations on fresh held-out prompts from the Hugging Face test split. Radii are calibrated from the training/cache safe probes, not from held-out test probes.

Held-out sample count:

```text
safe: 50
unsafe: 50
```

## Configurations

- `single_step_5_q098`: steps=5, q=0.98
- `single_step_30_q098`: steps=30, q=0.98
- `trajectory_5_30_q098`: steps=5 30, q=0.98
- `aggressive_1_5_10_q095`: steps=1 5 10, q=0.95

## Results

| Config | Steps | q | Safe blocked | Unsafe blocked | ASR post-defense | AUROC margin | AUPR margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_step_5_q098` | 5 | 0.98 | 0/50 (0.0%) | 30/50 (60.0%) | 40.0% | 0.9516 | 0.9569 |
| `single_step_30_q098` | 30 | 0.98 | 0/50 (0.0%) | 32/50 (64.0%) | 36.0% | 0.9412 | 0.9555 |
| `trajectory_5_30_q098` | 5 30 | 0.98 | 0/50 (0.0%) | 34/50 (68.0%) | 32.0% | 0.9448 | 0.9587 |
| `aggressive_1_5_10_q095` | 1 5 10 | 0.95 | 4/50 (8.0%) | 41/50 (82.0%) | 18.0% | 0.9324 | 0.9428 |

## Key Findings

Highest unsafe blocking:

```text
aggressive_1_5_10_q095
unsafe blocked: 41/50 (82.0%)
safe blocked:    4/50 (8.0%)
ASR post-defense: 18.0%
```

Best low-FPR setting:

```text
trajectory_5_30_q098
unsafe blocked: 34/50 (68.0%)
safe blocked:    0/50 (0.0%)
ASR post-defense: 32.0%
```

## Interpretation

The held-out results confirm that the Euclidean baseline generalizes: AUROC/AUPR remain high across all evaluated configurations, with margin AUROC between 0.9324 and 0.9516.

The conservative `q=0.98` settings become extremely safe on held-out prompts: all three low-FPR configurations block 0/50 safe samples. The cost is lower attack blocking compared with cached train-like evaluation. The two-step trajectory `5 30` still improves over either single step: 68% unsafe blocking versus 60% at step 5 and 64% at step 30.

The aggressive `1 5 10` configuration reaches 82% unsafe blocking, but safe blocking rises to 8%. This is a useful upper-blocking operating point, but the cleaner reportable low-FPR Euclidean trajectory baseline is `5 30` at `q=0.98`.

Compared with the hyperbolic detector result in `info.md`, Euclidean remains weaker as a defense, but it is now a well-tuned and held-out-tested baseline rather than a strawman.

## Artifacts

- `hf_heldout.log`
- `hf_heldout_summary.csv`
- `hf_heldout_rates.png`
- `heldout_safe_features.npz`
- `heldout_unsafe_features.npz`
