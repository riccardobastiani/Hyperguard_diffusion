# README

## Environment Setup

The working GPU environment uses a Python 3.12 virtual environment named
`.venv312`. The environment stays on disk between server sessions, but each new
terminal/login needs to activate it again.

Create the environment once:

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion

python3.12 -m venv .venv312
source .venv312/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install --force-reinstall \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

For every new server session, reactivate it:

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion
source .venv312/bin/activate
```

Verify CUDA is visible to PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected output should include:

```text
2.6.0+cu124
12.4
True
```

Use `.venv312/bin/python` in scripts:

```bash
PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
./run_probe_steps.sh
```

## Euclidean Baseline

Fit one Euclidean SVDD checkpoint per cached probe step:

```bash
python euclidean_baseline.py \
  --probe-dir probe_outputs/step_5/layer_23 \
  --probe-steps 5 \
  --nu 0.01
```

Evaluate it with the same metrics path used by the hyperbolic detector:

```bash
python test_guard.py \
  --detector-type euclidean \
  --ckpt-dir probe_outputs/step_5/layer_23 \
  --probe-steps 5 \
  --eval-all-safe \
  --eval-all-unsafe \
  --metrics \
  --device cpu
```

## Euclidean Baseline Results

Command used:

```bash
PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
DEVICE=cpu \
./run_euclidean_eval.sh
```

Per-step results on the cached 50 safe / 50 unsafe probe sets:

| Step | Layer | R | AUROC | AUPR | Safe blocked | Unsafe blocked | ASR post-defense | Confusion TP/FP/TN/FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 0.4603 | 0.8736 | 0.8402 | 1/50 (2.0%) | 1/50 (2.0%) | 98.0% | 1/1/49/49 |
| 5 | 23 | 0.2575 | 0.9320 | 0.9481 | 1/50 (2.0%) | 33/50 (66.0%) | 34.0% | 33/1/49/17 |
| 10 | 23 | 0.2648 | 0.9208 | 0.9292 | 1/50 (2.0%) | 24/50 (48.0%) | 52.0% | 24/1/49/26 |
| 15 | 22 | 0.2526 | 0.9076 | 0.9134 | 1/50 (2.0%) | 21/50 (42.0%) | 58.0% | 21/1/49/29 |
| 20 | 22 | 0.2572 | 0.9024 | 0.9001 | 1/50 (2.0%) | 17/50 (34.0%) | 66.0% | 17/1/49/33 |
| 25 | 22 | 0.2521 | 0.9088 | 0.9219 | 1/50 (2.0%) | 22/50 (44.0%) | 56.0% | 22/1/49/28 |
| 30 | 22 | 0.2505 | 0.9136 | 0.9239 | 1/50 (2.0%) | 28/50 (56.0%) | 44.0% | 28/1/49/22 |

Aggregate across these seven independently evaluated steps:

| Metric | Value |
| --- | ---: |
| Mean AUROC | 0.9084 |
| Mean AUPR | 0.9110 |
| Unsafe blocked | 146/350 (41.7%) |
| Safe blocked | 7/350 (2.0%) |
| ASR post-defense | 204/350 (58.3%) |
| Aggregate confusion TP/FP/TN/FN | 146/7/343/204 |

Interpretation:

The Euclidean baseline separates safe and unsafe probes reasonably well as a ranking method: AUROC stays around 0.90 or higher after step 5, and AUPR peaks at step 5 with 0.9481. However, the deployed threshold is intentionally conservative because the radius is fitted from the safe distribution with `NU=0.01`. That keeps false positives very low at 2.0%, but it misses many unsafe prompts. Step 5 is the strongest Euclidean checkpoint, blocking 66.0% of unsafe samples with only 2.0% safe blocking. Later steps remain useful but weaker, blocking 34.0-56.0% of unsafe samples.

Compared with the hyperbolic result in `info.md` at step 5, the Euclidean baseline is substantially weaker as an actual defense. The hyperbolic detector reported 155/157 unsafe blocked (98.7%) and 1/106 safe blocked (0.9%), while the Euclidean step-5 baseline blocks 33/50 unsafe (66.0%) and 1/50 safe (2.0%). This supports the project hypothesis: Euclidean hidden-state distance contains signal, but the hyperbolic projection plus SVDD geometry produces a much cleaner safety boundary.

## Future Euclidean Baseline TODOs

The current Euclidean baseline is the minimal one-class SVDD version: safe centroid, Euclidean distance, and radius threshold. It is sufficient as a baseline, but the following experiments would make the comparison stronger:

- Threshold sweep: evaluate several safe-distance quantiles such as 0.90, 0.95, 0.98, and 0.99 to report the TPR/FPR tradeoff instead of only the current conservative `NU=0.01` setting.
- Multi-step OR rule: evaluate a trajectory-style guard that blocks if any Euclidean detector fires across steps. This is closer to the intended defense behavior than treating each step independently.
- L2 normalization ablation: compare cached/raw probes against explicitly L2-normalized probes to verify whether normalization helps Euclidean distance.
- Layer ablation: sweep layers for the Euclidean baseline instead of reusing `best_layers.json`, since the best Euclidean layer may differ from the best hyperbolic layer.
- Stronger Euclidean classifiers: compare against logistic regression, linear SVM, one-class SVM, Mahalanobis distance, and PCA + centroid distance.
- Train/test split discipline: fit the Euclidean center/radius on train safe probes and evaluate on held-out HF test probes to avoid reporting train-cache performance as final test performance.

Recommended next two experiments:

```text
1. Threshold sweep over radius quantile
2. Multi-step OR evaluation across probe steps
```
