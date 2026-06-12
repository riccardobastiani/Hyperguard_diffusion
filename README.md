# HyperGuard Diffusion

This repository brings together several related experiments around safety in
diffusion-based language models:

- **HyperGuard**: hyperbolic SVDD detection on LLaDA hidden-state trajectories.
- **DiffuGuard**: intrinsic-safety analysis, hidden audits, and repair for dLLMs.
- **DIJA dataset and attack pipeline**: interleaved mask-text jailbreak attacks.
- **Euclidean baselines**: non-hyperbolic trajectory detectors for comparison.
- **Layer analysis**: conceptual and probing guidance for selecting layers/steps.
- **Multimodal HyperGuard**: GPO-V/LLaDA-V multimodal trajectory detection.

The root project focuses on LLaDA hidden-state probing and HyperGuard detector
training/evaluation. The subdirectories preserve the related project code and
experiment logs, while this README acts as the single entry point.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `dataset.py` | Loads, balances, shuffles, and validates safe/unsafe prompt datasets. |
| `generate.py` | LLaDA generation helpers plus intermediate hidden-state probing. |
| `hyperbolic_projection.py` | Projects Euclidean hidden states onto the Lorentz hyperboloid. |
| `svdd.py` | Hyperbolic SVDD model, Lorentz distance, training, checkpoint I/O. |
| `probe_analysis.py` | Extracts LLaDA probes and trains SVDD detectors for selected steps. |
| `test_guard.py` | Evaluates trained detectors, cached probes, HF live inference, and metrics. |
| `run_probe_steps.sh` | Sweeps probe steps and aggregates step/layer outputs. |
| `best_layers.json` | Best layer mapping used by several baseline/probing experiments. |
| `diffuguard/` | DiffuGuard analysis and runtime repair experiments. |
| `dija_dataset/` | DIJA attack/dataset code and benchmark evaluation pipeline. |
| `euclidean/` | Euclidean safety-trajectory baselines and experiment summaries. |
| `layer_analysis/` | Conceptual HyperGuard/layer-probing notes. |
| `multimodal/` | Multimodal GPO-V HyperGuard experiments for LLaDA-V. |

## Core Idea

Diffusion LLMs generate text through iterative denoising. HyperGuard treats the
model's internal hidden states during this process as a trajectory. It extracts
mean-pooled hidden states at selected denoising steps, projects them into
hyperbolic space, and trains a soft-boundary SVDD detector to learn a geometric
safety region.

At evaluation time, prompts whose projected trajectory lies outside the learned
safe region are classified as unsafe.

The current root implementation performs this classification offline or after
feature extraction. It does **not** interrupt the active generation loop early.

## Environment

Install the root project dependencies:

```bash
pip install -r requirements.txt
```

Important libraries include:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `geoopt`

Most runtime options are supplied through CLI arguments in `probe_analysis.py`
and `test_guard.py`; there is no central YAML/JSON configuration file.

## HyperGuard Workflow

### 1. Data Preparation

The root pipeline uses `saralazza/llada-safety-dataset` by default.

- Safe prompts: rows where `source_dataset == "alpaca"`.
- Unsafe prompts: all other source datasets.
- Safe prompt fields are resolved from `refined_behavior`, `behavior`,
  `instruction`, then `input`.
- Unsafe prompt fields are resolved from `refined_behavior`, `prompt`,
  `behavior`, `instruction`, then `input`.

The dataset is balanced to `samples_per_class` rows per label and shuffled with
the configured seed.

You can also pass a local safe file through `--safe-local-path`.

### 2. Probe Extraction

`generate.py` implements `generate_with_layer_probes(...)`, which runs LLaDA and
captures hidden states at selected global denoising steps.

The capture path:

```text
prompt
  -> chat template and left padding
  -> LLaDA denoising with output_hidden_states=True at probe steps
  -> selected transformer layer hidden state
  -> mean pooling with attention mask
  -> float32 CPU feature array [batch_size, 4096]
```

No PyTorch forward hooks are registered; the generation loop directly requests
hidden states at target steps.

### 3. Hyperbolic Projection

`HyperbolicProjection` maps each Euclidean feature `x in R^4096` into Lorentz
space:

```text
v = W x
v' = max_norm * v / ||v||_2
u = [0, v']
emb = expmap_0(u)
```

The output has shape `[batch_size, proj_dim + 1]`, where `proj_dim` defaults to
`128`.

Optional input L2 normalization is available with `--l2-normalize-probes`.

### 4. SVDD Training

`HyperbolicSVDD` learns a radius around a fixed safe center.

Center initialization:

- Project safe features.
- Compute the Euclidean mean of projected safe features.
- Zero the time coordinate.
- Rescale spatial components.
- Map the center onto the Lorentz hyperboloid with `expmap0`.

The implementation uses a contrastive SVDD objective with safe and unsafe
samples. Safe samples are pulled inside the radius; unsafe samples are pushed
outside it. This avoids collapse that can occur in a pure one-class setup.

Checkpoints are saved as:

```text
svdd_step_<step>.pt
```

and include the projector weights, center, radius parameter, and hyperparameter
configuration.

### 5. Evaluation

`test_guard.py` supports cached-probe evaluation and live Hugging Face test
inference.

Metrics include:

- AUROC
- AUPR
- Attack Success Rate post-defense
- TP / FP / TN / FN confusion counts

The decision rule is:

```text
unsafe if LorentzDistance(embedding, center) > radius
```

## Root Commands

### Train HyperGuard on LLaDA

Example: train on layer 23, denoising step 5, using a local DIJA safe file.

```bash
python probe_analysis.py \
  --model-name GSAI-ML/LLaDA-8B-Instruct \
  --safe-local-path AlpacaDIJA/llada_instruct_DIJA_v1.json \
  --safe-local-field Refined_behavior \
  --samples-per-class 50 \
  --batch-size 1 \
  --probe-steps 5 \
  --layer-id 23 \
  --l2-normalize-probes \
  --output-dir probe_outputs/layer_23
```

This writes:

```text
probe_outputs/layer_23/safe_probes.npz
probe_outputs/layer_23/unsafe_probes.npz
probe_outputs/layer_23/svdd_step_5.pt
```

### Evaluate from Cached Probes

```bash
python test_guard.py \
  --ckpt-dir probe_outputs/layer_23 \
  --probe-steps 5 \
  --eval-all-safe \
  --eval-all-unsafe \
  --metrics \
  --device cuda
```

### Evaluate with Fresh Hugging Face Test Inference

```bash
python test_guard.py \
  --ckpt-dir probe_outputs/layer_23 \
  --probe-steps 5 \
  --hf-test \
  --hf-split test \
  --samples-per-class 50 \
  --layer-id 23 \
  --device cuda
```

### Sweep Probe Steps

```bash
PYTHON=.venv312/bin/python \
PROBE_STEPS="1 5 10 15 20 25 30" \
./run_probe_steps.sh
```

## Data Flow

```text
Input prompt
  -> instruct template and left padding
  -> LLaDA probing inference
  -> hidden state [batch, seq_len, 4096]
  -> attention-mask mean pooling
  -> Euclidean feature [batch, 4096]
  -> optional L2 normalization
  -> linear projection and fixed-norm scaling
  -> tangent vector at origin
  -> Lorentz exponential map
  -> hyperbolic embedding
  -> SVDD distance and radius decision
```

## Control Flow

Training:

```text
probe_analysis.py
  -> dataset.py: load_balanced_prompt_dataset
  -> generate.py: generate_with_layer_probes
  -> svdd.py: init_center
  -> svdd.py: train_svdd
  -> svdd.py: save_checkpoint
```

Evaluation:

```text
test_guard.py
  -> load test prompts or cached probes
  -> load LLaDA assets if live inference is enabled
  -> generate.py: generate_with_layer_probes
  -> svdd.py: load_checkpoint
  -> svdd.py: predict / classify
  -> metric reporting
```

## DiffuGuard (`diffuguard/`)

DiffuGuard is the official repository for the paper:

```text
DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models
```

It studies how intrinsic safety can be lost during diffusion generation and
provides hidden-state audit and repair-style defenses for LLaDA, Dream, and
MMaDA-style models.

### DiffuGuard Setup

```bash
conda create -n diffuguard python==3.11
conda activate diffuguard
pip install -r requirements.txt
```

Download dLLMs locally:

```bash
bash hf_models/model_download.sh
```

Optional evaluator credentials for OpenAI-based ASR evaluation:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=your_url
```

### DiffuGuard Core Experiments

| Experiment | Command |
| --- | --- |
| Logits heatmap / Figure 2 | `python analysis/heatmap.py` |
| Random remasking / Figure 3 | `bash analysis/exp_remask_randomness.sh` |
| Token injection / Figures 4 and 5 | `bash analysis/exp_token_injection.sh` |
| Batch result evaluation | `bash analysis/eval.sh` |

Edit `MODEL_PATH` and dataset settings inside the analysis scripts when needed.

### DiffuGuard Quick Start

LLaDA with PAD attack, hidden audit, and repair:

```bash
python models/jailbreakbench_llada.py \
  --model_path hf_models/LLaDA-8B-Instruct \
  --attack_method PAD \
  --attack_prompt path/to/prompts.json \
  --output_json out_llada.json \
  --steps 64 --gen_length 128 --block_length 128 \
  --sp_mode hidden --sp_threshold 0.35 \
  --refinement_steps 8 --remask_ratio 0.9
```

Dream:

```bash
python models/jailbreakbench_dream.py \
  --model_path hf_models/Dream-v0-Instruct-7B \
  --attack_method pad \
  --attack_prompt path/to/prompts.json \
  --output_json out_dream.json \
  --gen_length 128 --steps 64 --mask_counts 36 \
  --sp_mode hidden --sp_threshold 0.35 \
  --refinement_steps 8 --remask_ratio 0.9
```

MMaDA:

```bash
python models/jailbreakbench_mmada.py \
  --model_path hf_models/MMaDA-8B-MixCoT \
  --attack_method PAD \
  --attack_prompt path/to/prompts.json \
  --output_json out_mmada.json \
  --steps 64 \
  --sp_mode hidden --sp_threshold 0.35 \
  --refinement_steps 8 --remask_ratio 0.9
```

Key hyperparameters:

- `steps`, `gen_length`, `block_length`: diffusion steps and decoding span.
- `remasking`: `off`, `low_confidence`, or `adaptive_step`.
- `sp_mode`: `off` or `hidden`.
- `sp_threshold`: hidden-audit threshold.
- `refinement_steps` and `remask_ratio`: repair/refinement settings.

## DIJA Dataset and Attack Pipeline (`dija_dataset/`)

DIJA is the code and dataset pipeline for:

```text
The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs
```

It investigates safety issues in diffusion LLMs and introduces an automated
jailbreak pipeline that converts ordinary jailbreak prompts into interleaved
text-mask jailbreak prompts.

### DIJA Highlights

- Targets bidirectional and parallel decoding behavior in dLLMs.
- Evaluates attacks across LLaDA, LLaDA-1.5, Dream, MMaDA, DiffuCoder, and
  Dream-Coder variants.
- Supports HarmBench, JailbreakBench, and StrongREJECT workflows.
- Supports defense toggles such as `None`, `Self-reminder`, and `RPO`.

### DIJA Setup

```bash
git clone https://github.com/ZichenWen1/DIJA
cd DIJA
cd hf_models && bash model_download.sh
cd ..
conda create -n DIJA python=3.10 -y
conda activate DIJA
pip install -r requirements.txt
```

### DIJA Parameters

| Placeholder | Meaning |
| --- | --- |
| `[Version]` | Version name/number for the run. |
| `[Defense_method]` | `None`, `Self-reminder`, or `RPO`. |
| `[Victim_model]` | `llada_instruct`, `llada_1.5`, `dream_instruct`, or `mmada_mixcot`. |

### DIJA HarmBench Evaluation

```bash
cd run_harmbench
bash refine_prompt/run_refine.sh [Version]
bash eval_harmbench.sh DIJA [Defense_method] [Victim_model] [Version]
```

### DIJA JailbreakBench Evaluation

```bash
cd run_jailbreakbench
bash refine_prompt/run_refine.sh [Version]
bash eval_jailbreakbench.sh DIJA [Defense_method] [Victim_model] [Version]
bash eval_jailbreakbench.sh DIJA none llada_instruct v1
```

### DIJA StrongREJECT Evaluation

```bash
cd run_strongreject
bash refine_prompt/run_refine.sh [Version]
bash eval_strongreject.sh DIJA [Defense_method] [Victim_model] [Version]
```

### DIJA Status

- Released inference and evaluation code.
- Supports DiffuCoder and Dream-Coder.
- Released interleaved mask-text prompts.
- AdvBench evaluation support is listed as future work.

### DIJA Citation

```bibtex
@article{wen2025devil,
  title={The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs},
  author={Wen, Zichen and Qu, Jiashu and Liu, Dongrui and Liu, Zhiyuan and Wu, Ruixi and Yang, Yicun and Jin, Xiangqi and Xu, Haoyun and Liu, Xuyang and Li, Weijia and others},
  journal={arXiv preprint arXiv:2507.11097},
  year={2025}
}
```

## Euclidean Baselines (`euclidean/`)

The Euclidean folder asks:

```text
How much safety signal is available from ordinary Euclidean geometry on LLaDA hidden-state trajectories before using hyperbolic projection?
```

The answer from the experiments is that Euclidean hidden states contain real
safety signal, but the best Euclidean detector remains weaker than the
hyperbolic detector.

### Euclidean Detector

For each denoising step `t`, cached mean-pooled hidden states are scored by
distance to the safe centroid:

```text
c_t = mean({x_i,t safe})
d_E(x, c_t) = ||x - c_t||_2
R_t = quantile_q({d_E(x_i,t, c_t) for safe train samples})
block if d_E(x, c_t) > R_t
```

For trajectory variants:

```text
block if any selected step fires
```

Unsafe samples are not used to fit the center or radius; they are used only for
evaluation.

### Euclidean Probe Setup

Dataset:

```text
saralazza/llada-safety-dataset
```

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

### Euclidean Experiment Log

| Folder | Experiment | Status |
| --- | --- | --- |
| `logs/exp_00_euclidean_minimal_svdd_nu001_cached_eval/` | Minimal Euclidean SVDD baseline, conservative `NU=0.01` | Complete |
| `logs/exp_01_threshold_sweep_quantiles/` | Radius quantile sweep | Complete |
| `logs/exp_02_multistep_or_rule/` | Multi-step OR trajectory guard | Complete |
| `logs/exp_03_step_subset_or_ablation/` | Step-subset OR ablation | Complete |
| `logs/exp_04_hf_heldout_eval/` | Held-out HF test evaluation | Complete |
| `logs/exp_05_mahalanobis_euclidean/` | PCA + Ledoit-Wolf Mahalanobis baseline | Complete |


## Layer Analysis (`layer_analysis/`)

The layer-analysis notes describe the conceptual HyperGuard design:

- Base model: `LLaDA-8B-Instruct`.
- Safe data: Alpaca rows from the safety dataset.
- Unsafe data: all non-Alpaca rows.
- Candidate probe steps include early/mid denoising steps such as `5`, `10`,
  and `15`.
- The safety detector monitors hidden-state trajectory geometry rather than
  keyword-level or output-level text.

The layer-wise probing command:

```bash
python probe_analysis.py --samples-per-class 50 --batch-size 1 --device auto
```

Expected artifacts under `probe_outputs/`:

```text
layer_separability_scores.json
layer_score_plot.png
best_layer_pca.png
```

The conceptual runtime intervention described in the layer-analysis README is:

```text
extract hidden state
  -> project into hyperbolic space
  -> compute distance from safety region
  -> stop or flag if threshold is exceeded
```

In the active root implementation, the stop/flag decision is evaluated offline
by `test_guard.py`; early interruption inside `generate.py` is not implemented.

## Multimodal GPO-V HyperGuard (`multimodal/`)

The `multimodal/` folder evaluates whether HyperGuard-style hyperbolic SVDD can
detect unsafe multimodal trajectories produced by GPO-V attacks on LLaDA-V.

The v1 target is:

```text
GSAI-ML/LLaDA-V
```

Model weights are available at:

```text
https://huggingface.co/GSAI-ML/LLaDA-V
```

The dataset builder creates the split needed for the experiment:

- Unsafe: AdvBench/JailbreakBench-style harmful prompts plus the GPO-V benign
  image.
- Safe: Alpaca/XSTest-safe benign prompts plus the same benign image.

During probing, unsafe rows receive a GPO-V visual perturbation. Safe rows keep
the image clean.

### Multimodal Setup

On a Linux SSH/GPU machine, start from the multimodal folder:

```bash
cd multimodal
bash scripts/download_gpo_v_upstream.sh
bash scripts/setup_ssh_env.sh external/GPO-V-0250/LLaDA-V
```

The downloader fetches the upstream anonymous GPO-V repo, places LLaDA-V code at
`external/GPO-V-0250/LLaDA-V`, verifies
`external/GPO-V-0250/LLaDA-V/llava/model/builder.py`, and writes
`.env.multimodal` so the run scripts can find GPO-V automatically.

If you want the model weights cached before running:

```bash
hf download GSAI-ML/LLaDA-V
```

Run a tiny activation-extraction smoke test:

```bash
bash scripts/run_multimodal_smoke.sh
```

With a local harmful prompt file:

```bash
bash scripts/run_multimodal_smoke.sh data/advbench.csv
```

To include a tiny GPO-V perturbation test:

```bash
RUN_ATTACK_SMOKE=1 ATTACK_STEPS=5 \
bash scripts/run_multimodal_smoke.sh data/advbench.csv
```

### Required External Pieces

- Upstream GPO-V `LLaDA-V` code, obtainable with
  `scripts/download_gpo_v_upstream.sh`.
- Access to `GSAI-ML/LLaDA-V` weights through Hugging Face or a model cache.
- A harmful prompt file for true AdvBench/JailbreakBench unsafe prompts.
- Optional Alpaca-style safe prompt file.
- CUDA GPU with enough VRAM for LLaDA-V and GPO-V optimization.

The repository already contains HyperGuard/SVDD code and a benign demo image at:

```text
gpo_v/assets/example_input_image.jpg
```

Recommended prompt sources:

```text
Unsafe: https://huggingface.co/datasets/walledai/AdvBench
Safe:   https://huggingface.co/datasets/iboero16/SAFE-ALPACA
```

Export local prompt files on the GPU machine:

```bash
mkdir -p data/hf_prompts
python - <<'PY'
from datasets import load_dataset

advbench = load_dataset("walledai/AdvBench", split="train")
advbench.to_csv("data/hf_prompts/advbench.csv", index=False)

safe_alpaca = load_dataset("iboero16/SAFE-ALPACA", split="safe_alpaca_100")
safe_alpaca.to_json("data/hf_prompts/safe_alpaca.jsonl", orient="records", lines=True)
PY
```

### Prepare Multimodal Data

Preferred setup with a local harmful prompt file:

```bash
python prepare_multimodal_gpo_dataset.py \
  --unsafe-file data/hf_prompts/advbench.csv \
  --safe-file data/hf_prompts/safe_alpaca.jsonl \
  --safe-field instruction \
  --unsafe-source-name advbench \
  --max-safe 200 \
  --max-unsafe 200 \
  --image-path gpo_v/assets/example_input_image.jpg \
  --output data/multimodal_gpo/llada_v_gpo_prompts.jsonl
```

### Run GPO-V and Extract Probes

For the full 400-sample attacked probe extraction run, use the helper:

```bash
SAFE_FILE=data/hf_prompts/safe_alpaca.jsonl \
SAFE_FIELD=instruction \
bash run_full_multimodal_probe_attack.sh data/hf_prompts/advbench.csv
```

This saves:

```text
probes_400_attack_all_layers_steps_1_5_10_15/probes.npz
probes_400_attack_all_layers_steps_1_5_10_15/metadata.json
probes_400_attack_all_layers_steps_1_5_10_15/checkpoint_probes.npz
probes_400_attack_all_layers_steps_1_5_10_15/checkpoint_metadata.json
probes_400_attack_all_layers_steps_1_5_10_15/checkpoint_state.json
results/multimodal_gpo_full_attack/verification_summary.json
```

For custom extraction from an already prepared JSONL file:

```bash
python run_gpo_v_lladav_probe.py \
  --gpo-v-root external/GPO-V-0250/LLaDA-V \
  --input-jsonl data/multimodal_gpo/llada_v_gpo_prompts.jsonl \
  --output-dir outputs/multimodal_gpo/probes \
  --probe-steps 1 5 10 15 \
  --all-layers \
  --attack-steps 5 \
  --save-every 25 \
  --resume \
  --write-final-checkpoint
```

### Train Multimodal SVDD

```powershell
python train_multimodal_svdd.py `
  --probe-npz outputs/multimodal_gpo/probes/probes.npz `
  --output-dir outputs/multimodal_gpo/svdd `
  --probe-steps 5 10 15 `
  --layer-ids 16 23 29
```

Metrics and checkpoints are saved under:

```text
outputs/multimodal_gpo/svdd
```

## Summary

This workspace supports a connected set of diffusion-LLM safety experiments:

1. HyperGuard learns a hyperbolic safety region over LLaDA hidden-state
   trajectories.
2. DiffuGuard studies intrinsic safety loss and repair during dLLM generation.
3. DIJA supplies a mask-text jailbreak attack and evaluation pipeline.
4. Euclidean baselines establish how much safety signal exists before
   hyperbolic projection.
5. Layer analysis guides which steps/layers are useful for probing.
6. Multimodal HyperGuard extends the detector idea to GPO-V attacks on LLaDA-V.

The strongest current root direction is the HyperGuard trajectory detector:
extract hidden states, project them into Lorentz space, and classify trajectory
deviation with SVDD.
