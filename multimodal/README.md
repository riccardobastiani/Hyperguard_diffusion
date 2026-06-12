# Multimodal GPO-V HyperGuard

This branch evaluates whether HyperGuard-style hyperbolic SVDD detection can
identify unsafe multimodal trajectories produced by GPO-V attacks on LLaDA-V.

The v1 target is `GSAI-ML/LLaDA-V`.
Model weights are available on Hugging Face at
<https://huggingface.co/GSAI-ML/LLaDA-V>.

GPO-V provides the attack code, the target dVLM setup, and a benign demo image.
It does not appear to provide a complete safe/unsafe multimodal dataset ready
for this experiment. The dataset builder therefore creates the split we need:

- unsafe: AdvBench/JailBreakBench-style harmful prompts + the GPO-V benign image
- safe: Alpaca/XSTest-safe benign prompts + the same GPO-V benign image

During probing, unsafe rows receive a GPO-V visual perturbation. Safe rows keep
the image clean.

## Setup

On a Linux SSH/GPU machine, start from a clean clone with:

```bash
cd multimodal
bash scripts/download_gpo_v_upstream.sh
bash scripts/setup_ssh_env.sh external/GPO-V-0250/LLaDA-V
```

The downloader places the upstream code at
`external/GPO-V-0250/LLaDA-V` and verifies that
`external/GPO-V-0250/LLaDA-V/llava/model/builder.py` exists. It also writes
`.env.multimodal`, so the smoke and full-run scripts can find GPO-V without
passing `--gpo-v-root` each time.

The model weights are loaded from Hugging Face as `GSAI-ML/LLaDA-V`. If the GPU
machine should run offline after setup, pre-download them into the Hugging Face
cache:

```bash
hf download GSAI-ML/LLaDA-V
```

Run a tiny activation-extraction smoke test:

```bash
bash scripts/run_multimodal_smoke.sh
```

With a local AdvBench/JailBreakBench prompt file:

```bash
bash scripts/run_multimodal_smoke.sh data/advbench.csv
```

If you keep GPO-V elsewhere, pass it explicitly:

```bash
bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V data/advbench.csv
```

The smoke script defaults to `--skip-attack` so it checks model loading,
multimodal preprocessing, generation, and activation capture first. To also run
a tiny GPO-V perturbation test, set:

```bash
RUN_ATTACK_SMOKE=1 ATTACK_STEPS=5 bash scripts/run_multimodal_smoke.sh data/advbench.csv
```

## Required External Pieces

The repo already contains the HyperGuard/SVDD code and one benign GPO-V demo
image at `gpo_v/assets/example_input_image.jpg`. You still need:

- upstream GPO-V `LLaDA-V` code folder, passed through `--gpo-v-root`
- access to `GSAI-ML/LLaDA-V` weights through Hugging Face/model cache:
  <https://huggingface.co/GSAI-ML/LLaDA-V>
- a harmful prompt file if you want true AdvBench/JailBreakBench unsafe prompts
- optionally, an Alpaca-style safe prompt file; otherwise the script uses
  Alpaca rows from `saralazza/llada-safety-dataset`
- a CUDA GPU with enough VRAM for LLaDA-V and GPO-V optimization

You do not need precomputed perturbations. The GPO-V loop computes the visual
perturbation during `run_gpo_v_lladav_probe.py` for rows with `label=1`.

## Prompt Sources

Recommended Hugging Face sources:

- unsafe prompts: `walledai/AdvBench`
  <https://huggingface.co/datasets/walledai/AdvBench>
- safe Alpaca prompts: `iboero16/SAFE-ALPACA`
  <https://huggingface.co/datasets/iboero16/SAFE-ALPACA>

`walledai/AdvBench` may require accepting the dataset conditions on Hugging Face
before download. After logging in on the machine, export local prompt files:

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

The dataset builder auto-detects common prompt fields such as `goal`, `prompt`,
`instruction`, and `question`. If a downloaded file uses an unusual field name,
set `UNSAFE_FIELD=...` or `SAFE_FIELD=...` when launching the run.

## 1. Prepare Data

Preferred GPO-V-style setup, using a local harmful prompt file:

```powershell
python prepare_multimodal_gpo_dataset.py `
  --unsafe-file data/advbench.csv `
  --unsafe-field goal `
  --unsafe-source-name advbench `
  --max-safe 50 `
  --max-unsafe 50 `
  --image-path gpo_v/assets/example_input_image.jpg `
  --output data/multimodal_gpo/llada_v_gpo_prompts.jsonl
```

The script accepts `.csv`, `.tsv`, `.json`, and `.jsonl` files, including JSON
lists of prompt strings. If `--unsafe-field` or `--safe-field` is omitted, it
tries common names such as `goal`, `behavior`, `prompt`, `instruction`, and
`question`.

Fallback setup, using `saralazza/llada-safety-dataset` for both sides:

```powershell
python prepare_multimodal_gpo_dataset.py `
  --max-safe 50 `
  --max-unsafe 50 `
  --image-path gpo_v/assets/example_input_image.jpg `
  --output data/multimodal_gpo/llada_v_gpo_prompts.jsonl
```

## 2. Run GPO-V And Extract Probes

For the full 400-sample attacked probe extraction run, use the self-contained
runner:

```bash
bash run_full_multimodal_probe_attack.sh data/advbench.csv
```

With the Hugging Face prompt files above:

```bash
SAFE_FILE=data/hf_prompts/safe_alpaca.jsonl \
SAFE_FIELD=instruction \
bash run_full_multimodal_probe_attack.sh data/hf_prompts/advbench.csv
```

This prepares 200 safe rows and 200 unsafe rows, runs unsafe samples with
`--attack-steps 5`, captures probe steps `1 5 10 15` for all layers, writes
final `checkpoint_*` artifacts, and verifies the result. Outputs stay inside
this folder by default:

```text
probes_400_attack_all_layers_steps_1_5_10_15/
results/multimodal_gpo_full_attack/
```

The main extracted-probe artifacts are:

```text
probes_400_attack_all_layers_steps_1_5_10_15/
  probes.npz
  metadata.json
  checkpoint_probes.npz
  checkpoint_metadata.json
  checkpoint_state.json
```

To verify a completed extraction later:

```bash
python -m scripts.verify_full_multimodal_probe_run \
  --checkpoint-dir probes_400_attack_all_layers_steps_1_5_10_15 \
  --expected-count 400 \
  --expected-safe-count 200 \
  --expected-unsafe-count 200 \
  --expected-unsafe-attack-steps 5 \
  --expected-hidden-dim 4096 \
  --probe-steps 1 5 10 15 \
  --layers $(seq 0 31)
```

For custom probe extraction from an already prepared JSONL file, call the probe
runner directly. Use `--skip-attack` for a pure extraction smoke test; omit it
for actual unsafe GPO-V perturbations.

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

The direct runner saves `probes.npz` and `metadata.json`; with
`--write-final-checkpoint`, it also leaves the checkpoint-compatible files used
by the verifier.

## 3. Train Hyperbolic SVDD

```powershell
python train_multimodal_svdd.py `
  --probe-npz outputs/multimodal_gpo/probes/probes.npz `
  --output-dir outputs/multimodal_gpo/svdd `
  --probe-steps 5 10 15 `
  --layer-ids 16 23 29
```

Metrics and checkpoints are saved under `outputs/multimodal_gpo/svdd`.
