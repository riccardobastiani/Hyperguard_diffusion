# Multimodal GPO-V HyperGuard

This branch evaluates whether HyperGuard-style hyperbolic SVDD detection can
identify unsafe multimodal trajectories produced by GPO-V attacks on LLaDA-V.

The v1 target is `GSAI-ML/LLaDA-V`.

GPO-V provides the attack code, the target dVLM setup, and a benign demo image.
It does not appear to provide a complete safe/unsafe multimodal dataset ready
for this experiment. The dataset builder therefore creates the split we need:

- unsafe: AdvBench/JailBreakBench-style harmful prompts + the GPO-V benign image
- safe: Alpaca/XSTest-safe benign prompts + the same GPO-V benign image

During probing, unsafe rows receive a GPO-V visual perturbation. Safe rows keep
the image clean.

## Setup

Install the project dependencies and keep a local copy of the upstream GPO-V
`LLaDA-V` folder available. The probing script imports LLaDA-V helpers from that
folder through `--gpo-v-root`.

```powershell
pip install -r requirements.txt
```

On a Linux SSH/GPU machine, the setup can be run with:

```bash
bash scripts/setup_ssh_env.sh /path/to/GPO-V/LLaDA-V
```

Then run a tiny activation-extraction smoke test:

```bash
bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V
```

If you already have an AdvBench/JailBreakBench prompt file, pass it as the second
argument:

```bash
bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V data/advbench.csv
```

The smoke script defaults to `--skip-attack` so it checks model loading,
multimodal preprocessing, generation, and activation capture first. To also run
a tiny GPO-V perturbation test, set:

```bash
RUN_ATTACK_SMOKE=1 ATTACK_STEPS=5 bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V data/advbench.csv
```

## Required External Pieces

The repo already contains the HyperGuard/SVDD code and one benign GPO-V demo
image at `gpo_v/assets/example_input_image.jpg`. You still need:

- upstream GPO-V `LLaDA-V` code folder, passed through `--gpo-v-root`
- access to `GSAI-ML/LLaDA-V` weights through Hugging Face/model cache
- a harmful prompt file if you want true AdvBench/JailBreakBench unsafe prompts
- optionally, an XSTest-safe/Alpaca safe prompt file; otherwise the script uses
  Alpaca rows from `saralazza/llada-safety-dataset`
- a CUDA GPU with enough VRAM for LLaDA-V and GPO-V optimization

You do not need precomputed perturbations. The GPO-V loop computes the visual
perturbation during `run_gpo_v_lladav_probe.py` for rows with `label=1`.

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

For a smoke test, use `--max-samples 2 --skip-attack`. For the real unsafe
trajectory run, omit `--skip-attack`.

```powershell
python run_gpo_v_lladav_probe.py `
  --gpo-v-root C:\path\to\GPO-V\LLaDA-V `
  --input-jsonl data/multimodal_gpo/llada_v_gpo_prompts.jsonl `
  --output-dir outputs/multimodal_gpo/probes `
  --probe-steps 5 10 15 `
  --layer-ids 16 23 29
```

The script saves `probes.npz` and `metadata.json`.

## 3. Train Hyperbolic SVDD

```powershell
python train_multimodal_svdd.py `
  --probe-npz outputs/multimodal_gpo/probes/probes.npz `
  --output-dir outputs/multimodal_gpo/svdd `
  --probe-steps 5 10 15 `
  --layer-ids 16 23 29
```

Metrics and checkpoints are saved under `outputs/multimodal_gpo/svdd`.
