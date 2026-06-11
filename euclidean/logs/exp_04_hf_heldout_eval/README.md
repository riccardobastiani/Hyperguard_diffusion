# exp_04: Held-out HF Test Evaluation

## Goal

Evaluate the best Euclidean baseline configurations on fresh held-out prompts from the Hugging Face test split. The Euclidean centers and radii come from the existing training/cache probes; held-out probes are used only for evaluation.

## Configurations

- `single_step_5_q098`: step 5, q=0.98
- `single_step_30_q098`: step 30, q=0.98
- `trajectory_5_30_q098`: steps 5 30, q=0.98
- `aggressive_1_5_10_q095`: steps 1 5 10, q=0.95

## Run

```bash
cd /media/hdd/huseynov/Hyperguard_diffusion
source .venv312/bin/activate

PYTHON=.venv312/bin/python \
DEVICE=cuda \
GPU_ID=0 \
SAMPLES_PER_CLASS=50 \
./run_euclidean_hf_heldout.sh
```

To rerun analysis from cached held-out probes:

```bash
PYTHON=.venv312/bin/python REUSE_CACHE=1 ./run_euclidean_hf_heldout.sh
```

## Outputs

- `hf_heldout.log`
- `hf_heldout_summary.csv`
- `hf_heldout_rates.png`
- `heldout_safe_features.npz`
- `heldout_unsafe_features.npz`
- `ANALYSIS.md`
