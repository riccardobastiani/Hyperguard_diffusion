# Full Multimodal GPO-V Probe Attack

Run from inside `mutimodal/`:

```bash
bash run_full_multimodal_probe_attack.sh /path/to/GPO-V/LLaDA-V/train /path/to/advbench.csv
```

The script prepares a 400-row multimodal dataset and then runs the full LLaDA-V
GPO-V probe extraction:

- 200 safe rows
- 200 unsafe AdvBench/JailbreakBench rows
- probe steps `1 5 10 15`
- all transformer layers, expected `0..31`
- unsafe GPO-V attack steps `5`
- safe rows use no attack

Default output stays inside `mutimodal/`:

```text
probes_400_attack_all_layers_steps_1_5_10_15/
  probes.npz
  metadata.json
  checkpoint_probes.npz
  checkpoint_metadata.json
  checkpoint_state.json

results/multimodal_gpo_full_attack/
  last_probe_command.txt
  verification_summary.json
```

Useful overrides:

```bash
VENV_DIR=/path/to/venv
SAFE_FILE=/path/to/safe_prompts.jsonl
SAFE_FIELD=prompt
UNSAFE_FIELD=goal
DEVICE_MAP=cuda:0
QUANTIZATION=4bit
OUTPUT_DIR=probes_400_attack_all_layers_steps_1_5_10_15_retry
RESUME=1
```

The final verification command is:

```bash
python -m scripts.verify_full_multimodal_probe_run \
  --checkpoint-dir probes_400_attack_all_layers_steps_1_5_10_15 \
  --expected-count 400 \
  --expected-safe-count 200 \
  --expected-unsafe-count 200 \
  --expected-unsafe-attack-steps 5 \
  --expected-hidden-dim 4096 \
  --probe-steps 1 5 10 15 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --report-json results/multimodal_gpo_full_attack/verification_summary.json
```
