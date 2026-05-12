# LLaDA Hyperbolic Trajectory Monitor

## Phase 1: LLaDA generation analysis

The official `generate.py` performs masked diffusion generation over a tensor `x` with shape `[batch, prompt_len + gen_length]`. Prompt tokens are copied into the prefix and every generated position is initialized to `mask_id=126336`.

For `saralazza/llada-safety-dataset`, prompts are read from the `refined_behavior` column. Safe rows are those with `resource_dataset == "alpaca"`; unsafe rows are all other `resource_dataset` values.

The generation loop has two levels:

1. `num_block in range(num_blocks)`: semi-autoregressive blocks, where `num_blocks = gen_length // block_length`.
2. `i in range(steps_per_block)`: denoising/unmasking iterations inside the current block.

The user-visible `steps` argument is divided by `num_blocks`, so the real global forward index is:

```python
global_step = num_block * steps_per_block + i + 1
```

At every denoising step:

1. `mask_index = (x == mask_id)` identifies still-masked tokens.
2. The model predicts logits for every position: `logits = model(x, attention_mask).logits`, shape `[batch, seq_len, vocab]`.
3. Optional classifier-free guidance doubles the batch as `[conditioned, unconditioned]`, then combines logits.
4. `x0 = argmax(add_gumbel_noise(logits))`, shape `[batch, seq_len]`.
5. Confidence is either model probability of `x0` or random.
6. Tokens outside the current block are blocked by assigning confidence `-inf`.
7. `topk` selects a fixed number of masked positions to reveal this step.
8. `x[transfer_index] = x0[transfer_index]` unmask-fills selected positions.

Hidden states can be intercepted inside transformer blocks during step 2, because the model forward is the only place where the current noisy sequence is encoded. Early stopping should be checked immediately after that forward pass and before token transfer, so the system can terminate on the anomalous trajectory state without committing another denoising update.

Best candidate layers are configurable. For a 32-layer 8B decoder, start with mid/late blocks such as `[15, 23, 31]`: middle layers often encode prompt/intent abstractions, late layers are closer to token decisions, and combining them makes the detector less brittle than a single layer.

## Code structure

```text
generate.py
  generate(..., monitor=None)
    monitor.on_step_begin(...)
    logits = model(...)
    monitor.on_step_observed(...)
    denoise/unmask selected tokens

project/
  hooks/
    hidden_state_hooks.py       # layer resolution, forward hooks, mean pooling
  geometry/
    lorentz.py                  # 4096->128 projection, Lorentz expmap, distance
  svdd/
    deep_svdd.py                # soft-boundary hyperbolic Deep SVDD
  monitoring/
    extraction.py               # offline hidden trajectory collection
    runtime.py                  # online score aggregation and early stopping
  evaluation/
    metrics.py                  # F1, FPR, ASR reduction, latency, T*
  scripts/
    train_svdd.py
    run_monitored_generation.py
    evaluate_monitor.py
  configs/
    default_monitoring.json
  tests/
```

## Phase 2: hidden-state extraction

Forward hooks are registered on selected transformer blocks. Each hook extracts the first tensor from the module output. For HuggingFace decoder blocks this is normally `[batch, seq_len, hidden_dim]`.

The runtime capture stores pooled tensors by default:

```text
hidden: [batch, seq_len, 4096]
attention_mask: [batch, seq_len]
pooled = sum(hidden * mask) / sum(mask)
pooled: [batch, 4096]
```

This is intentionally memory conservative. For `batch=1`, `hidden_dim=4096`, `float32`, one pooled layer-step is about 16 KB. Retaining the full sequence at `seq_len=256` would be about 4 MB per layer-step.

Hooks are removable through `HiddenStateCapture.remove()`, and both runtime and extraction monitors remove handles when requested. For quantized inference, hooks observe dequantized activation tensors emitted by layers; the detector can remain in fp32 while the base model runs bf16/int8.

## Phase 3: hyperbolic projection

Dimensionality reduction is required because raw 4096-dimensional hidden states are expensive and noisy for real-time scoring. The projector computes:

```text
h: [batch, 4096]
v = Linear(h): [batch, 128]
z = exp_0(v): [batch, 129]
```

The Lorentz model represents a point `z=(z0, z1, ..., zd)` on:

```text
-z0^2 + ||z_spatial||^2 = -k, z0 > 0
```

The exponential map at the origin is:

```text
z0 = sqrt(k) cosh(||v|| / sqrt(k))
z_spatial = sqrt(k) sinh(||v|| / sqrt(k)) v / ||v||
```

Distances are:

```text
d_L(x, y) = sqrt(k) arcosh(-<x,y>_L / k)
```

Lorentz is preferable to Poincare for this prototype because distances near the boundary are numerically better behaved in the hyperboloid coordinates. The Poincare ball is easier to visualize, but boundary saturation can be awkward under mixed precision. The implementation performs Lorentz math in fp32 and returns the caller dtype, with norm clipping to avoid `cosh/sinh` overflow.

## Phase 4: one-class SVDD

The detector learns a soft safe region around a manifold center `c`:

```text
loss = R^2 + (1 / nu) mean(max(0, d_L(z, c)^2 - R^2))
```

The center is initialized from safe embeddings, projected back onto the Lorentz sheet. The projection layer uses no bias and no bounded activation, following Deep SVDD collapse-prevention practice. Radius is learned through `softplus(raw_radius)` so it stays positive.

Collapse risks:

- Training only the projector and center jointly can collapse embeddings; this implementation fixes center as a buffer after initialization.
- Bounded activations can force features into a compact artificial range; none are used.
- Bias terms can map all inputs to the center; the projection defaults to `bias=False`.

Threshold calibration should be done on held-out safe and attack trajectories. Start with a high safe quantile, for example the 95th or 99th percentile of weighted safe scores, then report attack recall and false positive rate.

## Phase 5: real-time early stopping

The runtime monitor computes a per-step layer-reduced score, then aggregates:

```text
score = w5 * d5 + w10 * d10 + w15 * d15
```

If `score > threshold`, `generate` returns immediately with the current partially denoised `x`. This hard stop is low latency and simple to audit. A softer policy could continue generation but mask/refuse output downstream; that is easier for UX but weaker as a containment mechanism.

Latency overhead is mostly:

- hook callback dispatch for selected layers,
- mean pooling,
- one small linear projection per captured layer-step,
- Lorentz distance against one center.

The expensive base forward is unchanged.

## Phase 6: evaluation

Implemented metrics:

- F1, precision, recall, accuracy
- False Positive Rate on safe prompts
- ASR reduction from baseline and defended ASR
- Latency overhead
- Optimal stopping step `T*`

Suggested ablations:

- layer sets: early vs mid vs late vs multi-layer
- target steps: `{5}`, `{10}`, `{15}`, `{5,10,15}`, later schedules
- projection dimension: 32, 64, 128, 256
- Lorentz vs Poincare
- score aggregation: max, mean, weighted sum, learned logistic calibration
- CFG on/off and block length changes

## Debugging checklist

- Enable `HookConfig(log_every_capture=True)` to verify layer names, shapes, and monitored steps.
- Validate `LorentzManifoldOps.validate(z)` after projection; the constraint error should be small and all values finite.
- If scores are NaN, reduce `max_tangent_norm`, disable mixed precision around the detector, and inspect hidden-state norms.
- If no records are captured, run `dict(model.named_modules()).keys()` and pass exact layer names instead of integer indices.
- Profile with `torch.profiler`, Nsight Systems, or PyTorch CUDA memory summaries. The first bottleneck to check is accidental full-sequence retention via `keep_sequence=True`.
