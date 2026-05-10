# Hyperbolic Trajectory Detection for Diffusion-Based LLM Safety

This project explores a novel safety mechanism for diffusion-based language models (e.g., LLaDA-style architectures) by analyzing their internal generation dynamics in **hyperbolic space**. Instead of relying on surface-level input/output filtering, the system monitors the *geometry of hidden states* during generation to detect and interrupt adversarial behavior early in the denoising process.

---

## Key Idea

Diffusion language models generate text iteratively through a sequence of denoising steps. Recent findings suggest that **early steps (approximately steps 8–16)** are critical: perturbations during this phase can strongly influence the final output.

Instead of analyzing only text, this approach:
- Extracts **hidden states** from intermediate generation steps
- Projects them into a **hyperbolic (Lorentz) space**
- Learns the geometry of *safe trajectories*
- Detects deviations that indicate potential jailbreak or adversarial manipulation

In this space:
- Safe behaviors form a dense central manifold
- Attacks manifest as geometric outliers

---

## System Overview

### Phase 1 — Model & Dataset Setup
- Base model: **LLaDA-8B-Instruct**
- Hardware target: ~16GB VRAM
- Datasets:
  - Safe prompts (benign behavior)
  - Adversarial prompts (PAD / DIJA attacks from HarmBench)

---

### Phase 2 — Training the Safety Detector

The detector is trained using only **safe trajectories**:

1. Run the model on safe prompts
2. Extract hidden states at key denoising steps (e.g., 5, 10, 15)
3. Map representations into **hyperbolic space**
4. Train a **One-Class SVDD** model to learn the boundary of normal behavior

This creates a **geometric safety region** representing valid model dynamics.

---

### Phase 3 — Runtime Monitoring

During inference:

1. The model generates text step-by-step
2. At critical steps (5, 10, 15), hidden states are extracted
3. The system computes the **hyperbolic distance** from the safety region
4. Trajectory deviation is continuously evaluated

---

### Phase 4 — Intervention Mechanism

If the trajectory exceeds a learned safety threshold:

- Generation is **immediately stopped**
- The output is discarded or flagged
- The system prevents potential harmful completion

This acts as a **real-time safety brake on the model’s internal reasoning process**.

---

## Evaluation Metrics

The system is evaluated using:

- **F1 Score**  
  Measures classification accuracy between safe vs adversarial trajectories in early steps.

- **ASR (Attack Success Rate)**  
  Reduction in successful jailbreak or prompt injection attacks.

- **Optimal Stopping Step (T\*)**  
  Identifies the best trade-off between:
  - Early intervention (safety)
  - Overblocking (false positives)
  - Computational efficiency

---

## Conceptual Contribution

Unlike traditional safety systems that rely on:
- keyword filtering
- output moderation
- static classifiers

this approach operates on the **latent geometry of model cognition**.

It interprets generation as a **trajectory in hyperbolic space**, where:
> unsafe behavior corresponds to a deviation in the structure of thought itself, not just in language.

---

## Limitations & Considerations

- Requires access to internal model states
- Sensitive to model architecture changes
- Computational overhead due to geometric projections
- Threshold calibration is critical for avoiding false positives

---