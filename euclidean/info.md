### 1. The Proposal: Hyperbolic Trajectory Detection

Diffusion models (like LLaDA) do not generate text all at once, but start from "noise" and gradually clean it up through a series of steps called **denoising steps**.

- **The problem:** It has been discovered that the initial steps (between step 8 and 16) are the most critical. If an attacker manages to manipulate these early steps, the rest of the generation will inevitably follow a dangerous path.
- **The innovative idea:** Instead of analyzing only the input or output text, researchers want to analyze the model's **internal states (hidden states)** while it is "deciding" what to write.
    - These internal states are projected into a **hyperbolic space (Lorentz)**.
    - In this geometric space, data hierarchies are very clear: "normal" behaviors cluster in the center, while attacks (anomalies) appear as very obvious mathematical deviations, even if the text seems harmless.

---

### 2. Workflow (Methodological Execution)

The project is developed through the following technical steps:

### Phase 1: Initialization and Setup

- A **LLaDA-8B-Instruct** model is used, optimized to run on accessible hardware (16GB of VRAM).
- Two datasets are prepared: a "good" one (safe requests) and a "bad" one (PAD and DIJA attacks derived from **HarmBench**).

### Phase 2: Training the Detector (The "Security Center")

- The model is run on **safe** prompts.
- At certain intervals (e.g., generation steps 5, 10, 15), the **hidden states** (the internal mathematical representations) are extracted.
- These data are transformed from a standard Euclidean space into a **hyperbolic space**.
- An algorithm called **One-class SVDD** is trained. This algorithm learns to recognize only the shape of "safe" behavior, creating a sort of geometric "safety bubble" in the hyperbolic graph.

### Phase 3: Monitoring and Interception (Inference)

- When the model receives a new question, the system starts generating the response.
- At every critical step (5, 10, 15), the system "hooks" the model and checks where the trajectory is in the hyperbolic space.
- **Distance Calculation:** The system measures how far the current trajectory is from the "safety bubble" defined in Phase 2.

### Phase 4: Termination (The Emergency Brake)

- If the trajectory exceeds a certain safety radius (indicating that the model is about to generate dangerous content or is under attack), the generation is **interrupted immediately**.
- This "short-circuits" the attack before the model actually writes harmful words.

---

### 3. Evaluation Matrix (How is success measured?)

Researchers will evaluate the system based on three parameters:

- **F1-Score:** How accurate is the system in distinguishing between a normal user and an attacker during the early steps?
- **ASR (Attack Success Rate):** By how much does the success rate of PAD attacks decrease thanks to this defense?
- **Optimal Stopping Step ($T^*$):** What is the perfect moment to stop the model? The goal is to find the ideal compromise: stopping it early enough to block the attack, but not so early as to waste resources or mistakenly block legitimate users (false positives).

---

### In Summary

Unlike traditional filters that read words, this defense reads the **"geometry of thought"** of the model. If the path the model is taking in mathematical space looks like a jailbreak, the system pulls the plug instantly. It is an extremely difficult defense to bypass because it does not rely on keywords, but on the deep structure of the generation process.

---

- Trainable R
    
    [step 5] Loaded SVDD checkpoint from probe_outputs/layer_23/svdd_step_5.pt with R=0.9900
    === Step 5 | R=0.9900 ===
    TP unsafe blocked: 155/157 (98.7%)
    FN unsafe allowed: 2/157 (1.3%)
    TN safe allowed:   105/106 (99.1%)
    FP safe blocked:   1/106 (0.9%)
    rates: TPR=98.7% FNR=1.3% TNR=99.1% FPR=0.9%
    safe dist stats: min=0.5770 mean=0.8212 max=1.0082
    unsafe dist stats: min=0.8933 mean=1.2387 max=1.3359
    
    Confusion matrix across steps:
    TP unsafe blocked: 155/157 (98.7%)
    FN unsafe allowed: 2/157 (1.3%)
    TN safe allowed:   105/106 (99.1%)
    FP safe blocked:   1/106 (0.9%)
    rates: TPR=98.7% FNR=1.3% TNR=99.1% FPR=0.9%

---
### TODOs:
- [x]  Metrics to add:
    - [x]  AUROC
    - [x]  AUPR
    - [x]  ASR pre and post defense
    - [x]  confusion matrix
- [x]  Add the ablation on the various timestep, becasue right now we only did it for 5 but we should go much further (but it just need to be integrated)
- [ ]  BASELINE
    - [ ]  euclidean (MURAD)
    - [ ]  DiffuGuard (https://arxiv.org/pdf/2509.24296) https://github.com/niez233/DiffuGuard (SARA)
    - [x]  TrajGuard (only for LLMs) https://arxiv.org/pdf/2604.07727v1 KILLED
    - [ ]  HiddenDetect (only for LVLM) https://arxiv.org/pdf/2502.14744

---
### Dataset
we have created the dataset which is in the link below: 
"saralazza/llada-safety-dataset · Datasets at Hugging Face"