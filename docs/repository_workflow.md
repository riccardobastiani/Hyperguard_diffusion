# Repository Workflow: Hyperguard_diffusion

## 1. Purpose of the repository
This repository implements an experimental framework to train and evaluate a safety detection mechanism for diffusion-based language models, specifically **LLaDA-8B-Instruct**. The detector is inspired by hyperbolic trajectory detection: it extracts intermediate hidden states during LLaDA's parallel generative denoising process, projects them onto a hyperbolic space (the Lorentz hyperboloid), and evaluates them using a soft-boundary One-Class Support Vector Data Description (SVDD) classifier.

The primary objective is to learn a geometric "safety region" around normal (safe) prompt trajectories and identify deviations caused by adversarial (unsafe/jailbreak) prompts. The repository supports extracting layer-wise activations, training the hyperbolic SVDD detector contrastively using both safe and unsafe samples to prevent hypersphere collapse, and evaluating detection rates and distance statistics either on cached features or through live Hugging Face dataset inference.

---

## 3. Repository structure
Below is a map of the critical files and directories in the repository, explaining their roles and key symbols:

| Path | Role in the workflow | Important symbols |
| :--- | :--- | :--- |
| `dataset.py` | Handles dataset downloading, shuffling, balancing, text extraction, and label validation. | `load_balanced_prompt_dataset`, `_safe_alpaca_prompt`, `_unsafe_prompt`, `validate_labels` |
| `generate.py` | Contains custom LLaDA generation logic, incorporating weight-tying compatibility patches and the intermediate activation extraction hook. | `generate_with_layer_probes`, `_capture_layer_probes`, `_transformer_layer_states`, `_mean_pool`, `install_transformers_tied_weights_compat` |
| `hyperbolic_projection.py` | Implements the neural network module that projects Euclidean features to coordinates satisfying the Lorentz manifold constraints. | `HyperbolicProjection` |
| `svdd.py` | Implements the Lorentz distance formula, SVDD center initialization, the SVDD module, the training loop, and checkpoint loading/saving. | `HyperbolicSVDD`, `lorentz_distance`, `init_center`, `train_svdd`, `save_checkpoint`, `load_checkpoint` |
| `probe_analysis.py` | CLI script used to load LLaDA, extract activations, and train SVDD classifiers at target denoising steps. | `main`, `extract_layer_features`, `_l2_normalize_features` |
| `test_guard.py` | CLI script used to evaluate trained SVDD detectors, report ROC-AUC / ASR metrics, and calibrate the radius threshold. | `main`, `load_llada_assets`, `extract_features_from_prompts`, `_compute_metrics`, `apply_transformers_attribute_patch` |
| `run_probe_steps.sh` | Bash orchestration script to sweep multiple denoising steps, execute training, and aggregate optimal steps. | `PROBE_STEPS`, `OUTPUT_ROOT` |

---

## 4. End-to-end workflow

### 4.1 Environment and configuration
*   **Dependencies:** Configured in `requirements.in` and compiled in `requirements.txt`. Key libraries include `torch` (deep learning engine), `transformers` (model hosting), `datasets` (data downloading), `scikit-learn` (metrics), `matplotlib` (plotting), `numpy` (numerical operations), and `geoopt` (differential geometry in PyTorch, used for Riemannian optimization and manifold operations).
*   **Config files:** No static configuration files (such as YAML or JSON configs) are utilized. All runtime options are configured via CLI arguments using Python's `argparse` in `probe_analysis.py` and `test_guard.py`.
*   **CLI Arguments:**
    *   `probe_analysis.py` parses parameters such as `--model-name` (default: `"GSAI-ML/LLaDA-8B-Instruct"`), `--probe-steps` (list of target denoising steps), `--layer-id` (layer to probe, default: `15`), `--proj-dim` (default: `128`), `--curvature` (manifold curvature $k$, default: `1.0`), `--nu` (SVDD boundary parameter, default: `0.1`), and `--l2-normalize-probes`.
    *   `test_guard.py` parses arguments like `--ckpt-dir`, `--probe-steps`, `--eval-all-safe`, `--eval-all-unsafe`, `--metrics`, `--calibrate-quantile`, and `--hf-test` (to trigger live Hugging Face inference).

### 4.2 Model initialization
*   **Model class:** Loaded via `AutoModel.from_pretrained(model_name, trust_remote_code=True)`.
*   **Tokenizer:** Loaded via `AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)`.
*   **Compatibility Patches:**
    *   `install_transformers_tied_weights_compat()` (in `generate.py`): Patches `PreTrainedModel.tie_weights` and backfills `all_tied_weights_keys` to allow LLaDA's custom weight-tying logic to load without crashing in newer Hugging Face `transformers` versions.
    *   `apply_transformers_attribute_patch()` (in `test_guard.py:L246-L269`): Wraps `transformers.modeling_utils.get_total_byte_count` to ensure `model.all_tied_weights_keys` is initialized to an empty dictionary `{}` if absent.
    *   `ensure_llada_config_compat()` (in `generate.py`): Sets `use_cache = False` and `use_return_dict = True` in the model configuration.
*   **Device & Precision Setup:**
    *   Supports loading on GPU (`cuda`) or CPU (`cpu`). In GPU mode, models are initialized in `bfloat16` (`model_dtype()` returns `torch.bfloat16`) and placed on a target device index using a fixed `device_map = {"": gpu_id}` to avoid VRAM spikes. In CPU mode, `float32` is used.

### 4.3 Data preparation
*   **Safe Dataset:** Drawn from `saralazza/llada-safety-dataset` where the row `source_dataset` equals `"alpaca"`, or loaded from a local JSON/CSV file specified by `--safe-local-path`. Prompts are extracted using fallback priorities: `refined_behavior` $\rightarrow$ `behavior` $\rightarrow$ `instruction` $\rightarrow$ `input`.
*   **Attack (Unsafe) Dataset:** Drawn from the same dataset where `source_dataset` does NOT equal `"alpaca"`. Prompts are extracted using: `refined_behavior` $\rightarrow$ `prompt` $\rightarrow$ `behavior` $\rightarrow$ `instruction` $\rightarrow$ `input`.
*   **Preprocessing and Balancing:**
    *   Prompt sets are balanced to contain exactly `samples_per_class` rows per group.
    *   The sets are combined and shuffled deterministically using a seeded `random.Random(args.seed)` instance.
    *   Tokenization applies the tokenizer's chat template (via `apply_chat_template`) to structure inputs as user messages, left-pads the inputs, and truncates them to `--max-prompt-length` (default: 512).

### 4.4 Feature or hidden-state extraction
*   **Captured States:** Hidden state activations from the transformer block outputs (excluding the embedding layer at index 0).
*   **Location:** Extracted from the layer specified by `--layer-id` at the global steps listed in `--probe-steps` (one-based).
*   **Interception Hook:**
    *   There are no PyTorch forward hooks registered. Instead, `generate_with_layer_probes` intercepts states directly. During the denoising loop in `generate.py:L352-L430`, if `global_step in probe_steps`, the model forward pass is executed with `output_hidden_states=True`.
    *   `_capture_layer_probes` extracts the hidden state tensor corresponding to the target layer ID, crops classifier-free guidance conditional slices, and applies mean-pooling over the sequence dimension (dim=1) using the attention mask:
        ```python
        mask = attention_mask.to(device=hidden_state.device).unsqueeze(-1).to(dtype=hidden_state.dtype)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (hidden_state * mask).sum(dim=1) / denom
        ```
    *   **Output Shape:** Mean-pooled features are detached, cast to `float32`, moved to the CPU, and return as a numpy array of shape:
        $$\text{Shape: } [\text{batch\_size}, 4096]$$

### 4.5 Geometry or representation transformation
*   **Manifold Projection:** Implemented in `hyperbolic_projection.py` $\rightarrow$ `HyperbolicProjection.forward`.
    1.  The Euclidean feature $x \in \mathbb{R}^{4096}$ is projected to a target dimensionality via a linear layer without bias:
        $$v = Wx \quad \text{where } v \in \mathbb{R}^{\text{proj\_dim}} \quad (\text{default proj\_dim} = 128)$$
    2.  To prevent weight decay collapse (where weights shrink to zero to collapse distances), the projected vector is L2-normalized and scaled to a fixed magnitude `max_norm` (default: 1.0):
        $$v' = \text{max\_norm} \cdot \frac{v}{\|v\|_2}$$
    3.  A zero coordinate is prepended to represent the time component in Minkowski space, satisfying the tangent-at-origin constraint:
        $$u = [0, v'_1, v'_2, \dots, v'_{\text{proj\_dim}}] \in \mathbb{R}^{\text{proj\_dim} + 1}$$
    4.  The vector $u$ is projected onto the Lorentz hyperboloid manifold $H^n \subset \mathbb{R}^{n+1}$ using the exponential map at the origin:
        $$\text{emb} = \text{expmap}_0(u) \in \mathbb{R}^{\text{proj\_dim} + 1}$$
*   **Alternative Space:** Optional per-sample L2-normalization of features before projection is supported via the `--l2-normalize-probes` CLI flag.

### 4.6 Detector training
*   **Model:** `HyperbolicSVDD` in `svdd.py`.
*   **Center Initialization (`init_center`):**
    *   The SVDD hypersphere center is computed at the start of training by taking the Euclidean mean of projected safe features, zeroing out the time coordinate, scaling the spatial components to `max_norm`, and mapping the resulting tangent vector onto the Lorentz hyperboloid via `expmap0`. The center is registered as a fixed buffer and remains frozen.
*   **Objective Function:**
    *   The module optimizes the soft-boundary SVDD loss. To prevent the hypersphere from collapsing to a single point, the repository implements a **contrastive SVDD loss** in `svdd.py:L161-L191` that uses both safe and unsafe samples:
        $$\mathcal{L} = R^2 + \frac{1}{\nu} \mathbb{E}_{x \in \text{safe}} \left[ \max\left(0, d^2_{\text{Lorentz}}(z_x, c) - R^2\right) \right] + w_{\text{unsafe}} \mathbb{E}_{y \in \text{unsafe}} \left[ \max\left(0, R^2 - d^2_{\text{Lorentz}}(z_y, c)\right) \right]$$
*   **Fitting Procedure:**
    *   The radius $R$ is parameterized in tangent space as `self._log_R = nn.Parameter(raw_radius)` to guarantee positivity via softplus: $R = \text{softplus}(\text{\_log\_R}) + \text{radius\_eps}$.
    *   Trainable parameters (`svdd.projector.parameters()` and `svdd._log_R`) are optimized using the Adam optimizer for `--svdd-epochs` (default: 50) at learning rate `--svdd-lr` (default: 1e-3).
*   **Stored Artifacts:** Saved as `svdd_step_{step}.pt` containing the projector state dictionary, center, radius parameters, and hyperparameter config.

### 4.7 Inference-time monitoring
*   During evaluation, prompts are run through LLaDA using `generate_with_layer_probes`.
*   Mean-pooled features are extracted at the target step (e.g., step 5) and projected onto the Lorentz hyperboloid.
*   The Lorentz distance to the SVDD center is computed:
    $$d = d_{\text{Lorentz}}(\text{emb}, c)$$
*   **Classification decision:** The classifier checks if the distance exceeds the hypersphere radius $R$:
    $$\text{is\_unsafe} = d > R$$

### 4.8 Stopping or intervention logic
*   **Thresholding:** The classification boundary is defined by the trained/calibrated radius $R$.
*   **Early Denoising Interruption:** **Not implemented in the active LLaDA generation loop.**
    *   *Key Deviation:* The LLaDA generation loop in `generate_with_layer_probes` (in `generate.py`) does not contain any code to check SVDD decisions or break out of generation early. It always runs for the full number of block and step counts.
    *   *Real Implementation:* SVDD classification and radius checks are performed **offline** in `test_guard.py` on pre-extracted or post-inference activations for reporting block rates, TPR, and FPR, rather than interrupting active text generation.

### 4.9 Evaluation pipeline
*   **Metrics:** Evaluated in `test_guard.py:L222-L245` using:
    *   **AUROC / AUPR:** Computed using custom trapezoidal integration functions `_roc_auc` and `_pr_auc` on Lorentz distance scores and safe/unsafe labels.
    *   **ASR post-defense:** Attack Success Rate calculated as $100\% \times \frac{\text{False Negatives}}{\text{Total Unsafe}}$.
    *   **Confusion Matrix:** Reports True Positives (TP, unsafe blocked), False Positives (FP, safe blocked), True Negatives (TN, safe allowed), and False Negatives (FN, unsafe allowed).
*   **Quantile Calibration:**
    *   If `--calibrate-quantile` (e.g., 0.95) is passed, `test_guard.py` computes the corresponding quantile of safe distances and updates the SVDD radius.
    *   **Code Bug:** There is a bug in `test_guard.py:L506` which executes `svdd._R.fill_(new_R)`. Because `HyperbolicSVDD` does not define `_R` as a parameter or buffer, this causes an `AttributeError` at runtime. The correct method to update the radius is `svdd.set_radius(new_R)`.

---

## 5. File-by-file implementation trace

### `svdd.py`
*   **Role:** Implements the core mathematical structures of the hyperbolic detector.
*   **Functions:**
    *   `lorentz_distance` (L24): Computes the distance between coordinates on the Lorentz manifold.
    *   `init_center` (L43): Approximates the Frechet mean of safe embeddings in Minkowski space to initialize the center.
    *   `HyperbolicSVDD` (L96): Model module containing the projection hook, the contrastive SVDD `loss` method, and the `predict` / `classify` methods.
    *   `train_svdd` (L224): Optimizes the projection weights and log radius parameters.

### `hyperbolic_projection.py`
*   **Role:** Defines the neural network layer projecting Euclidean vectors into hyperbolic space.
*   **Classes/Methods:**
    *   `HyperbolicProjection` (L7): Initializes the underlying Linear layer, Lorentz manifold instance, and projection norm.
    *   `forward` (L34): Details the mapping flow: Linear projection $\rightarrow$ L2-normalization/scaling $\rightarrow$ prepending time-coordinate zero $\rightarrow$ `manifold.expmap0`.

### `test_guard.py`
*   **Role:** Evaluation script to score detectors offline and report statistical performance.
*   **Functions:**
    *   `load_llada_assets` (L270): Configures tokenizer padding and resolves weight-tying patches.
    *   `extract_features_from_prompts` (L308): Collects and optionally L2-normalizes prompt features batch-wise.
    *   `_compute_metrics` (L222): Calculates AUROC, AUPR, and post-defense ASR.

### `probe_analysis.py`
*   **Role:** Training entry point to extract features and fit SVDD models.
*   **Functions:**
    *   `extract_layer_features` (L97): Iterates over batches, formatting prompts and extracting intermediate layer hidden states.
    *   `main` (L175): Coordinates prompt ingestion, feature cache loading/creation, center initialization, detector training, and checkpoint serialization.

---

## 6. Actual data flow
The real data flow implemented in the repository operates as follows:

```
[Input Prompt]
      │
      ▼
[Instruct Template & Left-Padding]
      │
      ▼
[LLaDA Probing Inference (generate_with_layer_probes)]
      │
      ▼ (Run model forward with output_hidden_states=True at target probe_step)
[Extract Layer Hidden States (Shape: [batch, seq_len, 4096])]
      │
      ▼ (Mean pooling using attention mask)
[Euclidean Feature Representation (Shape: [batch, 4096])]
      │
      ▼ (Optional L2 Normalization: x / (||x|| + 1e-6))
[Normalized Euclidean Feature (Shape: [batch, 4096])]
      │
      ▼ (Linear projection and rescaling: Wx / ||Wx|| * max_norm)
[Scaled Latent Feature (Shape: [batch, proj_dim])]
      │
      ▼ (Prepend zero time component)
[Tangent Vector at Origin (Shape: [batch, proj_dim + 1])]
      │
      ▼ (Lorentz manifold exponential map: expmap0)
[Hyperbolic Embedding (Shape: [batch, proj_dim + 1])]
      │
      ▼
[SVDD Classifier (svdd.predict / svdd.classify)]
      │
      ├─► Training: Minimize contrastive SVDD loss and optimize projection & radius parameters.
      │
      └─► Evaluation: Compute Lorentz distance to center. If dist > R, return STOP (True); else CONTINUE (False).
```

---

## 7. Actual control flow
The interactions between the scripts and modules run in the following sequences:

### Training Flow
```
probe_analysis.py (main)
  │
  ├──► dataset.py (load_balanced_prompt_dataset)
  │     └─► Parses Hugging Face / local dataset and returns balanced lists
  │
  ├──► generate.py (generate_with_layer_probes)
  │     └─► Executes LLaDA forward pass and extracts mean-pooled activations
  │
  ├──► svdd.py (init_center)
  │     └─► Projects safe features and maps Euclidean mean onto the manifold
  │
  ├──► svdd.py (train_svdd)
  │     └─► Runs training epochs optimizing weights and log radius parameters
  │
  └──► svdd.py (save_checkpoint)
        └─► Serializes the HyperbolicSVDD model state to disk
```

### Evaluation Flow (HF Live Mode)
```
test_guard.py (main)
  │
  ├──► test_guard.py (load_hf_test_prompts)
  │     └─► Shuffles and filters dataset split from HF
  │
  ├──► test_guard.py (load_llada_assets)
  │     └─► Applies tied weights & byte-count attribute patches, loads LLaDA model
  │
  ├──► test_guard.py (extract_features_from_prompts)
  │     └─► Calls generate.py (generate_with_layer_probes) and L2-normalizes features
  │
  ├──► svdd.py (load_checkpoint)
  │     └─► Restores HyperbolicSVDD checkpoints
  │
  └──► svdd.py (predict / classify)
        └─► Computes Lorentz distances to evaluate AUROC, AUPR, and ASR
```

---

## 9. How to run the repository

### 1. Installation and Setup
Install the necessary package requirements compiled for the project:
```bash
pip install -r requirements.txt
```

### 2. Training the Hyperbolic SVDD Detector
Train the detector for LLaDA layer 23 at denoising step 5 using safe prompts from a local DIJA file:
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
*Note:* This generates `safe_probes.npz`, `unsafe_probes.npz`, and the trained SVDD checkpoint `svdd_step_5.pt` inside the directory `probe_outputs/layer_23/`.

### 3. Evaluating the Detector (Cache Mode)
Evaluate the performance (AUROC, AUPR, ASR) using the pre-extracted probe activations:
```bash
python test_guard.py \
  --ckpt-dir probe_outputs/layer_23 \
  --probe-steps 5 \
  --eval-all-safe \
  --eval-all-unsafe \
  --metrics \
  --device cuda
```

### 4. Evaluating the Detector (HF Live Inference Mode)
Extract fresh features on-the-fly from the Hugging Face test split and evaluate the classifier:
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

### 5. Running Calibration
Quantile-based radius calibration commands are supported in `test_guard.py` but incomplete/buggy due to the `svdd._R.fill_` attribute bug. Running the following command will crash on `AttributeError` unless the file is edited to use `svdd.set_radius(new_R)`:
```bash
python test_guard.py \
  --ckpt-dir probe_outputs/layer_23 \
  --probe-steps 5 \
  --eval-all-safe \
  --eval-all-unsafe \
  --calibrate-quantile 0.95 \
  --overwrite-checkpoints
```

---

## 11. Concise summary
The repository provides a complete implementation of the **Lorentz projection** and **Hyperbolic SVDD** modules, utilizing `geoopt` manifolds to construct a geometric classification space. 

However, it diverges from the original conceptual design in two major ways:
1.  **No Early Generation Interruption:** The model generation code in `generate.py` does not contain any stopping hooks or early exit branches based on the detector's classification. The denoising process always runs to completion (e.g., 64 steps), and the safety detector is only run post-hoc or offline in `test_guard.py` to evaluate the classification rate of the steps.
2.  **Contrastive Repulsion:** Traditional SVDD is trained in a one-class setting using only normal (safe) data. The training loop in `svdd.py` modifies this by incorporating a contrastive repulsion term that actively pushes unsafe samples outside the radius during optimization to prevent manifold representation collapse.
3.  **Calibration Bug:** The evaluation script `test_guard.py` contains a critical bug (`svdd._R.fill_` on line 506) that prevents quantile-based threshold calibration from executing successfully without causing an attribute error.
