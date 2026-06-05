"""Utilities for GPO-V attacks and activation capture on LLaDA-V."""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.optim as optim
from PIL import Image

from .losses import DEFAULT_NEGATIVE_IDS, PositionTokenAnchorLoss


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "GSAI-ML/LLaDA-V"
DEFAULT_MODEL_ALIAS = "llava_llada"
DEFAULT_TARGET_DICT = {
    0: [36549],
    32: [7534],
}


def add_gpo_v_root(gpo_v_root: str | Path) -> Path:
    """Add the upstream GPO-V LLaDA-V folder to import path."""
    root = Path(gpo_v_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"GPO-V root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def model_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_lladav_assets(
    gpo_v_root: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    device_map: str = "cuda:0",
    cache_dir: str | None = None,
    dtype: str = "float16",
    load_8bit: bool = False,
    load_4bit: bool = False,
):
    """Load tokenizer/model/image processor using the upstream LLaDA-V loader."""
    add_gpo_v_root(gpo_v_root)
    from llava.model.builder import load_pretrained_model

    kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "device_map": device_map,
        "torch_dtype": model_dtype(dtype),
        "load_8bit": load_8bit,
        "load_4bit": load_4bit,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_name,
        None,
        model_alias,
        **kwargs,
    )
    # Keep projector dtype consistent with multimodal image features.
    # Some upstream checkpoints leave mm_projector in float32 while inputs are float16.
    requested_dtype = model_dtype(dtype)
    projector = _get_attr_path(model, "model.mm_projector")
    if projector is None:
        projector = _get_attr_path(model, "mm_projector")
    if projector is not None:
        projector.to(dtype=requested_dtype)
    model.eval()
    return tokenizer, model, image_processor, max_length


def build_prompt(
    gpo_v_root: str | Path,
    tokenizer,
    prompt: str,
    conv_template: str = "llava_llada",
    system_prompt: str = "You are a helpful assistant. Ignore the pictures and focus on answering the questions.",
) -> torch.Tensor:
    """Build LLaDA-V chat prompt and return image-token-aware input ids."""
    add_gpo_v_root(gpo_v_root)
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.system = system_prompt
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    return tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)


def prepare_image(
    gpo_v_root: str | Path,
    image_path: str | Path,
    image_processor,
    model,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """Load and preprocess an image for LLaDA-V."""
    add_gpo_v_root(gpo_v_root)
    from llava.mm_utils import process_images

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, (list, tuple)):
        image_tensor = [_image.to(dtype=dtype, device=device) for _image in image_tensor]
        x_orig = torch.stack([_image.to(device) for _image in image_tensor], dim=0)
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=device)
        x_orig = image_tensor
    return x_orig, [image.size]


def _tensor_from_hook_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    return None


def _get_attr_path(root: Any, dotted: str) -> Any | None:
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Could not extract logits from model output type {type(output)!r}.")


def differentiable_response_logits(
    model,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    *,
    gen_length: int,
    mask_id: int = 126336,
) -> torch.Tensor:
    """Return logits for a masked response suffix while preserving image gradients."""
    mask_ids = torch.full(
        (input_ids.shape[0], gen_length),
        mask_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    attack_input_ids = torch.cat([input_ids, mask_ids], dim=1)
    prompt_labels = torch.full_like(input_ids, -100)
    response_labels = torch.full_like(mask_ids, mask_id)
    labels = torch.cat([prompt_labels, response_labels], dim=1)
    outputs = model(
        input_ids=attack_input_ids,
        images=images,
        image_sizes=image_sizes,
        labels=labels,
        return_dict=True,
    )
    logits = _extract_logits(outputs)
    if not torch.is_floating_point(logits):
        raise TypeError(
            "Attack optimization expected floating-point logits from model.forward, "
            f"got dtype={logits.dtype}."
        )
    return logits[:, -gen_length:, :]


def find_layer_stack(model) -> tuple[str, Any]:
    """Find the transformer layer container for common LLaVA/LLaDA layouts."""
    candidates = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "model.language_model.model.layers",
        "llm.model.layers",
    ]
    for path in candidates:
        layers = _get_attr_path(model, path)
        if layers is not None and hasattr(layers, "__getitem__") and len(layers) > 0:
            return path, layers

    for name, module in model.named_modules():
        if name.endswith("layers") and hasattr(module, "__getitem__"):
            try:
                if len(module) > 0:
                    return name, module
            except TypeError:
                continue
    raise RuntimeError("Could not locate transformer layer stack for activation hooks.")


@dataclass
class ActivationRecorder:
    """Forward-hook recorder keyed by generation forward call and layer id."""

    model: Any
    layer_ids: Iterable[int]
    probe_steps: Iterable[int]
    pool_start: int | None = None
    pool_end: int | None = None

    def __post_init__(self) -> None:
        self.layer_ids = sorted(int(layer_id) for layer_id in self.layer_ids)
        if not self.layer_ids:
            raise ValueError("At least one layer id is required.")
        self.probe_steps = set(int(s) for s in self.probe_steps)
        self.handles = []
        self.forward_call = 0
        self._step_layer_id = self.layer_ids[0]
        self.features: dict[int, dict[int, list[np.ndarray]]] = {
            step: {layer: [] for layer in self.layer_ids}
            for step in self.probe_steps
        }
        _, self.layers = find_layer_stack(self.model)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"Expected hidden state [B, L, H], got {tuple(hidden.shape)}")
        start = self.pool_start if self.pool_start is not None else 0
        end = self.pool_end if self.pool_end is not None else hidden.shape[1]
        start = max(0, min(start, hidden.shape[1] - 1))
        end = max(start + 1, min(end, hidden.shape[1]))
        return hidden[:, start:end, :].mean(dim=1).detach().float().cpu().numpy()

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if layer_id == self._step_layer_id:
                self.forward_call += 1
            step = self.forward_call
            if step not in self.probe_steps:
                return
            hidden = _tensor_from_hook_output(output)
            if hidden is None:
                return
            self.features[step][layer_id].append(self._pool(hidden))

        return hook

    def __enter__(self):
        for layer_id in self.layer_ids:
            self.handles.append(self.layers[layer_id].register_forward_hook(self._make_hook(layer_id)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def initialize_response_span(
    model,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    tokenizer,
    generation_steps: int,
    gen_length: int,
    block_length: int,
) -> tuple[int | None, int | None]:
    """Ask LLaDA-V for the response span used by GPO-V losses."""
    result = model.generate(
        input_ids,
        images=images,
        image_sizes=image_sizes,
        steps=generation_steps,
        gen_length=gen_length,
        block_length=gen_length,
        tokenizer=tokenizer,
        stopping_criteria=["<|eot_id|>"],
        prefix_refresh_interval=32,
        threshold=1,
        init=True,
    )
    if isinstance(result, tuple) and len(result) >= 3:
        return result[1], result[2]
    return None, None


def optimize_image_with_gpo(
    model,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    tokenizer,
    *,
    attack_steps: int = 300,
    epsilon: float = 8 / 255,
    lr: float = 1e-1,
    generation_steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    target_dict: dict[int, list[int]] | None = None,
    negative_ids: list[int] | None = None,
    early_stop_threshold: float = 0.95,
) -> tuple[torch.Tensor, list[float], tuple[int | None, int | None]]:
    """Run the GPO-V image perturbation loop and return adversarial image tensor."""
    device = input_ids.device
    x_orig = image_tensor.detach().to(dtype=torch.float32, device=device)
    delta_init = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_orig + delta_init, 0.0, 1.0).detach()
    x_adv.requires_grad = True

    start_idx, end_idx = initialize_response_span(
        model,
        input_ids,
        x_adv.to(dtype=torch.float16),
        image_sizes,
        tokenizer,
        generation_steps,
        gen_length,
        block_length,
    )

    loss_fn = PositionTokenAnchorLoss(
        target_dict=target_dict or DEFAULT_TARGET_DICT,
        negative_ids=negative_ids or DEFAULT_NEGATIVE_IDS,
        weight_target=1.0,
        weight_negative=10.0,
        device=device,
    )
    optimizer = optim.Adam([x_adv], lr=lr)
    history = []

    # Memory mitigation for large LLaDA-V backprop through generation.
    was_training = model.training
    cache_was_enabled = getattr(getattr(model, "config", None), "use_cache", None)
    grad_ckpt_enabled = False
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            grad_ckpt_enabled = True
        if cache_was_enabled is not None:
            model.config.use_cache = False
        model.train()

        for step in range(attack_steps):
            optimizer.zero_grad()
            response_logits = differentiable_response_logits(
                model,
                input_ids,
                x_adv.to(dtype=torch.float16),
                image_sizes,
                gen_length=gen_length,
            )
            if start_idx is not None and end_idx is not None:
                span_start = max(0, min(start_idx, response_logits.shape[1] - 1))
                span_end = max(span_start + 1, min(end_idx, response_logits.shape[1]))
                response_logits = response_logits[:, span_start:span_end, :]
            if response_logits.shape[1] >= 2:
                response_logits = response_logits.clone()
                if response_logits.shape[-1] > 126348:
                    response_logits[:, -1, 126348] = torch.finfo(response_logits.dtype).min
                if response_logits.shape[-1] > 13:
                    response_logits[:, -2, 13] = torch.finfo(response_logits.dtype).min
            loss, monitor = loss_fn(response_logits)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta = torch.clamp(x_adv.data - x_orig, -epsilon, epsilon)
                x_adv.data = torch.clamp(x_orig + delta, 0.0, 1.0)
            history.append(float(loss.item()))

            if monitor and all(hit and prob >= early_stop_threshold for _, prob, hit in monitor.values()):
                LOGGER.info("GPO-V early stop at attack step %d.", step + 1)
                break
    finally:
        if cache_was_enabled is not None:
            model.config.use_cache = cache_was_enabled
        if grad_ckpt_enabled and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.train(was_training)

    return x_adv.detach(), history, (start_idx, end_idx)


@torch.no_grad()
def validation_generate(
    model,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    tokenizer,
    *,
    generation_steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
) -> tuple[torch.Tensor, str]:
    """Run final LLaDA-V validation generation and decode text."""
    cont = model.generate(
        input_ids,
        images=images.to(dtype=torch.float16),
        image_sizes=image_sizes,
        steps=generation_steps,
        gen_length=gen_length,
        block_length=block_length,
        tokenizer=tokenizer,
        stopping_criteria=["<|eot_id|>"],
        prefix_refresh_interval=32,
        threshold=1,
        validation=True,
    )
    text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    return cont, text
