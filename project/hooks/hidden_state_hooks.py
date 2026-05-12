from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookConfig:
    """Configuration for hidden-state capture.

    selected_layers can contain integer transformer block indices or exact
    module names from model.named_modules(). Integer indices are resolved
    against common HuggingFace/LLaDA block containers.
    """

    selected_layers: tuple[int | str, ...] = (15, 23, 31)
    selected_steps: tuple[int, ...] = (5, 10, 15)
    pool: str = "mean"
    offload_to_cpu: bool = False
    capture_dtype: torch.dtype | None = torch.float32
    keep_sequence: bool = False
    enabled: bool = True
    log_every_capture: bool = False

    def __post_init__(self) -> None:
        if self.pool != "mean":
            raise ValueError(f"Unsupported pool={self.pool!r}; only mean pooling is implemented.")
        if not self.selected_layers:
            raise ValueError("selected_layers must contain at least one layer.")
        if not self.selected_steps:
            raise ValueError("selected_steps must contain at least one denoising step.")


@dataclass
class HookRecord:
    """A single captured hidden-state summary.

    hidden has shape [batch, hidden_dim] when keep_sequence=False and
    [batch, seq_len, hidden_dim] when keep_sequence=True.
    """

    layer_name: str
    layer_index: int | None
    global_step: int
    block_index: int
    local_step: int
    hidden: torch.Tensor
    original_shape: tuple[int, ...]
    pooled: bool


@dataclass
class _CaptureState:
    active: bool = False
    global_step: int = -1
    block_index: int = -1
    local_step: int = -1
    batch_size: int | None = None
    attention_mask: torch.Tensor | None = None


def _get_nested_attr(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _candidate_layer_containers(model: nn.Module) -> list[tuple[str, Any]]:
    """Return likely transformer-block containers for HF remote-code models."""

    paths = (
        "model.layers",
        "model.transformer.blocks",
        "model.transformer.layers",
        "model.decoder.layers",
        "transformer.blocks",
        "transformer.layers",
        "backbone.layers",
        "layers",
    )
    found: list[tuple[str, Any]] = []
    for path in paths:
        container = _get_nested_attr(model, path)
        if isinstance(container, (nn.ModuleList, list, tuple)):
            found.append((path, container))
    return found


def resolve_transformer_layers(model: nn.Module, selected_layers: Iterable[int | str]) -> dict[int | str, tuple[str, nn.Module, int | None]]:
    """Resolve selected layer ids to modules.

    Returns a mapping from the original layer specifier to
    (module_name, module, integer_index_or_None).
    """

    named_modules = dict(model.named_modules())
    containers = _candidate_layer_containers(model)
    resolved: dict[int | str, tuple[str, nn.Module, int | None]] = {}

    for layer in selected_layers:
        if isinstance(layer, str):
            if layer not in named_modules:
                raise KeyError(f"Layer name {layer!r} not found in model.named_modules().")
            resolved[layer] = (layer, named_modules[layer], None)
            continue

        if layer < 0:
            raise ValueError(f"Layer indices must be non-negative, got {layer}.")

        for container_name, container in containers:
            if layer < len(container):
                module = container[layer]
                module_name = f"{container_name}.{layer}"
                resolved[layer] = (module_name, module, layer)
                break
        else:
            available = ", ".join(f"{name}[0..{len(container) - 1}]" for name, container in containers)
            raise IndexError(
                f"Could not resolve transformer layer index {layer}. "
                f"Discovered containers: {available or 'none'}."
            )

    return resolved


def _first_tensor(output: Any) -> torch.Tensor:
    """Extract the hidden-state tensor from common module output structures."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    if hasattr(output, "hidden_states") and torch.is_tensor(output.hidden_states):
        return output.hidden_states
    raise TypeError(f"Could not extract a tensor from hook output type {type(output)!r}.")


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean pool [batch, seq_len, hidden_dim] using an optional [batch, seq_len] mask."""

    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden state [batch, seq_len, hidden_dim], got shape {tuple(hidden.shape)}.")

    if attention_mask is None:
        return hidden.mean(dim=1)

    if attention_mask.shape[:2] != hidden.shape[:2]:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} is incompatible with hidden shape {tuple(hidden.shape)}."
        )
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


class HiddenStateCapture:
    """Forward-hook manager for LLaDA hidden-state trajectories.

    The capture keeps only pooled states by default. For LLaDA-8B-like hidden
    size 4096, a captured pooled tensor costs batch * 4096 * bytes_per_value per
    layer/step; retaining full [batch, seq_len, hidden_dim] sequences is much
    more expensive and should be reserved for debugging.
    """

    def __init__(self, model: nn.Module, config: HookConfig) -> None:
        self.model = model
        self.config = config
        self._state = _CaptureState()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_specs: dict[int | str, tuple[str, nn.Module, int | None]] = {}
        self.records: list[HookRecord] = []

    @property
    def is_attached(self) -> bool:
        return bool(self._handles)

    def attach(self) -> None:
        if self.is_attached:
            return
        self._layer_specs = resolve_transformer_layers(self.model, self.config.selected_layers)
        for _, (layer_name, module, layer_index) in self._layer_specs.items():
            handle = module.register_forward_hook(self._make_hook(layer_name, layer_index))
            self._handles.append(handle)
        LOGGER.info("Attached %d hidden-state hooks.", len(self._handles))

    def remove(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        self._state = _CaptureState()
        LOGGER.info("Removed hidden-state hooks.")

    def clear(self) -> None:
        self.records.clear()

    def start_step(
        self,
        *,
        global_step: int,
        block_index: int,
        local_step: int,
        batch_size: int,
        attention_mask: torch.Tensor | None,
    ) -> None:
        should_capture = self.config.enabled and global_step in self.config.selected_steps
        self._state = _CaptureState(
            active=should_capture,
            global_step=global_step,
            block_index=block_index,
            local_step=local_step,
            batch_size=batch_size,
            attention_mask=attention_mask,
        )

    def finish_step(self) -> list[HookRecord]:
        step_records = [record for record in self.records if record.global_step == self._state.global_step]
        self._state.active = False
        return step_records

    def _make_hook(self, layer_name: str, layer_index: int | None):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._state.active:
                return

            hidden = _first_tensor(output)
            original_shape = tuple(hidden.shape)
            if hidden.ndim != 3:
                LOGGER.debug("Skipping %s because output shape is %s.", layer_name, original_shape)
                return

            # HF decoder blocks normally return [batch, seq_len, hidden_dim].
            # If a model uses [seq_len, batch, hidden_dim], transpose for pooling.
            if self._state.batch_size is not None and hidden.shape[0] != self._state.batch_size:
                if hidden.shape[1] == self._state.batch_size:
                    hidden = hidden.transpose(0, 1)

            if self._state.batch_size is not None and hidden.shape[0] > self._state.batch_size:
                # Classifier-free guidance doubles the batch as [conditioned, unconditioned].
                hidden = hidden[: self._state.batch_size]

            if self.config.keep_sequence:
                captured = hidden
                pooled = False
            else:
                captured = _mean_pool(hidden, self._state.attention_mask)
                pooled = True

            captured = captured.detach()
            if self.config.capture_dtype is not None:
                captured = captured.to(dtype=self.config.capture_dtype)
            if self.config.offload_to_cpu:
                captured = captured.cpu()

            record = HookRecord(
                layer_name=layer_name,
                layer_index=layer_index,
                global_step=self._state.global_step,
                block_index=self._state.block_index,
                local_step=self._state.local_step,
                hidden=captured,
                original_shape=original_shape,
                pooled=pooled,
            )
            self.records.append(record)

            if self.config.log_every_capture:
                LOGGER.info(
                    "Captured %s step=%d hidden=%s original=%s.",
                    layer_name,
                    self._state.global_step,
                    tuple(captured.shape),
                    original_shape,
                )

        return hook

    def __enter__(self) -> "HiddenStateCapture":
        self.attach()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.remove()
