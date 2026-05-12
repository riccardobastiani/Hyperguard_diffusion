from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from project.hooks import HiddenStateCapture, HookConfig, HookRecord
from project.monitoring.runtime import MonitoringDecision


@dataclass
class ExtractionState:
    records: list[HookRecord]


class HiddenTrajectoryExtractor:
    """Generation-compatible monitor that only records hidden trajectories."""

    def __init__(self, hook_config: HookConfig) -> None:
        self.hook_config = hook_config
        self.capture: HiddenStateCapture | None = None
        self.records: list[HookRecord] = []

    def attach(self, model: nn.Module) -> None:
        self.capture = HiddenStateCapture(model, self.hook_config)
        self.capture.attach()

    def remove(self) -> None:
        if self.capture is not None:
            self.capture.remove()
        self.capture = None

    def reset(self) -> None:
        self.records.clear()
        if self.capture is not None:
            self.capture.clear()

    def on_step_begin(
        self,
        *,
        global_step: int,
        block_index: int,
        local_step: int,
        batch_size: int,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if self.capture is None:
            raise RuntimeError("HiddenTrajectoryExtractor.attach(model) must be called before generation.")
        self.capture.start_step(
            global_step=global_step,
            block_index=block_index,
            local_step=local_step,
            batch_size=batch_size,
            attention_mask=attention_mask,
        )

    def on_step_observed(self, *, global_step: int) -> MonitoringDecision:
        if self.capture is None:
            raise RuntimeError("HiddenTrajectoryExtractor.attach(model) must be called before generation.")
        step_records = self.capture.finish_step()
        self.records.extend(step_records)
        return MonitoringDecision(
            should_stop=False,
            score=0.0,
            step_scores={},
            global_step=global_step,
            reason="extract_only",
        )
