from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from project.hooks import HiddenStateCapture, HookConfig, HookRecord
from project.svdd import HyperbolicDeepSVDD

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeMonitorConfig:
    selected_steps: tuple[int, ...] = (5, 10, 15)
    step_weights: dict[int, float] = field(default_factory=lambda: {5: 0.25, 10: 0.35, 15: 0.40})
    threshold: float = 1.0
    layer_reduce: str = "mean"
    accumulation: str = "weighted_sum"
    log_path: str | None = None
    stop_on_first_threshold: bool = True

    def __post_init__(self) -> None:
        if self.layer_reduce != "mean":
            raise ValueError("Only layer_reduce='mean' is implemented.")
        if self.accumulation != "weighted_sum":
            raise ValueError("Only accumulation='weighted_sum' is implemented.")


@dataclass
class MonitoringDecision:
    should_stop: bool
    score: float
    step_scores: dict[int, float]
    global_step: int
    reason: str


class RuntimeJailbreakMonitor:
    """Runtime detector that scores hidden-state trajectories during generation."""

    def __init__(
        self,
        detector: HyperbolicDeepSVDD,
        hook_config: HookConfig,
        runtime_config: RuntimeMonitorConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.detector = detector
        self.hook_config = hook_config
        self.runtime_config = runtime_config
        self.device = torch.device(device) if device is not None else None
        self.capture: HiddenStateCapture | None = None
        self.step_scores: dict[int, float] = {}
        self.events: list[dict] = []
        self.stopped = False

    def attach(self, model: nn.Module) -> None:
        self.capture = HiddenStateCapture(model, self.hook_config)
        self.capture.attach()
        if self.device is not None:
            self.detector.to(self.device)
        self.detector.eval()

    def remove(self) -> None:
        if self.capture is not None:
            self.capture.remove()
        self.capture = None

    def reset(self) -> None:
        self.step_scores.clear()
        self.events.clear()
        self.stopped = False
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
            raise RuntimeError("RuntimeJailbreakMonitor.attach(model) must be called before generation.")
        self.capture.start_step(
            global_step=global_step,
            block_index=block_index,
            local_step=local_step,
            batch_size=batch_size,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def on_step_observed(self, *, global_step: int) -> MonitoringDecision:
        if self.capture is None:
            raise RuntimeError("RuntimeJailbreakMonitor.attach(model) must be called before generation.")

        records = self.capture.finish_step()
        if not records:
            return self._decision(False, global_step, "step_not_monitored")

        layer_scores = self._score_records(records)
        step_score = float(torch.stack(layer_scores).mean().item())
        self.step_scores[global_step] = step_score
        decision = self._decision(self._aggregate_score() > self.runtime_config.threshold, global_step, "threshold")
        self._log_event(records=records, decision=decision, layer_scores=[float(x.item()) for x in layer_scores])

        if decision.should_stop:
            self.stopped = True
            LOGGER.warning(
                "Jailbreak monitor triggered at step=%d score=%.6f threshold=%.6f.",
                global_step,
                decision.score,
                self.runtime_config.threshold,
            )
        return decision

    def _score_records(self, records: list[HookRecord]) -> list[torch.Tensor]:
        scores: list[torch.Tensor] = []
        target_device = self.device or next(self.detector.parameters()).device
        for record in records:
            hidden = record.hidden
            if hidden.ndim != 2:
                raise ValueError(
                    f"Runtime scoring requires pooled [batch, hidden_dim] records, got {tuple(hidden.shape)}."
                )
            hidden = hidden.to(device=target_device, non_blocking=True)
            batch_scores = self.detector.score(hidden.float())
            scores.append(batch_scores.mean())
        return scores

    def _aggregate_score(self) -> float:
        score = 0.0
        for step, step_score in self.step_scores.items():
            score += self.runtime_config.step_weights.get(step, 1.0) * step_score
        return float(score)

    def _decision(self, should_stop: bool, global_step: int, reason: str) -> MonitoringDecision:
        return MonitoringDecision(
            should_stop=bool(should_stop and self.runtime_config.stop_on_first_threshold),
            score=self._aggregate_score(),
            step_scores=dict(self.step_scores),
            global_step=global_step,
            reason=reason,
        )

    def _log_event(self, *, records: list[HookRecord], decision: MonitoringDecision, layer_scores: list[float]) -> None:
        event = {
            "time": time.time(),
            "global_step": decision.global_step,
            "score": decision.score,
            "step_scores": decision.step_scores,
            "should_stop": decision.should_stop,
            "reason": decision.reason,
            "layers": [record.layer_name for record in records],
            "layer_scores": layer_scores,
            "hidden_shapes": [record.original_shape for record in records],
        }
        self.events.append(event)
        if self.runtime_config.log_path is not None:
            path = Path(self.runtime_config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
