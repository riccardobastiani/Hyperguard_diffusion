from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from project.geometry import HyperbolicProjector, LorentzManifoldOps


@dataclass(frozen=True)
class DeepSVDDConfig:
    nu: float = 0.1
    radius_init: float = 1.0
    center_eps: float = 1e-3
    score_squared: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.nu <= 1.0:
            raise ValueError("nu must be in (0, 1].")


@dataclass
class SVDDOutput:
    loss: torch.Tensor
    distances: torch.Tensor
    radius: torch.Tensor
    margin: torch.Tensor


class HyperbolicDeepSVDD(nn.Module):
    """Soft-boundary Deep SVDD using Lorentz geodesic distance."""

    def __init__(self, projector: HyperbolicProjector, config: DeepSVDDConfig = DeepSVDDConfig()) -> None:
        super().__init__()
        self.projector = projector
        self.config = config
        self.lorentz: LorentzManifoldOps = projector.lorentz

        center = torch.zeros(projector.hyperbolic_dim + 1, dtype=torch.float32)
        center[0] = self.lorentz.sqrt_k
        self.register_buffer("center", center)
        self.raw_radius = nn.Parameter(torch.tensor(float(config.radius_init)))

    @property
    def radius(self) -> torch.Tensor:
        return F.softplus(self.raw_radius) + 1e-6

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden)

    def distances(self, hidden: torch.Tensor, *, squared: bool | None = None) -> torch.Tensor:
        z = self.encode(hidden)
        center = self.center.to(device=z.device, dtype=z.dtype).expand_as(z)
        use_squared = self.config.score_squared if squared is None else squared
        return self.lorentz.distance(z, center, squared=use_squared)

    def score(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.distances(hidden, squared=self.config.score_squared)

    def forward(self, hidden: torch.Tensor) -> SVDDOutput:
        distances = self.distances(hidden, squared=False)
        radius = self.radius.to(device=distances.device, dtype=distances.dtype)
        margin = distances.pow(2) - radius.pow(2)
        loss = radius.pow(2) + (1.0 / self.config.nu) * torch.relu(margin).mean()
        return SVDDOutput(loss=loss, distances=distances, radius=radius, margin=margin)

    @torch.no_grad()
    def initialize_center(self, hidden_batches: Iterable[torch.Tensor], *, device: torch.device | str) -> torch.Tensor:
        """Initialize center from safe hidden states and project it to Lorentz.

        This computes an ambient mean of safe embeddings, clamps near-zero
        coordinates, then projects to the upper sheet. It is deterministic for a
        fixed projector and hidden input order.
        """

        self.eval()
        total: torch.Tensor | None = None
        count = 0
        for hidden in hidden_batches:
            z = self.encode(hidden.to(device))
            batch_sum = z.float().sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += z.shape[0]
        if total is None or count == 0:
            raise ValueError("Cannot initialize SVDD center from an empty hidden-state stream.")

        mean = total / count
        mean[1:][mean[1:].abs() < self.config.center_eps] = self.config.center_eps
        projected = self.lorentz.project(mean.unsqueeze(0)).squeeze(0).detach().cpu()
        self.center.copy_(projected)
        return self.center

    def save_checkpoint(self, path: str | Path, *, metadata: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "hidden_dim": self.projector.hidden_dim,
            "hyperbolic_dim": self.projector.hyperbolic_dim,
            "metadata": metadata or {},
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "HyperbolicDeepSVDD":
        payload = torch.load(Path(path), map_location=map_location)
        projector = HyperbolicProjector(
            hidden_dim=int(payload["hidden_dim"]),
            hyperbolic_dim=int(payload["hyperbolic_dim"]),
        )
        model = cls(projector, payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model
