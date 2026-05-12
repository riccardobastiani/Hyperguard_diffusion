from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

try:
    import geoopt
except ImportError:  # pragma: no cover - exercised only when geoopt is absent.
    geoopt = None


@dataclass(frozen=True)
class LorentzConfig:
    """Lorentz model settings.

    Points have shape [..., dim + 1]. The first coordinate is time-like and
    satisfies -x0^2 + ||x_spatial||^2 = -k.
    """

    k: float = 1.0
    eps: float = 1e-6
    max_tangent_norm: float = 15.0


@dataclass(frozen=True)
class LorentzValidationResult:
    max_constraint_error: float
    finite: bool
    min_time: float


class LorentzManifoldOps:
    """Numerically stable Lorentz operations with optional Geoopt backing."""

    def __init__(self, config: LorentzConfig = LorentzConfig()) -> None:
        self.config = config
        self.geoopt_manifold = geoopt.manifolds.Lorentz(k=config.k) if geoopt is not None else None

    @property
    def k(self) -> float:
        return self.config.k

    @property
    def sqrt_k(self) -> float:
        return math.sqrt(self.config.k)

    def origin(self, batch_shape: tuple[int, ...], dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        spatial = torch.zeros(*batch_shape, dim, device=device, dtype=dtype)
        time = torch.full((*batch_shape, 1), self.sqrt_k, device=device, dtype=dtype)
        return torch.cat([time, spatial], dim=-1)

    def lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"Lorentz inner expects equal shapes, got {tuple(x.shape)} and {tuple(y.shape)}.")
        time = -x[..., :1] * y[..., :1]
        spatial = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        return time + spatial

    def expmap0(self, tangent_spatial: torch.Tensor) -> torch.Tensor:
        """Map Euclidean tangent vectors [batch, dim] to Lorentz points [batch, dim + 1]."""

        work = tangent_spatial.float()
        norm = work.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
        clipped_norm = norm.clamp_max(self.config.max_tangent_norm)
        work = work * (clipped_norm / norm)

        scaled = clipped_norm / self.sqrt_k
        time = self.sqrt_k * torch.cosh(scaled)
        spatial = self.sqrt_k * torch.sinh(scaled) * work / clipped_norm.clamp_min(self.config.eps)
        point = torch.cat([time, spatial], dim=-1)
        return self.project(point).to(dtype=tangent_spatial.dtype)

    def project(self, point: torch.Tensor) -> torch.Tensor:
        """Project arbitrary ambient coordinates back to the upper Lorentz sheet."""

        spatial = point[..., 1:].float()
        time = torch.sqrt(torch.clamp(self.k + spatial.pow(2).sum(dim=-1, keepdim=True), min=self.config.eps))
        projected = torch.cat([time, spatial], dim=-1)
        return projected.to(dtype=point.dtype)

    def distance(self, x: torch.Tensor, y: torch.Tensor, *, squared: bool = False) -> torch.Tensor:
        """Geodesic distance on the Lorentz hyperboloid.

        x and y have shape [batch, dim + 1] or broadcast-compatible leading
        dimensions. The returned tensor has shape [batch].
        """

        x_f = x.float()
        y_f = y.float()
        z = -self.lorentz_inner(x_f, y_f).squeeze(-1) / self.k
        z = z.clamp_min(1.0 + self.config.eps)
        dist = self.sqrt_k * torch.acosh(z)
        if squared:
            dist = dist.pow(2)
        return dist.to(dtype=x.dtype)

    def validate(self, point: torch.Tensor) -> LorentzValidationResult:
        with torch.no_grad():
            point_f = point.float()
            constraint = self.lorentz_inner(point_f, point_f).squeeze(-1) + self.k
            return LorentzValidationResult(
                max_constraint_error=float(constraint.abs().max().item()),
                finite=bool(torch.isfinite(point_f).all().item()),
                min_time=float(point_f[..., 0].min().item()),
            )


class HyperbolicProjector(nn.Module):
    """Linear projection 4096 -> 128 followed by Lorentz expmap at the origin.

    Input shape: [batch, hidden_dim].
    Output shape: [batch, hyperbolic_dim + 1], where coordinate 0 is time-like.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        hyperbolic_dim: int = 128,
        *,
        lorentz: LorentzManifoldOps | None = None,
        bias: bool = False,
        normalize_tangent: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.linear = nn.Linear(hidden_dim, hyperbolic_dim, bias=bias)
        self.lorentz = lorentz or LorentzManifoldOps()
        self.normalize_tangent = normalize_tangent
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"HyperbolicProjector expects [batch, hidden_dim], got {tuple(hidden.shape)}.")
        tangent = self.linear(hidden.float())
        if self.normalize_tangent:
            tangent = F.layer_norm(tangent, (self.hyperbolic_dim,))
        return self.lorentz.expmap0(tangent)
