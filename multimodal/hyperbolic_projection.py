import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


class HyperbolicProjection(nn.Module):
    """Projects Euclidean mean-pooled features to the Lorentz hyperboloid.

    Pipeline:
        1. Linear  R^in_dim  →  R^proj_dim
        2. Prepend a zero component to satisfy the tangent-at-origin constraint.
        3. Exponential map (Lorentz)  →  H^proj_dim  ⊂  R^{proj_dim+1}

    Args:
        in_dim:     Dimension of the input Euclidean features (e.g. 4096 for LLaDA-8B).
        proj_dim:   Dimension of the projected space before mapping (e.g. 128).
        curvature:  Curvature constant k of the Lorentz manifold (default 1.0).
    """

    def __init__(
        self,
        in_dim: int = 4096,
        proj_dim: int = 128,
        curvature: float = 1.0,
        max_norm: float = 1.0,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.linear = nn.Linear(in_dim, proj_dim, bias=False)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Euclidean feature vectors  [batch, in_dim]

        Returns:
            Points on the Lorentz hyperboloid  [batch, proj_dim + 1]
        """
        v = self.linear(x).float()                                      # [batch, proj_dim]
        # Normalize to unit sphere then scale to fixed norm.
        # This makes weight magnitude irrelevant — only directions are learned —
        # which prevents the trivial collapse where weights → 0 shrinks all distances.
        v = F.normalize(v, dim=-1) * self.max_norm
        zeros = torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)
        u = torch.cat([zeros, v], dim=-1)                               # [batch, proj_dim + 1]
        return self.manifold.expmap0(u)                                 # [batch, proj_dim + 1]
