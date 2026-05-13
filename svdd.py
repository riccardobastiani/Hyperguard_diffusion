import logging
from pathlib import Path
from typing import Optional

import geoopt
import torch
import torch.nn as nn

from hyperbolic_projection import HyperbolicProjection

LOGGER = logging.getLogger("svdd")


# ---------------------------------------------------------------------------
# Hyperbolic distance
# ---------------------------------------------------------------------------

def lorentz_distance(manifold: geoopt.manifolds.Lorentz, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lorentz (hyperbolic) distance between batched points p and q.

    Args:
        manifold: A geoopt Lorentz manifold instance.
        p: Points  [batch, dim+1]
        q: Points  [batch, dim+1]  or  [dim+1]  (broadcast)

    Returns:
        Distances  [batch]
    """
    return manifold.dist(p, q)


# ---------------------------------------------------------------------------
# SVDD center initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_center(
    projector: HyperbolicProjection,
    safe_features: torch.Tensor,
    eps: float = 0.1,
    max_norm: float = 15.0,
) -> torch.Tensor:
    """Compute the initial SVDD center as the mean of safe hyperbolic embeddings.

    Projects all safe features, computes their Euclidean mean, then projects
    that mean back onto the Lorentz manifold via expmap0.  This is a fast
    approximation of the Fréchet mean suitable for initialization.

    Args:
        projector:     Trained-or-random HyperbolicProjection module.
        safe_features: Euclidean mean-pooled safe hooks  [n_safe, in_dim].
        eps:           If the Euclidean norm of the mean tangent vector is
                       smaller than eps, the center is nudged to avoid
                       collapse to the manifold origin.

    Returns:
        Center point on the Lorentz hyperboloid  [proj_dim + 1].
    """
    projector.eval()
    embeddings = projector(safe_features)           # [n_safe, proj_dim + 1]
    if not torch.isfinite(embeddings).all():
        raise ValueError("Non-finite values in projected safe embeddings; check probe cache and projection.")

    # Average in the ambient Minkowski space then project back
    mean = embeddings.mean(dim=0)                   # [proj_dim + 1]

    # Convert to tangent vector at origin: zero out the time component
    tangent = mean.clone()
    tangent[0] = 0.0

    if tangent[1:].norm() < eps:
        tangent[1:] = tangent[1:] + eps

    if max_norm is not None:
        norm = tangent[1:].norm().clamp_min(1e-6)
        scale = min(max_norm / float(norm), 1.0)
        tangent[1:] = tangent[1:] * scale

    center = projector.manifold.expmap0(tangent.unsqueeze(0)).squeeze(0)  # [proj_dim + 1]
    LOGGER.info("Center initialized. Lorentz norm check: %.6f (should be ~-1/k = %.6f).",
                float(-center[0] ** 2 + (center[1:] ** 2).sum()),
                -1.0 / float(projector.manifold.k))
    return center


# ---------------------------------------------------------------------------
# One-Class SVDD (soft-boundary, hyperbolic distance)
# ---------------------------------------------------------------------------

class HyperbolicSVDD(nn.Module):
    """Soft-boundary One-Class SVDD on the Lorentz hyperboloid.

    The loss is the soft-boundary Deep SVDD objective:
        L = R^2 + (1/nu*n) * sum_i max(0, d_i^2 - R^2)

    where d_i = Lorentz distance from sample i to center c.

    Args:
        projector:  HyperbolicProjection that maps hooks to the hyperboloid.
        center:     Fixed center on the hyperboloid  [proj_dim + 1].
        nu:         Fraction of training samples allowed outside the sphere
                    (controls the trade-off between volume and outliers). (0, 1].
    """

    def __init__(
        self,
        projector: HyperbolicProjection,
        center: torch.Tensor,
        nu: float = 0.1,
    ):
        super().__init__()
        self.projector = projector
        self.manifold = projector.manifold
        self.nu = nu

        # Center is fixed after initialization (not a learnable parameter)
        self.register_buffer("center", center)

        # Radius R is updated analytically after each epoch (not via gradient)
        self.register_buffer("_R", torch.ones(1))

    @property
    def R(self) -> torch.Tensor:
        return self._R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean features to the hyperboloid.

        Args:
            x: [batch, in_dim]

        Returns:
            Hyperbolic embeddings [batch, proj_dim + 1]
        """
        return self.projector(x)

    def loss(self, safe: torch.Tensor, unsafe: Optional[torch.Tensor] = None, unsafe_weight: float = 1.0) -> torch.Tensor:
        """Contrastive SVDD loss.

        Safe samples are pulled toward the center; unsafe samples are pushed
        away from it.  Using both prevents hypersphere collapse.

        Args:
            safe:          [batch, in_dim]  — safe samples.
            unsafe:        [batch, in_dim]  — unsafe samples (optional).
            unsafe_weight: Weight on the repulsion term.

        Returns:
            Scalar loss.
        """
        # Safe: penalize only samples *outside* the sphere (hinge on d² - R²)
        safe_emb = self.forward(safe)
        center = self.center.unsqueeze(0).expand_as(safe_emb)
        safe_dist_sq = lorentz_distance(self.manifold, safe_emb, center) ** 2
        R_sq = self._R.detach() ** 2
        safe_loss = torch.clamp(safe_dist_sq - R_sq, min=0.0).mean() / self.nu

        if unsafe is not None:
            # Unsafe: maximize distance — penalize if unsafe falls inside (R + margin)
            unsafe_emb = self.forward(unsafe)
            center_u = self.center.unsqueeze(0).expand_as(unsafe_emb)
            unsafe_dist_sq = lorentz_distance(self.manifold, unsafe_emb, center_u) ** 2
            R_sq = self._R.detach() ** 2
            repulsion = torch.clamp(R_sq - unsafe_dist_sq, min=0.0).mean()
            return safe_loss + unsafe_weight * repulsion

        return safe_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return hyperbolic distance to center for each sample.

        Args:
            x: [batch, in_dim]

        Returns:
            Distances  [batch]
        """
        embeddings = self.forward(x)
        center = self.center.unsqueeze(0).expand_as(embeddings)
        return lorentz_distance(self.manifold, embeddings, center)

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return True for samples predicted as unsafe (distance > R).

        Args:
            x: [batch, in_dim]

        Returns:
            Bool tensor  [batch]  — True = STOP (unsafe), False = CONTINUE (safe)
        """
        return self.predict(x) > self.R


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_svdd(
    svdd: HyperbolicSVDD,
    safe_features: torch.Tensor,
    unsafe_features: Optional[torch.Tensor] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    unsafe_weight: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the HyperbolicSVDD with safe (and optionally unsafe) features.

    Args:
        svdd:            HyperbolicSVDD module.
        safe_features:   Safe hooks  [n_safe, in_dim].
        unsafe_features: Unsafe hooks  [n_unsafe, in_dim]  — strongly recommended
                         to prevent hypersphere collapse.
        epochs:          Number of training epochs.
        lr:              Learning rate for Adam.
        batch_size:      Mini-batch size.
        unsafe_weight:   Weight on the unsafe repulsion term.
        device:          Torch device.
    """
    svdd = svdd.to(device)
    safe_features = safe_features.to(device)
    if unsafe_features is not None:
        unsafe_features = unsafe_features.to(device)

    # Only optimize projector weights — R is updated analytically
    optimizer = torch.optim.Adam(svdd.projector.parameters(), lr=lr)
    n_safe = safe_features.shape[0]
    n_unsafe = unsafe_features.shape[0] if unsafe_features is not None else 0

    svdd.train()
    for epoch in range(1, epochs + 1):
        safe_perm = torch.randperm(n_safe, device=device)
        unsafe_perm = torch.randperm(n_unsafe, device=device) if n_unsafe > 0 else None
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n_safe, batch_size):
            safe_batch = safe_features[safe_perm[start:start + batch_size]]

            unsafe_batch = None
            if unsafe_perm is not None:
                u_start = start % n_unsafe
                unsafe_batch = unsafe_features[unsafe_perm[u_start:u_start + batch_size]]

            optimizer.zero_grad()
            loss = svdd.loss(safe_batch, unsafe_batch, unsafe_weight=unsafe_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        # Update R analytically as the (1-nu) quantile of safe training distances
        with torch.no_grad():
            all_dists = svdd.predict(safe_features)
            svdd._R.fill_(torch.quantile(all_dists, 1.0 - svdd.nu).item())

        if epoch % 10 == 0 or epoch == 1:
            LOGGER.info(
                "Epoch %d/%d — loss: %.6f  R: %.6f",
                epoch, epochs, epoch_loss / steps, float(svdd.R),
            )

    svdd.eval()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(svdd: HyperbolicSVDD, path: Path) -> None:
    """Save projector weights, center, and radius to a .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector_state": svdd.projector.state_dict(),
            "center": svdd.center.cpu(),
            "R": svdd._R.cpu(),
            "nu": svdd.nu,
            "in_dim": svdd.projector.linear.in_features,
            "proj_dim": svdd.projector.proj_dim,
            "curvature": svdd.manifold.k.item(),
        },
        path,
    )
    LOGGER.info("Checkpoint saved to %s.", path)


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> HyperbolicSVDD:
    """Load a HyperbolicSVDD from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    projector = HyperbolicProjection(
        in_dim=ckpt["in_dim"],
        proj_dim=ckpt["proj_dim"],
        curvature=ckpt["curvature"],
    )
    projector.load_state_dict(ckpt["projector_state"])
    svdd = HyperbolicSVDD(projector=projector, center=ckpt["center"].to(device), nu=ckpt["nu"])
    svdd._R.copy_(ckpt["R"].to(device))
    svdd.to(device)
    svdd.eval()
    LOGGER.info("Checkpoint loaded from %s. R=%.6f.", path, float(svdd.R))
    return svdd
