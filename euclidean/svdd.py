import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

try:
    import geoopt
except ModuleNotFoundError:  # Euclidean baseline does not require geoopt.
    geoopt = None
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from hyperbolic_projection import HyperbolicProjection

LOGGER = logging.getLogger("svdd")


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive tensors."""
    return x + torch.log(-torch.expm1(-x))


# ---------------------------------------------------------------------------
# Hyperbolic distance
# ---------------------------------------------------------------------------

def lorentz_distance(manifold, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lorentz (hyperbolic) distance between batched points p and q.

    Args:
        manifold: A geoopt Lorentz manifold instance.
        p: Points  [batch, dim+1]
        q: Points  [batch, dim+1]  or  [dim+1]  (broadcast)

    Returns:
        Distances  [batch]
    """
    return manifold.dist(p, q)


def euclidean_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between batched points p and q."""
    return torch.linalg.vector_norm(p - q, dim=-1)


# ---------------------------------------------------------------------------
# SVDD center initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_center(
    projector: "HyperbolicProjection",
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
        nu:         Soft-boundary weight. Smaller values penalize safe samples
                    outside the sphere more strongly. (0, 1].
    """

    def __init__(
        self,
        projector: "HyperbolicProjection",
        center: torch.Tensor,
        nu: float = 0.1,
        initial_R: float = 1.0,
        radius_eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 < nu <= 1.0:
            raise ValueError("nu must be in the interval (0, 1].")
        if initial_R <= 0.0:
            raise ValueError("initial_R must be positive.")

        self.projector = projector
        self.manifold = projector.manifold
        self.nu = nu
        self.radius_eps = radius_eps

        # Center is fixed after initialization (not a learnable parameter)
        self.register_buffer("center", center)

        # Radius R is optimized jointly with the projector.  The raw parameter
        # is transformed through softplus so the exposed radius stays positive.
        initial_radius = torch.tensor([initial_R], dtype=center.dtype, device=center.device)
        raw_radius = _inverse_softplus((initial_radius - radius_eps).clamp_min(1e-12))
        self._log_R = nn.Parameter(raw_radius)

    @property
    def R(self) -> torch.Tensor:
        return F.softplus(self._log_R) + self.radius_eps

    @torch.no_grad()
    def set_radius(self, value: Union[float, torch.Tensor]) -> None:
        """Set R while preserving the positive softplus parameterization."""
        radius = torch.as_tensor(value, dtype=self._log_R.dtype, device=self._log_R.device).reshape_as(self._log_R)
        raw_radius = _inverse_softplus((radius - self.radius_eps).clamp_min(1e-12))
        self._log_R.copy_(raw_radius)

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
        # Safe: minimize hypersphere volume and penalize samples outside it.
        safe_emb = self.forward(safe)
        center = self.center.unsqueeze(0).expand_as(safe_emb)
        safe_dist_sq = lorentz_distance(self.manifold, safe_emb, center) ** 2
        R_sq = self.R ** 2
        safe_loss = torch.clamp(safe_dist_sq - R_sq, min=0.0).mean() / self.nu
        loss = R_sq.squeeze() + safe_loss

        if unsafe is not None:
            # Unsafe: maximize distance by penalizing unsafe samples inside R.
            unsafe_emb = self.forward(unsafe)
            center_u = self.center.unsqueeze(0).expand_as(unsafe_emb)
            unsafe_dist_sq = lorentz_distance(self.manifold, unsafe_emb, center_u) ** 2
            repulsion = torch.clamp(R_sq - unsafe_dist_sq, min=0.0).mean()
            return loss + unsafe_weight * repulsion

        return loss

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


class EuclideanSVDD(nn.Module):
    """One-class SVDD baseline in the original Euclidean probe space.

    This is the baseline counterpart to HyperbolicSVDD: hidden-state probes are
    scored by their L2 distance to the safe centroid, with the radius calibrated
    from safe distances.
    """

    def __init__(
        self,
        center: torch.Tensor,
        initial_R: float = 1.0,
        radius_eps: float = 1e-6,
    ):
        super().__init__()
        if initial_R <= 0.0:
            raise ValueError("initial_R must be positive.")

        self.radius_eps = radius_eps
        self.register_buffer("center", center)

        initial_radius = torch.tensor([initial_R], dtype=center.dtype, device=center.device)
        raw_radius = _inverse_softplus((initial_radius - radius_eps).clamp_min(1e-12))
        self._log_R = nn.Parameter(raw_radius, requires_grad=False)

    @property
    def R(self) -> torch.Tensor:
        return F.softplus(self._log_R) + self.radius_eps

    @torch.no_grad()
    def set_radius(self, value: Union[float, torch.Tensor]) -> None:
        """Set R while preserving the positive softplus parameterization."""
        radius = torch.as_tensor(value, dtype=self._log_R.dtype, device=self._log_R.device).reshape_as(self._log_R)
        raw_radius = _inverse_softplus((radius - self.radius_eps).clamp_min(1e-12))
        self._log_R.copy_(raw_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probes unchanged so the detector matches the SVDD interface."""
        return x.float()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return Euclidean distance to the safe centroid for each sample."""
        features = self.forward(x)
        center = self.center.unsqueeze(0).expand_as(features)
        return euclidean_distance(features, center)

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return True for samples predicted as unsafe (distance > R)."""
        return self.predict(x) > self.R


@torch.no_grad()
def fit_euclidean_svdd(
    safe_features: torch.Tensor,
    nu: float = 0.01,
    radius_quantile: Optional[float] = None,
) -> EuclideanSVDD:
    """Fit the Euclidean one-class baseline from safe probe features.

    Args:
        safe_features:    Safe hooks [n_safe, in_dim].
        nu:               Fraction of safe samples allowed outside the sphere
                          when radius_quantile is not provided.
        radius_quantile:  Optional explicit safe-distance quantile for R.

    Returns:
        EuclideanSVDD with center=mean(safe_features) and calibrated R.
    """
    if not 0.0 < nu <= 1.0:
        raise ValueError("nu must be in the interval (0, 1].")
    if radius_quantile is None:
        radius_quantile = 1.0 - nu
    if not 0.0 < radius_quantile <= 1.0:
        raise ValueError("radius_quantile must be in the interval (0, 1].")

    safe_features = safe_features.float()
    if not torch.isfinite(safe_features).all():
        raise ValueError("Safe probes contain NaN/Inf; delete cache and recompute.")

    center = safe_features.mean(dim=0)
    distances = euclidean_distance(safe_features, center.unsqueeze(0).expand_as(safe_features))
    radius = torch.quantile(distances, radius_quantile).clamp_min(1e-6)
    detector = EuclideanSVDD(center=center, initial_R=float(radius))
    LOGGER.info(
        "Euclidean SVDD fitted. safe distance stats: min=%.6f mean=%.6f max=%.6f R(q=%.4f)=%.6f.",
        float(distances.min()),
        float(distances.mean()),
        float(distances.max()),
        radius_quantile,
        float(detector.R.detach()),
    )
    return detector


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

    optimizer = torch.optim.Adam(list(svdd.projector.parameters()) + [svdd._log_R], lr=lr)
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

        if epoch % 10 == 0 or epoch == 1:
            LOGGER.info(
                "Epoch %d/%d — loss: %.6f  R: %.6f",
                epoch, epochs, epoch_loss / steps, float(svdd.R.detach()),
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
            "R": svdd.R.detach().cpu(),
            "log_R": svdd._log_R.detach().cpu(),
            "radius_eps": svdd.radius_eps,
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
    from hyperbolic_projection import HyperbolicProjection

    ckpt = torch.load(path, map_location=device)
    projector = HyperbolicProjection(
        in_dim=ckpt["in_dim"],
        proj_dim=ckpt["proj_dim"],
        curvature=ckpt["curvature"],
    )
    projector.load_state_dict(ckpt["projector_state"])
    svdd = HyperbolicSVDD(
        projector=projector,
        center=ckpt["center"].to(device),
        nu=ckpt["nu"],
        initial_R=float(torch.as_tensor(ckpt.get("R", 1.0)).reshape(-1)[0]),
        radius_eps=ckpt.get("radius_eps", 1e-6),
    )
    if "log_R" in ckpt:
        svdd._log_R.data.copy_(ckpt["log_R"].to(device).reshape_as(svdd._log_R))
    elif "R" in ckpt:
        svdd.set_radius(ckpt["R"].to(device))
    svdd.to(device)
    svdd.eval()
    LOGGER.info("Checkpoint loaded from %s. R=%.6f.", path, float(svdd.R.detach()))
    return svdd


def save_euclidean_checkpoint(detector: EuclideanSVDD, path: Path) -> None:
    """Save a EuclideanSVDD baseline checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "detector_type": "euclidean_svdd",
            "center": detector.center.cpu(),
            "R": detector.R.detach().cpu(),
            "log_R": detector._log_R.detach().cpu(),
            "radius_eps": detector.radius_eps,
            "in_dim": detector.center.numel(),
        },
        path,
    )
    LOGGER.info("Euclidean checkpoint saved to %s.", path)


def load_euclidean_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> EuclideanSVDD:
    """Load a EuclideanSVDD baseline checkpoint."""
    ckpt = torch.load(path, map_location=device)
    detector = EuclideanSVDD(
        center=ckpt["center"].to(device),
        initial_R=float(torch.as_tensor(ckpt.get("R", 1.0)).reshape(-1)[0]),
        radius_eps=ckpt.get("radius_eps", 1e-6),
    )
    if "log_R" in ckpt:
        detector._log_R.data.copy_(ckpt["log_R"].to(device).reshape_as(detector._log_R))
    elif "R" in ckpt:
        detector.set_radius(ckpt["R"].to(device))
    detector.to(device)
    detector.eval()
    LOGGER.info("Euclidean checkpoint loaded from %s. R=%.6f.", path, float(detector.R.detach()))
    return detector

