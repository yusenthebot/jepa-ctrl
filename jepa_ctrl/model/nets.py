from __future__ import annotations

import torch
from torch import nn

from .simnorm import SimNorm


def _latent_norm(latent_norm: str, simnorm_groups: int) -> nn.Module:
    """Latent normalizer for the encoder/predictor output: SimNorm (simplex, default) or
    Identity (RAW latent, for the SIGReg arm — SIGReg targets a Gaussian, not a simplex)."""
    if latent_norm == "simnorm":
        return SimNorm(simnorm_groups)
    if latent_norm == "none":
        return nn.Identity()
    raise ValueError(f"latent_norm must be 'simnorm' or 'none', got {latent_norm!r}")


def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * log(|x| + 1). Compresses wide-range targets near 0."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class Encoder(nn.Module):
    """f_theta: obs_dim -> hidden -> LayerNorm+Mish -> hidden -> Mish -> latent, then SimNorm.

    Exposes the pre-SimNorm activation (`pre`) so the predictor can add its residual delta in
    that space before re-normalizing. forward() returns the SimNorm latent.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden: int = 256,
        simnorm_groups: int = 8,
        latent_norm: str = "simnorm",
    ) -> None:
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, latent_dim)
        self.act = nn.Mish()
        self.simnorm = _latent_norm(latent_norm, simnorm_groups)

    def pre_simnorm(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln(self.l1(obs)))
        h = self.act(self.l2(h))
        return self.proj(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.simnorm(self.pre_simnorm(obs))


class Predictor(nn.Module):
    """g_phi: action head A(a) -> action_head_dim (never raw action scalars); body
    Linear(latent + a_dim) -> Mish -> Linear(hidden) -> Mish -> Linear(latent) predicts a delta
    in pre-SimNorm space. z_hat_{t+1} = SimNorm(z_t_pre + delta). Not wider than the encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        hidden: int = 256,
        action_head_dim: int = 64,
        simnorm_groups: int = 8,
        latent_norm: str = "simnorm",
    ) -> None:
        super().__init__()
        self.action_head = nn.Sequential(nn.Linear(act_dim, action_head_dim), nn.Mish())
        self.body = nn.Sequential(
            nn.Linear(latent_dim + action_head_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, latent_dim),
        )
        self.simnorm = _latent_norm(latent_norm, simnorm_groups)

    def forward(self, z_pre: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_pre is the CURRENT latent in pre-SimNorm space. Returns next SimNorm latent."""
        a = self.action_head(action)
        delta = self.body(torch.cat([z_pre, a], dim=-1))
        return self.simnorm(z_pre + delta)


class DistHead(nn.Module):
    """Distributional scalar head (reward or value). Predicts logits over `bins` evenly-spaced
    locations in symlog space [vmin, vmax]; to_scalar = symexp(softmax-weighted bin centers).

    twohot_encode maps a symlog-space target to a soft two-hot label for the cross-entropy /
    HL-Gauss loss. Optionally an ensemble of `num_q` independent heads (value).
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        hidden: int = 256,
        bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        num_q: int = 1,
    ) -> None:
        super().__init__()
        self.bins = int(bins)
        self.num_q = int(num_q)
        self.register_buffer("bin_centers", torch.linspace(vmin, vmax, bins))
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(latent_dim + act_dim, hidden),
                nn.Mish(),
                nn.Linear(hidden, hidden),
                nn.Mish(),
                nn.Linear(hidden, bins),
            )
            for _ in range(self.num_q)
        )

    def logits(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns logits stacked over the ensemble: shape (num_q, ..., bins)."""
        x = torch.cat([z, action], dim=-1)
        return torch.stack([h(x) for h in self.heads], dim=0)

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Map bin logits -> real scalar: symexp of the softmax-weighted symlog bin centers."""
        probs = torch.softmax(logits, dim=-1)
        sym = (probs * self.bin_centers).sum(dim=-1)
        return symexp(sym)

    def twohot_encode(self, target_sym: torch.Tensor) -> torch.Tensor:
        """Soft two-hot label over bins for a target ALREADY in symlog space."""
        centers = self.bin_centers
        x = target_sym.clamp(centers[0], centers[-1]).unsqueeze(-1)
        # distance to each center; the two nearest bins receive the interpolated mass
        dist = (x - centers).abs()
        idx = torch.topk(dist, 2, dim=-1, largest=False).indices
        lo = torch.minimum(idx[..., 0], idx[..., 1])
        hi = torch.maximum(idx[..., 0], idx[..., 1])
        c_lo = centers[lo]
        c_hi = centers[hi]
        denom = (c_hi - c_lo).clamp_min(1e-8)
        w_hi = ((x.squeeze(-1) - c_lo) / denom).clamp(0.0, 1.0)
        out = torch.zeros(*target_sym.shape, self.bins, device=target_sym.device)
        out.scatter_(-1, lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
        out.scatter_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return out


class InverseDynamicsHead(nn.Module):
    """g_inv: predict the action that drove the transition z_t -> z_{t+1}.

    MLP on concat(z_t, z_{t+1}) -> act_dim. The task-AGNOSTIC anti-collapse signal of the
    inverse-dynamics arm: to recover a from (z_t, z_{t+1}) the latents must encode
    action-discriminative state information, so the encoder/predictor cannot collapse to a
    constant. Operates on the SHARED rolled (predicted) latents so it shapes the PREDICTED
    dynamics, not only the encoder's near-bijective state map.
    """

    def __init__(self, latent_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """z_t, z_next: (..., latent_dim). Returns predicted action (..., act_dim)."""
        return self.net(torch.cat([z_t, z_next], dim=-1))
