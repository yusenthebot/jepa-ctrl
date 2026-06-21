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


class CNNEncoder(nn.Module):
    """Pixel f_theta: DrQ / TD-MPC2-style conv tower -> Linear -> latent, then latent_norm.

    obs_shape = (C, H, W) of a stacked uint8 frame stack (e.g. (9, 84, 84) for k=3 RGB). The
    uint8 -> float normalization (x/255 - 0.5) happens INSIDE forward/pre_simnorm so the buffer
    can store raw uint8. Four 3x3 conv layers (32 ch, stride 2 then 1,1,1) + ReLU, flatten, a
    Linear to latent_dim, then the SAME latent_norm switch as the MLP Encoder. Exposes
    pre_simnorm() so WorldModel can use it interchangeably with Encoder (the predictor adds its
    residual delta in pre-norm space). Accepts uint8 OR float input.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        latent_dim: int,
        channels: int = 32,
        simnorm_groups: int = 8,
        latent_norm: str = "simnorm",
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = (int(c), int(h), int(w))
        self.convs = nn.Sequential(
            nn.Conv2d(c, channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.convs(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.proj = nn.Linear(n_flat, latent_dim)
        self.simnorm = _latent_norm(latent_norm, simnorm_groups)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """uint8 (or any) pixel input -> centered float in [-0.5, 0.5]."""
        return obs.float() / 255.0 - 0.5

    def pre_simnorm(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._normalize(obs)
        h = self.convs(x).flatten(1)
        return self.proj(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.simnorm(self.pre_simnorm(obs))


class Decoder(nn.Module):
    """Pixel decoder (RECONSTRUCTION baseline only): transpose-conv mirror of CNNEncoder.

    latent -> Linear -> conv feature map -> transpose-conv tower -> reconstructed frames in
    [-0.5, 0.5] (the SAME centered space the CNNEncoder normalizes uint8 pixels into, so the
    recon MSE is computed against `obs/255 - 0.5`). This Dreamer-style observation model is
    built ONLY for the reconstruction arm — the JEPA arm (latent consistency) never owns one.

    The transpose stack reverses the encoder's (stride 2, 1, 1, 1) conv tower: three stride-1
    transpose convs then one stride-2 transpose conv, with `output_padding` solved so the
    output spatial size matches obs_shape exactly (works for arbitrary H, W, not just 84).
    """

    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple[int, int, int] = (9, 84, 84),
        channels: int = 32,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = (int(c), int(h), int(w))
        # probe the encoder conv tower to recover its output feature-map size (kernel/stride
        # arithmetic mirrored): four 3x3 convs, the first stride 2 then three stride 1.
        with torch.no_grad():
            probe = nn.Sequential(
                nn.Conv2d(c, channels, 3, stride=2),
                nn.Conv2d(channels, channels, 3, stride=1),
                nn.Conv2d(channels, channels, 3, stride=1),
                nn.Conv2d(channels, channels, 3, stride=1),
            )(torch.zeros(1, c, h, w))
        _, _, fh, fw = probe.shape
        self.feat_shape = (channels, int(fh), int(fw))
        # after the three stride-1 transpose convs the map grows by +2 each (kernel 3, pad 0)
        mid_h, mid_w = fh + 6, fw + 6
        # the final stride-2 transpose conv: out = (mid-1)*2 + 3 + output_padding; solve for it
        out_pad_h = h - ((mid_h - 1) * 2 + 3)
        out_pad_w = w - ((mid_w - 1) * 2 + 3)
        if not (0 <= out_pad_h <= 1 and 0 <= out_pad_w <= 1):
            raise ValueError(f"cannot mirror obs_shape {obs_shape} with this conv tower")
        self.proj = nn.Linear(latent_dim, channels * fh * fw)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                channels, c, 3, stride=2, output_padding=(out_pad_h, out_pad_w)
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., latent_dim) -> reconstructed frames (..., C, H, W) centered in [-0.5, 0.5]."""
        lead = z.shape[:-1]
        ch, fh, fw = self.feat_shape
        h = self.proj(z).reshape(-1, ch, fh, fw)
        img = self.deconvs(h)
        return img.reshape(*lead, *self.obs_shape)


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


class PredictorEnsemble(nn.Module):
    """R18: N INDEPENDENT latent predictors g_phi^{1..N} for epistemic disagreement on a
    decoder-free JEPA. Each head is a full `Predictor` with its own random init + parameters, so
    they can be trained with independent SGD. They SHARE the (external) encoder + EMA target —
    that sharing is exactly what the R18 calibration diagnostic interrogates: does a single smooth
    EMA target collapse cross-head disagreement, or does it stay correlated with prediction error?

    forward -> stacked predictions (N, batch, latent). disagreement -> per-sample cross-head
    variance of the predicted next latent (the candidate epistemic-uncertainty / intrinsic-reward
    signal). No encoder here; callers pass pre-SimNorm latents (the same interface as Predictor).
    """

    def __init__(
        self,
        n_heads: int,
        latent_dim: int,
        act_dim: int,
        hidden: int = 256,
        action_head_dim: int = 64,
        simnorm_groups: int = 8,
        latent_norm: str = "simnorm",
    ) -> None:
        super().__init__()
        if n_heads < 2:
            raise ValueError(f"need >=2 heads for disagreement, got {n_heads}")
        self.n_heads = int(n_heads)
        self.heads = nn.ModuleList(
            Predictor(latent_dim, act_dim, hidden, action_head_dim, simnorm_groups, latent_norm)
            for _ in range(n_heads)
        )

    def forward(self, z_pre: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns (n_heads, batch, latent): every head's predicted next SimNorm latent."""
        return torch.stack([h(z_pre, action) for h in self.heads], dim=0)

    def mean_prediction(self, z_pre: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Ensemble-mean predicted next latent, (batch, latent)."""
        return self(z_pre, action).mean(dim=0)

    def disagreement(self, z_pre: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-sample epistemic disagreement: variance across heads of the predicted next latent,
        averaged over latent dims. Returns (batch,). Zero iff all heads agree exactly."""
        preds = self(z_pre, action)  # (N, B, LD)
        return preds.var(dim=0, unbiased=False).mean(dim=-1)  # (B,)


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


class QuasimetricHead(nn.Module):
    """R22 asymmetric QUASIMETRIC on latents (QRL/IQE-style) — a learned reachability distance that
    fixes what L2 on the SimNorm latent cannot (R21: latent-L2 vs true-distance rho=0.23).

    A potential map v: latent -> R^k (small MLP); d(z_s, z_g) = sum_k relu(v(z_g)_k - v(z_s)_k).
    Satisfies d>=0, d(z,z)=0, and the TRIANGLE INEQUALITY by construction (subadditivity of relu:
    relu(a+b) <= relu(a)+relu(b)); ASYMMETRIC (d(s,g) != d(g,s)) — the right inductive bias for a
    steps-to-go goal metric. Trained on a FROZEN control encoder (local hinge d<=gap on real pairs +
    contrastive spread on random pairs), so the working control representation is never disturbed.
    """

    def __init__(self, latent_dim: int, k: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, k),
        )

    def potentials(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (..., k)

    def forward(self, z_s: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Quasimetric distance s->g, shape = batch dims of z_s/z_g (broadcast)."""
        return torch.relu(self.potentials(z_g) - self.potentials(z_s)).sum(dim=-1)
