"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun 2511.08544).

The anti-collapse pressure that REPLACES SimNorm+EMA: instead of structurally bounding the
latent (a simplex) and stop-gradding an EMA target, SIGReg pushes the latent *distribution*
toward an isotropic standard Gaussian. It does so by random 1-D projections (slices) and an
Epps-Pulley empirical-characteristic-function goodness-of-fit to N(0, 1):

    For each unit direction v, p = Z @ v (the 1-D projected sample). Standardize p, then over a
    Gaussian-weighted grid of frequencies t measure the squared modulus of the difference between
    the empirical CF  phi_hat(t) = mean_j exp(i t p_j)  and the standard-normal CF  exp(-t^2/2):

        EP(p) = INT |phi_hat(t) - exp(-t^2/2)|^2 * w(t) dt,   w(t) = N(t; 0, 1).

Averaging EP over many slices estimates a distance to the isotropic Gaussian (Cramér-Wold:
all 1-D marginals Gaussian + isotropy => the joint is isotropic Gaussian). A collapsed (point
or low-rank) batch has 1-D marginals far from N(0, 1) on most slices, so SIGReg is large and
its gradient spreads the latent out — de-collapsing it WITHOUT a decoder, EMA, or task reward.

Fully vectorized over slices (no Python loop) so the per-step cost is a few matmuls — cheap
enough to run every training step inside the <2h laptop budget.
"""

from __future__ import annotations

import torch


def sigreg_loss(
    z: torch.Tensor,
    n_proj: int = 256,
    lam: float = 0.05,
    t_max: float = 5.0,
    n_t: int = 17,
) -> torch.Tensor:
    """Sliced isotropic-Gaussian regularizer on a latent batch.

    Args:
        z: (N, d) batch of latents (RAW — NOT SimNorm; SIGReg targets a Gaussian, not a simplex).
        n_proj: number of random unit-direction slices averaged over.
        lam: scaling coefficient.
        t_max, n_t: half-width and count of the symmetric frequency grid for the CF integral.

    Returns:
        Scalar = lam * mean_v EP(z @ v) — small when z's marginals are ~N(0,1) on every slice,
        large under point or low-rank collapse. Differentiable w.r.t. z (anti-collapse gradient).
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (N, d), got shape {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        raise ValueError(f"need >= 2 samples for SIGReg, got {n}")

    # random unit directions on the sphere S^{d-1}
    dirs = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    proj = z @ dirs  # (N, P)

    # standardize each projection over the batch -> the statistic measures SHAPE vs N(0,1)
    proj = proj - proj.mean(dim=0, keepdim=True)
    proj = proj / (proj.std(dim=0, unbiased=False, keepdim=True) + 1e-8)

    t = torch.linspace(-t_max, t_max, n_t, device=z.device, dtype=z.dtype)  # (T,)
    # empirical CF per slice: phi_hat(t) = mean_j exp(i t p_j), batched over slices P and freqs T
    phase = proj.unsqueeze(2) * t.view(1, 1, n_t)  # (N, P, T)
    re = torch.cos(phase).mean(dim=0)  # (P, T)
    im = torch.sin(phase).mean(dim=0)  # (P, T)

    target = torch.exp(-0.5 * t**2)  # standard-normal CF (real), (T,)
    sq_mod = (re - target) ** 2 + im**2  # |phi_hat - phi_N|^2, (P, T)
    weight = torch.exp(-0.5 * t**2)  # Gaussian weight w(t), (T,)
    ep = (sq_mod * weight).mean(dim=1)  # per-slice EP statistic, (P,)
    return lam * ep.mean()
