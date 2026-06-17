"""Acceptance metrics — the SOLE quantitative gate for "the robot is controlled".

Hardened against self-deception (round-1 adversarial review): the cosine is
centered (offset cannot fool it), latent fidelity gates on TARGET-ENCODER collapse
(a model that copies a collapsed EMA target must NOT score a pass), collapse is
detected across both dimensional and point modes, and every raw metric has a
CODIFIED boolean acceptance (`is_collapsed`, `fidelity_ok`) with version-controlled
thresholds — so the bar is code, not a human eyeballing numbers each round.

Scale note: latent thresholds below are calibrated for SimNorm-bounded latents
(values in simplices, O(0.1) healthy per-dim std). Revisit with real model data in
round 2; structural signals (effective rank, participation ratio) are scale-invariant
and carry the load when scale is uncertain.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

# --- codified acceptance thresholds (revisit in round 2 with real model data) ---
COLLAPSE_EFF_RANK_MIN = 3.0  # ABSOLUTE effective-rank floor (not fraction-of-d): a latent
# spanning < ~3 effective dims is degenerate. Absolute so an OVERSIZED latent for a low-dim
# state is not false-flagged — the R3 lesson: a 256-d latent that controlled cheetah (17-d state)
# to return 557 was wrongly flagged "collapsed" by the old fraction-of-d rule. The gold signals
# for representation quality remain real control return + obs<->latent correlation (logged by
# the trainer); is_collapsed is only a cheap guard against true degeneracy.
COLLAPSE_STD_FLOOR = 0.02  # per-dim std below this -> point collapse
COLLAPSE_PAIRWISE_FLOOR = 0.05  # mean pairwise dist below this -> point collapse
FIDELITY_CENTERED_COS_MIN = 0.6  # centered cosine at k=1 must clear this
FIDELITY_NRMSE_MAX = 1.0  # collapse-aware NRMSE at k=1 must stay under this


def aggregate_returns(per_seed_means: Sequence[float]) -> dict:
    """Cross-seed aggregation. A single seed is flagged low_confidence — NEVER an acceptance."""
    a = np.asarray(per_seed_means, dtype=float)
    n = len(a)
    std = float(a.std(ddof=1)) if n > 1 else 0.0  # sample std, consistent with ci95
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": float(a.mean()),
        "std": std,
        "ci95": ci95,
        "n_seeds": n,
        "per_seed": a.tolist(),
        "low_confidence": n < 2,
    }


def collapse_diagnostics(Z: np.ndarray, dead_thresh: float = COLLAPSE_STD_FLOOR) -> dict:
    """Representation-collapse health on latents Z [N, d].

    Reports BOTH collapse modes: dimensional (latents on a low-dim subspace — caught by
    effective_rank / participation_ratio, scale-invariant) and point (latents -> a
    constant — caught by per_dim_std / mean_pairwise_dist / total_variance, scale-aware).
    Also doubles as the live VICReg training monitor.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be [N, d], got shape {Z.shape}")
    N, d = Z.shape

    per_dim_std = Z.std(axis=0)
    median_std = float(np.median(per_dim_std))
    dead_frac = float((per_dim_std < dead_thresh).mean())
    # scale-INVARIANT dead-dim test: dims tiny relative to the typical dim
    rel_dead_frac = float((per_dim_std < 0.05 * (median_std + 1e-12)).mean())

    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        eff_rank = 1.0  # all singular values zero: total collapse, trivially rank-1
    else:
        p = s / s.sum()
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    cov = np.atleast_2d(np.cov(Z, rowvar=False)) if N > 1 else np.zeros((d, d))
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    pr = float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0
    total_variance = float(lam.sum())

    offdiag = cov.copy()
    np.fill_diagonal(offdiag, 0.0)
    offdiag_mean = float(np.abs(offdiag).mean()) if d > 1 else 0.0

    m = min(N, 256)
    sub = Z if N <= m else Z[np.random.default_rng(0).choice(N, m, replace=False)]
    mpd = float(pdist(sub).mean()) if m > 1 else 0.0

    return {
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_median": median_std,
        "total_variance": total_variance,
        "dead_dim_fraction": dead_frac,
        "rel_dead_dim_fraction": rel_dead_frac,
        "effective_rank": eff_rank,
        "rank_fraction": eff_rank / d,
        "participation_ratio": pr,
        "pr_fraction": pr / d,
        "offdiag_cov_mean": offdiag_mean,
        "mean_pairwise_dist": mpd,
        "latent_dim": d,
        "n": N,
    }


def is_collapsed(
    diag: dict,
    *,
    eff_rank_min: float = COLLAPSE_EFF_RANK_MIN,
    std_floor: float = COLLAPSE_STD_FLOOR,
    pairwise_floor: float = COLLAPSE_PAIRWISE_FLOOR,
) -> bool:
    """Codified collapse verdict — fires only on TRUE degeneracy, NOT on an oversized-but-healthy
    latent. Dimensional collapse: effective_rank below an ABSOLUTE floor (fraction-of-d over-fires
    when latent_dim >> intrinsic state dim — R3 proved this). Point collapse: per-dim std or mean
    pairwise distance near zero. Representation *quality* is judged by control return + obs<->latent
    correlation (the trainer logs both), not by this cheap catastrophic-degeneracy guard.
    """
    return bool(
        diag["effective_rank"] < eff_rank_min  # dimensional collapse (absolute)
        or diag["per_dim_std_mean"] < std_floor  # point collapse (scale)
        or diag["mean_pairwise_dist"] < pairwise_floor  # point collapse (concentration)
    )


def latent_rollout_fidelity(z_hat: np.ndarray, z_true: np.ndarray) -> dict:
    """Open-loop latent-rollout fidelity vs the real sim, collapse-aware by construction.

    z_hat, z_true: [K, N, d] — predicted vs EMA-target latents at horizon k=1..K under the
    TRUE action sequence. Per horizon reports:
      - cosine: raw per-sample cosine (kept for reference; offset-foolable, do NOT gate on it)
      - centered_cosine: cosine after subtracting the per-k batch mean — a constant/mean
        predictor scores ~0 here, so this measures real per-sample prediction
      - nrmse: ||z_hat - z_true|| / (sqrt(d) * std(z_true)), scale/collapse-aware
      - target_collapsed: collapse verdict on z_true itself — if the target encoder collapsed,
        a perfect-looking match is meaningless and this flag kills the pass
    """
    z_hat = np.asarray(z_hat, dtype=float)
    z_true = np.asarray(z_true, dtype=float)
    if z_hat.shape != z_true.shape or z_hat.ndim != 3:
        raise ValueError(f"need matching [K,N,d], got {z_hat.shape} vs {z_true.shape}")

    out = {}
    for k in range(z_hat.shape[0]):
        a, b = z_hat[k], z_true[k]
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cosine = float((an * bn).sum(axis=1).mean())

        ac = a - a.mean(axis=0, keepdims=True)
        bc = b - b.mean(axis=0, keepdims=True)
        acn = ac / (np.linalg.norm(ac, axis=1, keepdims=True) + 1e-8)
        bcn = bc / (np.linalg.norm(bc, axis=1, keepdims=True) + 1e-8)
        centered_cosine = float((acn * bcn).sum(axis=1).mean())

        denom = np.sqrt(b.shape[1]) * (b.std(axis=0).mean() + 1e-8)
        nrmse = float(np.linalg.norm(a - b, axis=1).mean() / denom)

        tdiag = collapse_diagnostics(b)
        out[k + 1] = {
            "cosine": cosine,
            "centered_cosine": centered_cosine,
            "nrmse": nrmse,
            "target_eff_rank": tdiag["effective_rank"],
            "target_rank_fraction": tdiag["rank_fraction"],
            "target_dead_dim_fraction": tdiag["dead_dim_fraction"],
            "target_collapsed": is_collapsed(tdiag),
        }
    return out


def fidelity_ok(
    fid: dict,
    *,
    k: int = 1,
    centered_cos_min: float = FIDELITY_CENTERED_COS_MIN,
    nrmse_max: float = FIDELITY_NRMSE_MAX,
) -> bool:
    """Codified fidelity verdict at horizon k. A collapsed TARGET can never pass."""
    if not fid:
        return False
    if k not in fid:
        k = min(fid)
    f = fid[k]
    return bool(
        (not f["target_collapsed"])
        and f["centered_cosine"] >= centered_cos_min
        and f["nrmse"] <= nrmse_max
    )


def plot_sample_efficiency(
    curves: dict[str, tuple[Sequence, Sequence, Sequence]],
    path: str | Path,
    title: str = "sample efficiency",
    baselines: dict[str, tuple[Sequence, Sequence]] | None = None,
) -> str:
    """curves: name -> (steps, mean, std). baselines: name -> (steps, vals). TD-MPC2 axes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (steps, mean, std) in curves.items():
        s, m, sd = np.asarray(steps), np.asarray(mean), np.asarray(std)
        ax.plot(s, m, label=name)
        ax.fill_between(s, m - sd, m + sd, alpha=0.2)
    if baselines:
        for name, (steps, vals) in baselines.items():
            ax.plot(np.asarray(steps), np.asarray(vals), "--", label=f"{name} (baseline)")
    ax.set_xlabel("env steps")
    ax.set_ylabel("episode return")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=90)
    plt.close(fig)
    return str(path)
