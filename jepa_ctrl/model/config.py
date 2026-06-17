from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Immutable config for the action-conditioned JEPA world model (locked architecture).

    Defaults are the cheetah/walker sizing (latent_dim 256); use latent_dim 128 for
    cartpole/reacher. Coefficients follow TD-MPC2-grounded JEPA: consistency dominates,
    reward/value are subordinate grounding signals.
    """

    obs_dim: int
    act_dim: int
    latent_dim: int = 256
    simnorm_groups: int = 8
    horizon: int = 5
    rho: float = 0.5
    consistency_coef: float = 20.0
    rv_coef: float = 0.1
    bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    enc_lr_scale: float = 0.3
    ema_tau: float = 0.99
    value_tau: float = 0.01
    num_q: int = 5
    enc_hidden: int = 256
    action_head_dim: int = 64
    pred_hidden: int = 256

    def __post_init__(self) -> None:
        if self.latent_dim % self.simnorm_groups != 0:
            raise ValueError(
                f"latent_dim {self.latent_dim} not divisible by "
                f"simnorm_groups {self.simnorm_groups}"
            )
        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")
        if self.vmax <= self.vmin:
            raise ValueError(f"vmax {self.vmax} must exceed vmin {self.vmin}")
