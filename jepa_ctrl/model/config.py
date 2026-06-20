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
    encoder_type: str = "mlp"  # "mlp" (default, state input) | "cnn" (pixel frame-stack)
    obs_shape: tuple[int, int, int] | None = None  # (C,H,W) required when encoder_type=="cnn"
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
    latent_norm: str = "simnorm"  # "simnorm" (default) | "none" (RAW latent, for the SIGReg arm)
    # R18+: N independent predictor heads for epistemic disagreement (Plan2Explore-style intrinsic
    # reward). Default 1 = no ensemble (byte-identical to every prior run). >=2 builds a
    # PredictorEnsemble trained alongside the main predictor on DETACHED latents (it MEASURES
    # uncertainty, it does not reshape the representation — so the encoder is identical across the
    # reward-MPC and disagreement-exploration arms; only the planning objective differs).
    n_pred_heads: int = 1
    # R19 leg3 (Plan2Explore proper): an INTRINSIC value head trained with ensemble disagreement as
    # the reward, so the disagreement-MPC bootstraps long-horizon "expected future novelty" at the
    # plan horizon instead of greedily summing 1-step disagreement. Requires n_pred_heads>=2.
    explore_value: bool = False

    def __post_init__(self) -> None:
        if self.latent_norm not in ("simnorm", "none"):
            raise ValueError(f"latent_norm must be 'simnorm' or 'none', got {self.latent_norm!r}")
        if self.encoder_type not in ("mlp", "cnn"):
            raise ValueError(f"encoder_type must be 'mlp' or 'cnn', got {self.encoder_type!r}")
        if self.encoder_type == "cnn" and self.obs_shape is None:
            raise ValueError("encoder_type 'cnn' requires obs_shape=(C,H,W)")
        if self.latent_dim % self.simnorm_groups != 0:
            raise ValueError(
                f"latent_dim {self.latent_dim} not divisible by "
                f"simnorm_groups {self.simnorm_groups}"
            )
        if self.n_pred_heads < 1:
            raise ValueError(f"n_pred_heads must be >= 1, got {self.n_pred_heads}")
        if self.explore_value and self.n_pred_heads < 2:
            raise ValueError("explore_value requires n_pred_heads>=2 (disagreement is its reward)")
        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")
        if self.vmax <= self.vmin:
            raise ValueError(f"vmax {self.vmax} must exceed vmin {self.vmin}")
