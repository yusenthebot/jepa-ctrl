"""Differentiable action-conditioned JEPA world-model core (state input, no decoder).

Locked architecture (round 2): SimNorm encoder f_theta + EMA target f_xi + residual
action-conditioned predictor g_phi + distributional reward/value heads. Trained with
multi-step latent consistency (consistency_coef dominates) and subordinate reward/value
grounding; planned with latent-space MPPI (planner lands in a later file).
"""

from __future__ import annotations

from .buffer import PixelReplayBuffer, ReplayBuffer
from .config import ModelConfig
from .jepa_controller import JepaController
from .mppi import MPPIConfig, MPPIPlanner, eval_mppi, train_mppi
from .nets import (
    CNNEncoder,
    Decoder,
    DistHead,
    Encoder,
    InverseDynamicsHead,
    Predictor,
    symexp,
    symlog,
)
from .sigreg import sigreg_loss
from .simnorm import SimNorm
from .trainer import TrainConfig, Trainer
from .world_model import WorldModel

__all__ = [
    "ModelConfig",
    "SimNorm",
    "Encoder",
    "CNNEncoder",
    "Decoder",
    "Predictor",
    "DistHead",
    "InverseDynamicsHead",
    "WorldModel",
    "sigreg_loss",
    "symlog",
    "symexp",
    "ReplayBuffer",
    "PixelReplayBuffer",
    "MPPIConfig",
    "MPPIPlanner",
    "train_mppi",
    "eval_mppi",
    "Trainer",
    "TrainConfig",
    "JepaController",
]
