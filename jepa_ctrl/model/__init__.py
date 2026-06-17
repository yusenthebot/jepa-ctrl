"""Differentiable action-conditioned JEPA world-model core (state input, no decoder).

Locked architecture (round 2): SimNorm encoder f_theta + EMA target f_xi + residual
action-conditioned predictor g_phi + distributional reward/value heads. Trained with
multi-step latent consistency (consistency_coef dominates) and subordinate reward/value
grounding; planned with latent-space MPPI (planner lands in a later file).
"""

from __future__ import annotations

from .buffer import ReplayBuffer
from .config import ModelConfig
from .jepa_controller import JepaController
from .mppi import MPPIConfig, MPPIPlanner, eval_mppi, train_mppi
from .nets import DistHead, Encoder, Predictor, symexp, symlog
from .simnorm import SimNorm
from .trainer import TrainConfig, Trainer
from .world_model import WorldModel

__all__ = [
    "ModelConfig",
    "SimNorm",
    "Encoder",
    "Predictor",
    "DistHead",
    "WorldModel",
    "symlog",
    "symexp",
    "ReplayBuffer",
    "MPPIConfig",
    "MPPIPlanner",
    "train_mppi",
    "eval_mppi",
    "Trainer",
    "TrainConfig",
    "JepaController",
]
