from __future__ import annotations

import torch

from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.trainer import TrainConfig, Trainer


def _trainer():
    wm = WorldModel(ModelConfig(obs_dim=5, act_dim=1, latent_dim=64))
    cfg = TrainConfig(explore_std=0.3, explore_std_end=0.05, explore_anneal_steps=1000)
    return Trainer(wm, cfg, act_low=torch.full((1,), -1.0), act_high=torch.full((1,), 1.0))


def test_explore_std_anneals_linearly_then_clamps():
    tr = _trainer()
    tr.step = 0
    assert abs(tr._explore_std() - 0.30) < 1e-6
    tr.step = 500
    assert abs(tr._explore_std() - 0.175) < 1e-6  # halfway
    tr.step = 1000
    assert abs(tr._explore_std() - 0.05) < 1e-6  # reaches floor
    tr.step = 5000
    assert abs(tr._explore_std() - 0.05) < 1e-6  # clamped at floor
