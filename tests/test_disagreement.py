"""R18+ Plan2Explore wiring: ensemble lives in the WorldModel, trains without reshaping the
shared representation, and the MPPI planner can plan on intrinsic disagreement instead of reward.
All sim-free CPU unit tests of the mechanics; the real sparse-task campaign verifies the behaviour.
"""
from __future__ import annotations

import torch

from jepa_ctrl.model.config import ModelConfig
from jepa_ctrl.model.mppi import MPPIConfig, MPPIPlanner
from jepa_ctrl.model.world_model import WorldModel

torch.manual_seed(0)

OBS, ACT, LD = 6, 2, 64


def _cfg(n_heads: int, latent_norm: str = "simnorm") -> ModelConfig:
    return ModelConfig(
        obs_dim=OBS, act_dim=ACT, latent_dim=LD, simnorm_groups=8, horizon=3,
        enc_hidden=64, pred_hidden=64, action_head_dim=16, n_pred_heads=n_heads,
        latent_norm=latent_norm,
    )


def test_ensemble_built_only_when_n_heads_ge_2():
    assert WorldModel(_cfg(1)).predictor_ensemble is None
    wm = WorldModel(_cfg(5))
    assert wm.predictor_ensemble is not None
    assert wm.predictor_ensemble.n_heads == 5


def test_disagreement_shape_and_nonneg():
    wm = WorldModel(_cfg(5))
    z_pre = torch.randn(8, LD)
    a = torch.randn(8, ACT)
    d = wm.ensemble_disagreement(z_pre, a)
    assert d.shape == (8,)
    assert torch.all(d >= 0)


def test_disagreement_raises_without_ensemble():
    wm = WorldModel(_cfg(1))
    try:
        wm.ensemble_disagreement(torch.randn(2, LD), torch.randn(2, ACT))
        raise AssertionError("expected RuntimeError without an ensemble")
    except RuntimeError:
        pass


def test_ensemble_loss_updates_only_ensemble_params():
    """The 'measure, don't reshape' invariant: ensemble_consistency_loss must NOT push gradient
    into the encoder or the main predictor — only into the ensemble heads. This is what keeps the
    representation identical across the reward-MPC and disagreement arms."""
    wm = WorldModel(_cfg(4))
    h = wm.cfg.horizon
    obs_seq = torch.randn(h + 1, 8, OBS)
    action_seq = torch.randn(h, 8, ACT)
    latents = wm.rollout_latents(obs_seq, action_seq)
    loss = wm.ensemble_consistency_loss_from(obs_seq, action_seq, latents)
    assert loss.ndim == 0 and loss.item() >= 0
    wm.zero_grad(set_to_none=True)
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.encoder.parameters())
    pred_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in wm.predictor.parameters()
    )
    ens_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in wm.predictor_ensemble.parameters()
    )
    assert not enc_grad, "ensemble loss leaked gradient into the encoder"
    assert not pred_grad, "ensemble loss leaked gradient into the main predictor"
    assert ens_grad, "ensemble loss must train the ensemble heads"


def test_mppi_objective_validation():
    MPPIConfig(objective="reward")
    MPPIConfig(objective="disagreement")
    try:
        MPPIConfig(objective="bogus")
        raise AssertionError("expected ValueError for bad objective")
    except ValueError:
        pass


def _planner(wm: WorldModel, objective: str) -> MPPIPlanner:
    cfg = MPPIConfig(horizon=3, iters=2, num_samples=64, num_elites=16, objective=objective)
    low = -torch.ones(ACT)
    high = torch.ones(ACT)
    return MPPIPlanner(wm, cfg, low, high, device="cpu")


def test_disagreement_planner_returns_valid_bounded_action():
    wm = WorldModel(_cfg(5))
    planner = _planner(wm, "disagreement")
    a = planner.plan(torch.randn(OBS))
    assert a.shape == (ACT,)
    assert torch.all(a >= -1.0) and torch.all(a <= 1.0)
    assert torch.isfinite(a).all()


def test_disagreement_score_matches_manual_rollout():
    """_score_disagreement = discounted sum of per-step ensemble disagreement along the shared
    open-loop rollout. Verify against a hand rollout so the planner optimises the real signal."""
    wm = WorldModel(_cfg(5))
    cfg = MPPIConfig(horizon=3, iters=1, num_samples=4, num_elites=2, objective="disagreement")
    planner = MPPIPlanner(wm, cfg, -torch.ones(ACT), torch.ones(ACT), device="cpu")
    z0 = wm.encode_pre(torch.randn(1, OBS)).expand(4, -1).contiguous()
    actions = torch.randn(3, 4, ACT).clamp(-1, 1)
    got = planner._score(z0, actions)
    # manual
    z_in = z0
    expect = torch.zeros(4)
    for hh in range(3):
        expect = expect + (cfg.gamma**hh) * wm.ensemble_disagreement(z_in, actions[hh])
        z_in = wm.predictor(z_in, actions[hh])
    assert torch.allclose(got, expect, atol=1e-5)
