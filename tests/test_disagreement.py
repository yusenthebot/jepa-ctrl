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


def _cfg(n_heads: int, latent_norm: str = "simnorm", explore_value: bool = False) -> ModelConfig:
    return ModelConfig(
        obs_dim=OBS, act_dim=ACT, latent_dim=LD, simnorm_groups=8, horizon=3,
        enc_hidden=64, pred_hidden=64, action_head_dim=16, n_pred_heads=n_heads,
        latent_norm=latent_norm, explore_value=explore_value,
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
    MPPIConfig(objective="hybrid")
    try:
        MPPIConfig(objective="bogus")
        raise AssertionError("expected ValueError for bad objective")
    except ValueError:
        pass


def test_hybrid_score_is_reward_plus_beta_times_disagreement():
    """hybrid score = extrinsic reward score + explore_beta * intrinsic disagreement score. At
    beta=0 it must equal the pure reward score; the trainer anneals beta 1->0 (explore->exploit)."""
    wm = WorldModel(_cfg(5))
    cfg = MPPIConfig(horizon=3, iters=1, num_samples=4, num_elites=2, objective="hybrid")
    planner = MPPIPlanner(wm, cfg, -torch.ones(ACT), torch.ones(ACT), device="cpu")
    z0 = wm.encode_pre(torch.randn(1, OBS)).expand(4, -1).contiguous()
    actions = torch.randn(3, 4, ACT).clamp(-1, 1)
    r = planner._score_reward(z0, actions)
    d = planner._score_disagreement(z0, actions)
    planner.explore_beta = 0.7
    got = planner._score(z0, actions)
    assert torch.allclose(got, r + 0.7 * d, atol=1e-5)
    planner.explore_beta = 0.0  # fully exploit => identical to pure reward objective
    assert torch.allclose(planner._score(z0, actions), r, atol=1e-5)


def test_trainer_explore_beta_anneals():
    from jepa_ctrl.model.trainer import TrainConfig, Trainer
    wm = WorldModel(_cfg(5))
    tc = TrainConfig(explore_beta_start=1.0, explore_beta_end=0.0, explore_beta_anneal_steps=100,
                     seed_steps=0)
    tr = Trainer(wm, tc, -torch.ones(ACT), torch.ones(ACT), device="cpu")
    tr.step = 0
    assert abs(tr._explore_beta() - 1.0) < 1e-6
    tr.step = 50
    assert abs(tr._explore_beta() - 0.5) < 1e-6
    tr.step = 100
    assert abs(tr._explore_beta() - 0.0) < 1e-6
    tr.step = 999  # clamped
    assert abs(tr._explore_beta() - 0.0) < 1e-6


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


def test_explore_value_head_built_only_with_flag_and_requires_ensemble():
    assert WorldModel(_cfg(5)).explore_value_head is None  # flag off
    wm = WorldModel(_cfg(5, explore_value=True))
    assert wm.explore_value_head is not None and wm.target_explore_value_head is not None
    # explore_value with n_heads<2 is rejected at config construction
    try:
        _cfg(1, explore_value=True)
        raise AssertionError("expected ValueError: explore_value needs n_pred_heads>=2")
    except ValueError:
        pass


def test_normalized_disagreement_divides_by_scale():
    wm = WorldModel(_cfg(5))
    z, a = torch.randn(8, LD), torch.randn(8, ACT)
    raw = wm.ensemble_disagreement(z, a)
    wm.disag_scale.fill_(4.0)
    norm = wm.normalized_disagreement(z, a)
    assert torch.allclose(norm, raw / (4.0 + 1e-8), atol=1e-6)


def test_update_disag_scale_is_ema():
    wm = WorldModel(_cfg(5))
    wm.disag_scale.fill_(1.0)
    wm.update_disag_scale(torch.full((8,), 5.0), tau=0.9)
    assert abs(wm.disag_scale.item() - (0.9 * 1.0 + 0.1 * 5.0)) < 1e-6


def test_intrinsic_value_loss_updates_only_explore_value_head():
    """measure-don't-reshape: the intrinsic value loss must not leak gradient into the encoder,
    main predictor, or the disagreement ensemble — only into the explore value head."""
    wm = WorldModel(_cfg(4, explore_value=True))
    z0 = wm.encode(torch.randn(8, OBS))
    a0 = torch.randn(8, ACT)
    tgt = torch.randn(8)
    loss = wm.intrinsic_value_loss(z0.detach(), a0, tgt)
    assert loss.ndim == 0
    wm.zero_grad(set_to_none=True)
    loss.backward()
    def _has_grad(mod):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    assert not _has_grad(wm.encoder), "intrinsic value leaked grad into encoder"
    assert not _has_grad(wm.predictor), "intrinsic value leaked grad into main predictor"
    assert not _has_grad(wm.predictor_ensemble), "intrinsic value leaked grad into ensemble"
    assert _has_grad(wm.explore_value_head), "intrinsic value must train the explore value head"


def test_ema_update_moves_target_explore_value_head():
    wm = WorldModel(_cfg(5, explore_value=True))
    before = [p.clone() for p in wm.target_explore_value_head.parameters()]
    for p in wm.explore_value_head.parameters():  # perturb online head
        p.data.add_(1.0)
    wm.ema_update()
    moved = any(
        not torch.equal(b, a)
        for b, a in zip(before, wm.target_explore_value_head.parameters(), strict=True)
    )
    assert moved, "EMA must move the target explore value head toward the online head"


def test_disagreement_planner_with_intrinsic_value_bootstraps_terminal():
    """With an explore value head, the disagreement score = myopic sum + gamma^H * intrinsic value.
    Verify it equals the myopic score plus the terminal bootstrap term."""
    wm = WorldModel(_cfg(5, explore_value=True))
    cfg = MPPIConfig(horizon=3, iters=1, num_samples=4, num_elites=2, objective="disagreement")
    planner = MPPIPlanner(wm, cfg, -torch.ones(ACT), torch.ones(ACT), device="cpu")
    z0 = wm.encode_pre(torch.randn(1, OBS)).expand(4, -1).contiguous()
    actions = torch.randn(3, 4, ACT).clamp(-1, 1)
    got = planner._score(z0, actions)
    # manual myopic sum (normalized) + terminal intrinsic value bootstrap
    z_in = z0
    expect = torch.zeros(4)
    for hh in range(3):
        expect = expect + (cfg.gamma**hh) * wm.normalized_disagreement(z_in, actions[hh])
        z_in = wm.predictor(z_in, actions[hh])
    ev = wm.explore_value_head
    v_term = ev.to_scalar(ev.logits(z_in, actions[-1])).mean(0)
    expect = expect + (cfg.gamma**3) * v_term
    assert torch.allclose(got, expect, atol=1e-5)
    a = planner.plan(torch.randn(OBS))
    assert a.shape == (ACT,) and torch.isfinite(a).all()


def test_goal_objective_validation_and_requires_set_goal():
    MPPIConfig(objective="goal")
    wm = WorldModel(_cfg(5))
    planner = _planner(wm, "goal")
    try:
        planner._score(wm.encode_pre(torch.randn(2, OBS)), torch.randn(3, 2, ACT))
        raise AssertionError("expected RuntimeError before set_goal")
    except RuntimeError:
        pass


def test_set_goal_encodes_simnorm_latent():
    wm = WorldModel(_cfg(5))
    planner = _planner(wm, "goal")
    planner.set_goal(torch.randn(OBS))
    assert planner.goal_z is not None
    assert planner.goal_z.shape == (1, LD)


def test_goal_score_higher_when_rollout_nearer_goal():
    """The goal score must prefer trajectories whose rolled latents land nearer the goal latent.
    Build two action sets; set the goal to the terminal latent of set A -> A must outscore B."""
    wm = WorldModel(_cfg(5))
    planner = _planner(wm, "goal")
    z0 = wm.encode_pre(torch.randn(1, OBS)).expand(2, -1).contiguous()
    a_good = torch.zeros(3, 2, ACT)            # one deterministic trajectory (both rows identical)
    a_bad = torch.ones(3, 2, ACT)
    # goal = the SimNorm latent the GOOD actions actually reach at the final step
    z_good_final = wm.rollout(z0[:1], a_good[:, :1])[-1]  # (1, LD) SimNorm
    planner.goal_z = z_good_final
    s_good = planner._score(z0[:1], a_good[:, :1])
    s_bad = planner._score(z0[:1], a_bad[:, :1])
    assert s_good.item() > s_bad.item(), "goal score must reward reaching the goal latent"
    assert s_good.item() <= 1e-5, "score is negative distance; reaching the goal -> ~0 (max)"


def test_goal_planner_returns_valid_action():
    wm = WorldModel(_cfg(5))
    planner = _planner(wm, "goal")
    planner.set_goal(torch.randn(OBS))
    a = planner.plan(torch.randn(OBS))
    assert a.shape == (ACT,) and torch.isfinite(a).all()
    assert torch.all(a >= -1.0) and torch.all(a <= 1.0)


def test_disagreement_score_matches_manual_rollout():
    """_score_disagreement = discounted sum of per-step ensemble disagreement along the shared
    open-loop rollout. Verify against a hand rollout so the planner optimises the real signal."""
    wm = WorldModel(_cfg(5))
    cfg = MPPIConfig(horizon=3, iters=1, num_samples=4, num_elites=2, objective="disagreement")
    planner = MPPIPlanner(wm, cfg, -torch.ones(ACT), torch.ones(ACT), device="cpu")
    z0 = wm.encode_pre(torch.randn(1, OBS)).expand(4, -1).contiguous()
    actions = torch.randn(3, 4, ACT).clamp(-1, 1)
    got = planner._score(z0, actions)
    # manual: myopic normalized-disagreement sum (no explore value head here => no bootstrap)
    assert wm.explore_value_head is None
    z_in = z0
    expect = torch.zeros(4)
    for hh in range(3):
        expect = expect + (cfg.gamma**hh) * wm.normalized_disagreement(z_in, actions[hh])
        z_in = wm.predictor(z_in, actions[hh])
    assert torch.allclose(got, expect, atol=1e-5)
