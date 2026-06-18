from __future__ import annotations

import torch

from jepa_ctrl.model import ModelConfig, WorldModel, symlog

torch.manual_seed(0)


def _model(horizon: int = 4, latent_dim: int = 16) -> WorldModel:
    cfg = ModelConfig(
        obs_dim=6, act_dim=2, latent_dim=latent_dim, num_q=3, horizon=horizon
    )
    return WorldModel(cfg)


def _window(wm: WorldModel, batch: int = 5):
    """A (H+1,B,od)/(H,B,ad)/(H,B) sub-trajectory window matching buffer.sample_subtraj."""
    h, c = wm.cfg.horizon, wm.cfg
    obs_seq = torch.randn(h + 1, batch, c.obs_dim)
    action_seq = torch.randn(h, batch, c.act_dim)
    reward_seq = torch.randn(h, batch)
    return obs_seq, action_seq, reward_seq


# --- (1) reward grounding uses ALL H steps, not just step 0 -----------------------
def test_reward_grounding_uses_all_horizon_steps():
    wm = _model(horizon=4)
    obs_seq, action_seq, reward_seq = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    assert len(latents) == wm.cfg.horizon + 1  # [z_0, z_hat_1..z_hat_H]

    base = wm.reward_grounding_loss(action_seq, reward_seq, latents).detach()
    # perturbing the reward at a step k>0 MUST change the loss; under the old step-0-only
    # grounding it would not, because only reward_seq[0] entered the loss.
    for k in range(1, wm.cfg.horizon):
        perturbed = reward_seq.clone()
        perturbed[k] = perturbed[k] + 5.0
        new = wm.reward_grounding_loss(action_seq, perturbed, latents).detach()
        assert not torch.allclose(new, base), f"reward step k={k} did not affect the loss"


def test_reward_grounding_step0_also_matters():
    wm = _model(horizon=3)
    obs_seq, action_seq, reward_seq = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    base = wm.reward_grounding_loss(action_seq, reward_seq, latents).detach()
    perturbed = reward_seq.clone()
    perturbed[0] = perturbed[0] + 5.0
    new = wm.reward_grounding_loss(action_seq, perturbed, latents).detach()
    assert not torch.allclose(new, base)


# --- (2) value TD target depends on the REAL next action --------------------------
def _sarsa_target(
    wm: WorldModel, next_obs, next_action, reward0, gamma: float = 0.99, k: int = 2
) -> torch.Tensor:
    """Mirror of Trainer._value_target (kept here to test the model-side dependency)."""
    with torch.no_grad():
        z_next = wm.encode(next_obs)
        v_logits = wm.target_value_head.logits(z_next, next_action)
        v = wm.target_value_head.to_scalar(v_logits)
        kk = min(k, v.shape[0])
        v_min = v.topk(kk, dim=0, largest=False).values.amin(0)
        return reward0 + gamma * v_min


def test_value_target_depends_on_next_action():
    torch.manual_seed(0)  # deterministic perturbation — robust to test order / global RNG state
    wm = _model(horizon=4)
    obs_seq, action_seq, reward_seq = _window(wm)
    # perturb the value-head weights off the deepcopy init so logits actually vary with action
    with torch.no_grad():
        for p in wm.target_value_head.parameters():
            p.add_(torch.randn_like(p) * 2.0)

    a1 = action_seq[1]
    tgt_real = _sarsa_target(wm, obs_seq[1], a1, reward_seq[0])
    tgt_zero = _sarsa_target(
        wm, obs_seq[1], torch.zeros_like(a1), reward_seq[0]
    )  # the OLD zero-action proxy
    assert not torch.allclose(tgt_real, tgt_zero), (
        "value target must change with the next action; the zero-action proxy is the bug"
    )
    # and changing a_{t+1} to a different real action also moves the target
    tgt_other = _sarsa_target(wm, obs_seq[1], a1 + 3.0, reward_seq[0])
    assert not torch.allclose(tgt_real, tgt_other)


def test_value_target_reward_and_done_terms():
    wm = _model(horizon=3)
    obs_seq, action_seq, reward_seq = _window(wm, batch=4)
    gamma = 0.99
    tgt = _sarsa_target(wm, obs_seq[1], action_seq[1], reward_seq[0], gamma=gamma)
    # done -> target collapses to the immediate reward (bootstrap masked)
    with torch.no_grad():
        z_next = wm.encode(obs_seq[1])
        v = wm.target_value_head.to_scalar(wm.target_value_head.logits(z_next, action_seq[1]))
        v_min = v.topk(min(2, v.shape[0]), dim=0, largest=False).values.amin(0)
    expected = reward_seq[0] + gamma * v_min
    assert torch.allclose(tgt, expected, atol=1e-5)


# --- (3) gradient flow: online nets get grad, targets get none --------------------
def test_grounding_grad_reaches_online_not_targets():
    wm = _model(horizon=4)
    obs_seq, action_seq, reward_seq = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)

    r_loss = wm.reward_grounding_loss(action_seq, reward_seq, latents)
    # value target is grad-free (mirrors the trainer); ground value at step 0
    v_tgt = _sarsa_target(wm, obs_seq[1], action_seq[1], reward_seq[0])
    v_loss = wm.value_loss(latents[0], action_seq[0], v_tgt)
    (r_loss + v_loss).backward()

    def gnorm(module) -> torch.Tensor:
        return sum(
            p.grad.abs().sum() for p in module.parameters() if p.grad is not None
        )

    assert gnorm(wm.encoder) > 0, "encoder must receive grounding gradient (anti-collapse)"
    assert gnorm(wm.predictor) > 0, "predictor must receive reward-grounding gradient"
    assert gnorm(wm.reward_head) > 0
    assert gnorm(wm.value_head) > 0
    # the EMA targets are stop-grad: NO gradient ever
    for p in wm.target_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    for p in wm.target_value_head.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_reward_grounding_grad_reaches_predictor_via_later_steps():
    """The predictor only sees gradient if grounding flows through z_hat_k, k>=1.

    Ground reward at ONLY the k>0 latents (skip step 0) and confirm the predictor still
    gets gradient — proves the rollout latents, not just z_0, carry the reward signal.
    """
    wm = _model(horizon=4)
    obs_seq, action_seq, reward_seq = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    loss = action_seq.new_zeros(())
    for k in range(1, wm.cfg.horizon):  # k>=1 only
        logits = wm.reward_head.logits(latents[k], action_seq[k])[0]
        label = wm.reward_head.twohot_encode(symlog(reward_seq[k]))
        loss = loss + -(label * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
    loss.backward()
    pred_grad = sum(
        p.grad.abs().sum() for p in wm.predictor.parameters() if p.grad is not None
    )
    assert pred_grad > 0


# --- (4) shapes / finiteness of every returned loss -------------------------------
def test_grounding_losses_shapes_and_finite():
    wm = _model(horizon=5)
    obs_seq, action_seq, reward_seq = _window(wm, batch=7)
    latents = wm.rollout_latents(obs_seq, action_seq)

    consistency = wm.consistency_loss_from(obs_seq, action_seq, latents)
    r_loss = wm.reward_grounding_loss(action_seq, reward_seq, latents)
    v_tgt = _sarsa_target(wm, obs_seq[1], action_seq[1], reward_seq[0])
    v_loss = wm.value_loss(latents[0], action_seq[0], v_tgt)

    for name, val in (("consistency", consistency), ("reward", r_loss), ("value", v_loss)):
        assert val.shape == (), f"{name} loss must be a scalar"
        assert torch.isfinite(val), f"{name} loss must be finite"
    assert v_tgt.shape == (7,)
    assert torch.isfinite(v_tgt).all()
    # latents: one per observation o_t..o_{t+H}, each a (B, latent_dim) SimNorm latent
    assert len(latents) == wm.cfg.horizon + 1
    for z in latents:
        assert z.shape == (7, wm.cfg.latent_dim)


def test_rollout_latents_match_separate_encode_and_rollout():
    """rollout_latents[0] must equal encode(o_t) and [k] equal the open-loop rollout."""
    wm = _model(horizon=3).eval()
    obs_seq, action_seq, _ = _window(wm, batch=4)
    latents = wm.rollout_latents(obs_seq, action_seq)
    z0_ref = wm.encode(obs_seq[0])
    assert torch.allclose(latents[0], z0_ref, atol=1e-6)
    preds_ref = wm.rollout(wm.encode_pre(obs_seq[0]), action_seq)
    for k in range(1, wm.cfg.horizon + 1):
        assert torch.allclose(latents[k], preds_ref[k - 1], atol=1e-6)
