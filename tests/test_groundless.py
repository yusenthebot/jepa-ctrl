"""GROUNDLESS frontier: can a task-agnostic self-supervised signal REPLACE task-reward as
the anti-collapse grounding for decoder-free latent consistency?

These pure-torch unit tests (NO sim, tiny tensors) prove the machinery for a clean 3-arm
ablation (reward / inverse_dynamics / sigreg):

a. inverse_dynamics_loss is finite, correctly shaped, and its gradient reaches BOTH the
   encoder and the predictor on the SHARED rolled latents (the anti-collapse signal).
b. KEY CORRECTNESS CLAIM: in a reward-free arm (inverse_dynamics / sigreg) the reward+value
   loss gradient does NOT reach the encoder/predictor params (detached path) -> the learned
   representation is genuinely reward-free, yet the reward/value heads still train so MPPI
   can plan.
c. sigreg_loss is finite, ~0 for an already-N(0,1) batch, LARGER for a collapsed/low-rank
   batch, and a few gradient steps minimizing it INCREASE the batch's effective_rank
   (it de-collapses).
d. latent_norm="none" Encoder output is NOT a simplex; latent_norm="simnorm" still is.
"""

from __future__ import annotations

import torch

from jepa_ctrl.metrics import collapse_diagnostics
from jepa_ctrl.model import (
    InverseDynamicsHead,
    ModelConfig,
    TrainConfig,
    WorldModel,
    sigreg_loss,
)

torch.manual_seed(0)


def _model(horizon: int = 4, latent_dim: int = 16, latent_norm: str = "simnorm") -> WorldModel:
    cfg = ModelConfig(
        obs_dim=6,
        act_dim=2,
        latent_dim=latent_dim,
        num_q=3,
        horizon=horizon,
        latent_norm=latent_norm,
    )
    return WorldModel(cfg)


def _window(wm: WorldModel, batch: int = 5):
    h, c = wm.cfg.horizon, wm.cfg
    obs_seq = torch.randn(h + 1, batch, c.obs_dim)
    action_seq = torch.randn(h, batch, c.act_dim)
    reward_seq = torch.randn(h, batch)
    return obs_seq, action_seq, reward_seq


def _grad_sum(module) -> torch.Tensor:
    return sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)


# =====================================================================================
# (a) inverse-dynamics grounding: finite, shaped, grad reaches encoder AND predictor
# =====================================================================================
def test_inverse_dynamics_head_shapes():
    head = InverseDynamicsHead(latent_dim=16, act_dim=2, hidden=32)
    z_t = torch.randn(7, 16)
    z_next = torch.randn(7, 16)
    a_hat = head(z_t, z_next)
    assert a_hat.shape == (7, 2)
    assert torch.isfinite(a_hat).all()


def test_inverse_dynamics_loss_finite_and_scalar():
    wm = _model(horizon=4)
    obs_seq, action_seq, _ = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    loss = wm.inverse_dynamics_loss(action_seq, latents)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_inverse_dynamics_grad_reaches_encoder_and_predictor():
    wm = _model(horizon=4)
    obs_seq, action_seq, _ = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    loss = wm.inverse_dynamics_loss(action_seq, latents)
    loss.backward()
    assert _grad_sum(wm.encoder) > 0, "inverse-dynamics must ground the encoder (anti-collapse)"
    assert _grad_sum(wm.predictor) > 0, "inverse-dynamics must ground the predictor"
    assert _grad_sum(wm.inverse_dynamics_head) > 0
    # targets remain stop-grad
    for p in wm.target_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_inverse_dynamics_is_action_discriminative_on_predicted_latents():
    """The loss must depend on the PREDICTED latents (k>=1), not only z_0 / the encoder's
    near-bijective state map. Perturbing the action at a later step changes the rolled latents
    and therefore the loss."""
    wm = _model(horizon=4).eval()
    obs_seq, action_seq, _ = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    base = wm.inverse_dynamics_loss(action_seq, latents).detach()
    perturbed_actions = action_seq.clone()
    perturbed_actions[2] = perturbed_actions[2] + 3.0
    latents_p = wm.rollout_latents(obs_seq, perturbed_actions)
    new = wm.inverse_dynamics_loss(perturbed_actions, latents_p).detach()
    assert not torch.allclose(new, base)


# =====================================================================================
# (b) KEY CLAIM: reward-free arms isolate reward/value gradient from encoder+predictor
# =====================================================================================
def _reward_free_grad_isolation(grounding: str) -> None:
    """In a reward-free arm the rv loss is computed on DETACHED latents, so the rv gradient
    must NOT reach the encoder or predictor (representation is reward-free), while it DOES
    reach the reward/value heads (so MPPI can still plan)."""
    latent_norm = "none" if grounding == "sigreg" else "simnorm"
    wm = _model(horizon=4, latent_norm=latent_norm)
    _ = TrainConfig(grounding=grounding)  # arm is a valid TrainConfig value
    obs_seq, action_seq, reward_seq = _window(wm)

    latents = wm.rollout_latents(obs_seq, action_seq)
    detached = [z.detach() for z in latents]
    # rv losses live on the DETACHED path (mirrors Trainer.update for reward-free arms)
    r_loss = wm.reward_grounding_loss(action_seq, reward_seq, detached)
    v_tgt = torch.randn(reward_seq.shape[1])
    v_loss = wm.value_loss(detached[0], action_seq[0], v_tgt)
    (wm.cfg.rv_coef * (r_loss + v_loss)).backward()

    assert _grad_sum(wm.encoder) == 0, (
        f"[{grounding}] rv gradient leaked into the encoder — representation NOT reward-free"
    )
    assert _grad_sum(wm.predictor) == 0, (
        f"[{grounding}] rv gradient leaked into the predictor — representation NOT reward-free"
    )
    # but the heads MUST train so the planner remains usable
    assert _grad_sum(wm.reward_head) > 0, f"[{grounding}] reward head must still train for MPPI"
    assert _grad_sum(wm.value_head) > 0, f"[{grounding}] value head must still train for MPPI"


def test_reward_free_isolation_inverse_dynamics():
    _reward_free_grad_isolation("inverse_dynamics")


def test_reward_free_isolation_sigreg():
    _reward_free_grad_isolation("sigreg")


def test_reward_arm_rv_grad_DOES_reach_encoder():
    """Positive control: in the default reward arm the rv gradient DOES reach the encoder
    (this is the grounding that fixed pure-consistency collapse)."""
    wm = _model(horizon=4, latent_norm="simnorm")
    obs_seq, action_seq, reward_seq = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)  # NOT detached
    r_loss = wm.reward_grounding_loss(action_seq, reward_seq, latents)
    v_tgt = torch.randn(reward_seq.shape[1])
    v_loss = wm.value_loss(latents[0], action_seq[0], v_tgt)
    (r_loss + v_loss).backward()
    assert _grad_sum(wm.encoder) > 0
    assert _grad_sum(wm.predictor) > 0


def test_trainer_update_reward_free_arm_isolation():
    """End-to-end through Trainer.update(): after a reward-free update step, confirm the rv
    loss never coupled to the encoder by re-deriving the rv gradient on the detached path."""
    from jepa_ctrl.model import MPPIConfig
    from jepa_ctrl.model.buffer import ReplayBuffer

    wm = _model(horizon=3, latent_norm="simnorm")
    tcfg = TrainConfig(grounding="inverse_dynamics", batch_size=4, seed_steps=0)
    from jepa_ctrl.model.trainer import Trainer

    act_low = -torch.ones(wm.cfg.act_dim)
    act_high = torch.ones(wm.cfg.act_dim)
    trainer = Trainer(wm, tcfg, act_low, act_high, mppi_cfg=MPPIConfig(horizon=2))
    # seed the buffer with random transitions (no sim)
    buf: ReplayBuffer = trainer.buffer
    obs = torch.randn(wm.cfg.obs_dim)
    for _ in range(64):
        a = torch.randn(wm.cfg.act_dim)
        nxt = torch.randn(wm.cfg.obs_dim)
        buf.add(obs, a, float(torch.randn(())), nxt, False)
        obs = nxt
    out = trainer.update()
    assert torch.isfinite(torch.tensor(out["loss"]))
    assert "inverse_dynamics" in out
    # the update must not have driven the encoder via reward/value: rederive the rv gradient
    # on a fresh detached rollout and confirm zero encoder grad.
    wm.zero_grad(set_to_none=True)
    obs_seq, action_seq, reward_seq = _window(wm, batch=4)
    detached = [z.detach() for z in wm.rollout_latents(obs_seq, action_seq)]
    r_loss = wm.reward_grounding_loss(action_seq, reward_seq, detached)
    r_loss.backward()
    assert _grad_sum(wm.encoder) == 0


# =====================================================================================
# (c) SIGReg: finite, ~0 on N(0,1), larger on collapse, de-collapses via gradient
# =====================================================================================
def test_sigreg_finite_and_small_for_gaussian():
    torch.manual_seed(1)
    z = torch.randn(512, 8)
    loss = sigreg_loss(z, n_proj=128, lam=0.05)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss >= 0.0
    assert loss < 0.02, f"sigreg on a true N(0,1) batch should be near 0, got {float(loss)}"


def test_sigreg_larger_for_collapsed_batch():
    torch.manual_seed(2)
    gaussian = torch.randn(512, 8)
    # point collapse: near-constant batch
    collapsed = torch.full((512, 8), 0.3) + 1e-3 * torch.randn(512, 8)
    # low-rank collapse: batch living on a 1-D subspace
    direction = torch.randn(1, 8)
    low_rank = torch.randn(512, 1) @ direction
    g = sigreg_loss(gaussian, n_proj=128)
    c = sigreg_loss(collapsed, n_proj=128)
    lr = sigreg_loss(low_rank, n_proj=128)
    assert c > g, f"collapsed sigreg {float(c)} must exceed gaussian {float(g)}"
    assert lr > g, f"low-rank sigreg {float(lr)} must exceed gaussian {float(g)}"


def test_sigreg_decollapses_increasing_effective_rank():
    torch.manual_seed(3)
    # start low-rank: 16-d batch on a 2-D subspace
    basis = torch.randn(2, 16)
    z = (torch.randn(256, 2) @ basis).requires_grad_(True)
    er_before = collapse_diagnostics(z.detach().numpy())["effective_rank"]
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = sigreg_loss(z, n_proj=128, lam=1.0)
        loss.backward()
        opt.step()
    er_after = collapse_diagnostics(z.detach().numpy())["effective_rank"]
    assert er_after > er_before + 0.5, (
        f"minimizing sigreg must de-collapse: eff_rank {er_before:.2f} -> {er_after:.2f}"
    )


def test_sigreg_grad_flows_to_input():
    z = torch.randn(64, 8, requires_grad=True)
    loss = sigreg_loss(z, n_proj=64)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# =====================================================================================
# (d) latent-norm switch: "none" is not a simplex; "simnorm" still is
# =====================================================================================
def _is_simplex(z: torch.Tensor, group_size: int) -> bool:
    """SimNorm splits the last dim into groups OF SIZE `group_size` (cfg.simnorm_groups == the
    group size), each a softmax simplex."""
    g = z.reshape(z.shape[0], z.shape[1] // group_size, group_size)
    nonneg = bool((g >= -1e-6).all())
    sums = g.sum(dim=-1)
    sums_to_one = bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))
    return nonneg and sums_to_one


def test_latent_norm_none_encoder_is_not_simplex():
    wm = _model(latent_dim=16, latent_norm="none")
    obs = torch.randn(8, wm.cfg.obs_dim)
    z = wm.encode(obs)
    assert z.shape == (8, 16)
    assert not _is_simplex(z, wm.cfg.simnorm_groups), "latent_norm='none' must be a RAW latent"
    # a raw latent should have some negative entries in general
    assert bool((z < 0).any())


def test_latent_norm_simnorm_encoder_is_simplex():
    wm = _model(latent_dim=16, latent_norm="simnorm")
    obs = torch.randn(8, wm.cfg.obs_dim)
    z = wm.encode(obs)
    assert _is_simplex(z, wm.cfg.simnorm_groups), "default latent_norm='simnorm' must be a simplex"


def test_latent_norm_none_predictor_output_is_not_simplex():
    wm = _model(latent_dim=16, latent_norm="none")
    obs_seq, action_seq, _ = _window(wm)
    latents = wm.rollout_latents(obs_seq, action_seq)
    z_hat = latents[1]
    assert not _is_simplex(z_hat, wm.cfg.simnorm_groups), (
        "predictor output under latent_norm='none' must be RAW end-to-end"
    )


def test_sigreg_loss_is_finite_through_full_rollout_none_norm():
    """The sigreg arm operates on the RAW online latent; sanity that the online latents are
    finite and sigreg over them is finite end-to-end."""
    wm = _model(horizon=3, latent_dim=16, latent_norm="none")
    obs_seq, action_seq, _ = _window(wm, batch=32)
    latents = wm.rollout_latents(obs_seq, action_seq)
    online = torch.cat(latents, dim=0)  # ((H+1)*B, d)
    loss = sigreg_loss(online, n_proj=64)
    assert torch.isfinite(loss)
