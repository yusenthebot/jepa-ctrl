from __future__ import annotations

import numpy as np
import torch
from torch import nn

from jepa_ctrl.model import (
    JepaController,
    ModelConfig,
    MPPIPlanner,
    ReplayBuffer,
    WorldModel,
    eval_mppi,
    train_mppi,
)

torch.manual_seed(0)


def _tiny_model(obs_dim: int = 6, act_dim: int = 2, latent_dim: int = 16) -> WorldModel:
    cfg = ModelConfig(obs_dim=obs_dim, act_dim=act_dim, latent_dim=latent_dim, num_q=3)
    return WorldModel(cfg).eval()


# --- (1) MPPI returns a correctly-shaped, in-bounds, deterministic action ----------
def test_mppi_action_shape_bounds_and_deterministic():
    model = _tiny_model()
    low = -torch.ones(model.cfg.act_dim)
    high = torch.ones(model.cfg.act_dim)
    obs = torch.randn(model.cfg.obs_dim)

    def run() -> torch.Tensor:
        torch.manual_seed(123)
        planner = MPPIPlanner(model, train_mppi(), low, high)
        return planner.plan(obs)

    a1 = run()
    a2 = run()
    assert a1.shape == (model.cfg.act_dim,)
    assert torch.all(a1 >= -1.0) and torch.all(a1 <= 1.0)
    assert torch.allclose(a1, a2), "MPPI must be deterministic under a fixed torch seed"


# --- (2) sanity-optimization: a reward that prefers action ~ +1 pulls MPPI positive ---
class _PlusOneReward(nn.Module):
    """Hand-crafted reward head: scalar reward = -sum((a-1)^2), maximised at a=+1.

    Mimics the DistHead interface used by MPPIPlanner._score: logits(z,a) carries the
    action through (stacked over a dummy ensemble dim), to_scalar maps it to the reward.
    """

    def logits(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return action.unsqueeze(0)  # (1, N, act_dim) — the ensemble dim _score indexes [0]

    def to_scalar(self, a: torch.Tensor) -> torch.Tensor:
        return -((a - 1.0) ** 2).sum(-1)


class _ZeroValue(nn.Module):
    """Value head stub: contributes nothing to the score (isolates the reward signal)."""

    def logits(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.zeros(3, action.shape[0], 1)  # (num_q, N, 1)

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.squeeze(-1)  # (num_q, N) all zeros


def test_mppi_optimizes_toward_rewarded_action():
    model = _tiny_model(act_dim=2)
    model.reward_head = _PlusOneReward()
    model.value_head = _ZeroValue()
    low = -torch.ones(2)
    high = torch.ones(2)
    obs = torch.zeros(model.cfg.obs_dim)

    torch.manual_seed(7)
    planner = MPPIPlanner(model, eval_mppi(), low, high)
    action = planner.plan(obs)
    # reward is maximised at +1; the optimiser must trend the returned action positive,
    # decisively better than the zero-mean prior it started from.
    assert action.mean() > 0.5, f"MPPI did not optimise the reward (action={action})"
    assert torch.all(action > 0.0)


# --- (3) ReplayBuffer add / sample / sub-trajectory shapes -------------------------
def test_replay_buffer_shapes():
    obs_dim, act_dim, cap = 6, 2, 50
    buf = ReplayBuffer(cap, obs_dim, act_dim)
    horizon = 3
    # fill one clean episode of 20 transitions, done only on the last
    for t in range(20):
        buf.add(
            torch.randn(obs_dim),
            torch.randn(act_dim),
            float(t),
            torch.randn(obs_dim),
            done=(t == 19),
        )
    assert len(buf) == 20

    single = buf.sample(8)
    assert single["obs"].shape == (8, obs_dim)
    assert single["action"].shape == (8, act_dim)
    assert single["reward"].shape == (8,)
    assert single["next_obs"].shape == (8, obs_dim)

    traj = buf.sample_subtraj(batch=5, length=horizon)
    assert traj["obs_seq"].shape == (horizon + 1, 5, obs_dim)
    assert traj["action_seq"].shape == (horizon, 5, act_dim)
    assert traj["reward"].shape == (horizon, 5)


def test_replay_buffer_subtraj_respects_episode_boundary():
    obs_dim, act_dim = 4, 1
    buf = ReplayBuffer(200, obs_dim, act_dim)
    # episodes of length 5; done every 5th step. A clean window must not contain an
    # interior done (which would splice two episodes).
    for t in range(100):
        buf.add(
            torch.full((obs_dim,), float(t)),
            torch.zeros(act_dim),
            0.0,
            torch.full((obs_dim,), float(t + 1)),
            done=((t + 1) % 5 == 0),
        )
    traj = buf.sample_subtraj(batch=32, length=3)
    # the first `length-1` steps of each window are guaranteed done-free; verify obs are
    # contiguous (o increments by 1) within those steps.
    obs_seq = traj["obs_seq"]  # (4, 32, obs_dim)
    diffs = obs_seq[1:3, :, 0] - obs_seq[0:2, :, 0]
    assert torch.allclose(diffs, torch.ones_like(diffs)), "windows are not contiguous"


# --- (4) JepaController.act: np.float32 within bounds, no env ----------------------
def test_jepa_controller_act_in_bounds():
    model = _tiny_model(obs_dim=8, act_dim=3)
    low = np.array([-1.0, -0.5, -2.0], np.float32)
    high = np.array([1.0, 0.5, 2.0], np.float32)
    ctrl = JepaController(model, low, high)
    ctrl.reset()

    obs = np.random.randn(8).astype(np.float32)
    a = ctrl.act(obs)
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float32
    assert a.shape == (3,)
    assert np.all(a >= low) and np.all(a <= high)
