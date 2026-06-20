"""R17 reset-curriculum (cover-to-recover) probe — pure-python/torch unit tests, NO sim.

Covers the bank + Trainer wiring with synthetic state vectors and a tiny fake env. The LIVE
dm_control state roundtrip (get_state -> reset(from_state=...) reproduces the obs) is documented
here by test_state_roundtrip_DOC but NOT executed — the orchestrator runs it on a real env
(one sim at a time, 64GB host)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.trainer import NearFallBank, TrainConfig, Trainer

# --- NearFallBank -----------------------------------------------------------------


def test_bank_only_keeps_states_from_low_return_episodes():
    """The Trainer gates banking on the return threshold; states from a HIGH-return episode
    must never enter the bank, so only near-fall coverage accumulates."""
    wm = WorldModel(ModelConfig(obs_dim=5, act_dim=1, latent_dim=32))
    cfg = TrainConfig(reset_curriculum=True, bank_return_thresh=150.0)
    tr = Trainer(wm, cfg, act_low=torch.full((1,), -1.0), act_high=torch.full((1,), 1.0))

    # low-return episode (sum < 150): banked
    for r in [10.0, 20.0, 30.0]:  # return 60 < 150
        tr.note_step(np.full(7, 1.0), r, done=False)
    tr.note_step(np.full(7, 1.0), 0.0, done=True)
    assert len(tr.bank) == 4  # all 4 transition states harvested

    # high-return episode (sum >= 150): NOT banked
    for r in [100.0, 100.0]:  # return 200 >= 150
        tr.note_step(np.full(7, 2.0), r, done=False)
    tr.note_step(np.full(7, 2.0), 0.0, done=True)
    assert len(tr.bank) == 4  # unchanged — high-return states rejected


def test_bank_sample_returns_banked_state_plus_bounded_noise():
    """sample() = a stored state + zero-mean Gaussian pose noise; with small noise it stays
    close to the base, and repeated draws differ (noise is fresh per call)."""
    bank = NearFallBank(capacity=10, pose_noise=0.02, rng_seed=1)
    base = np.array([1.0, -2.0, 3.0, 0.5], np.float64)
    bank.add_episode([base], ep_return=0.0)

    draws = np.stack([bank.sample() for _ in range(200)])
    # each draw is within a few sigma of the base in every dim
    assert np.all(np.abs(draws - base) < 0.02 * 8)
    # the empirical std per-dim is ~pose_noise (bounded, non-degenerate)
    assert np.all(draws.std(axis=0) < 0.02 * 3)
    assert np.all(draws.std(axis=0) > 0.02 * 0.3)
    # two consecutive draws are not identical (fresh noise)
    assert not np.allclose(bank.sample(), bank.sample())


def test_bank_sample_empty_raises():
    bank = NearFallBank(capacity=4, pose_noise=0.01, rng_seed=0)
    with pytest.raises(IndexError):
        bank.sample()


def test_bank_capacity_is_a_reservoir():
    """A full bank stays at capacity; reservoir replacement keeps the size fixed."""
    bank = NearFallBank(capacity=3, pose_noise=0.0, rng_seed=2)
    bank.add_episode([np.full(2, float(i)) for i in range(50)], ep_return=0.0)
    assert len(bank) == 3


# --- Trainer reset-curriculum wiring ----------------------------------------------


class _FakeEnv:
    """Minimal DMCEnv-protocol stand-in. Records whether each reset used from_state, so we can
    measure the bank-draw fraction without a simulator."""

    def __init__(self, obs_dim: int = 5, state_dim: int = 7) -> None:
        self.obs_dim = obs_dim
        self._state_dim = state_dim
        self.from_state_calls = 0
        self.plain_calls = 0

    def get_state(self) -> np.ndarray:
        return np.zeros(self._state_dim, np.float64)

    def reset(self, from_state: np.ndarray | None = None) -> np.ndarray:
        if from_state is None:
            self.plain_calls += 1
        else:
            self.from_state_calls += 1
        return np.zeros(self.obs_dim, np.float32)


def _trainer(cfg: TrainConfig) -> Trainer:
    wm = WorldModel(ModelConfig(obs_dim=5, act_dim=1, latent_dim=32))
    return Trainer(wm, cfg, act_low=torch.full((1,), -1.0), act_high=torch.full((1,), 1.0))


def test_reset_curriculum_off_is_a_noop_bank_never_used():
    """Default reset_curriculum=False -> reset_env() is always a plain reset, note_step never
    banks, and the bank stays empty regardless of returns."""
    tr = _trainer(TrainConfig(reset_curriculum=False, bank_return_thresh=150.0))
    env = _FakeEnv()
    for _ in range(50):
        tr.reset_env(env)
        tr.note_step(np.zeros(7), 1.0, done=True)  # low return, but curriculum OFF
    assert len(tr.bank) == 0
    assert env.from_state_calls == 0
    assert env.plain_calls == 50


def test_reset_p_fraction_of_resets_draw_from_bank():
    """With the curriculum on and a non-empty bank, the bank-draw fraction over many resets
    matches reset_p (seeded rng -> statistically tight)."""
    cfg = TrainConfig(reset_curriculum=True, reset_p=0.3, bank_return_thresh=150.0)
    tr = _trainer(cfg)
    # pre-seed the bank so draws are possible from reset #1
    tr.bank.add_episode([np.zeros(7) for _ in range(20)], ep_return=0.0)
    env = _FakeEnv()
    n = 4000
    for _ in range(n):
        tr.reset_env(env)
    frac = env.from_state_calls / n
    assert abs(frac - 0.3) < 0.03  # within 3pp of reset_p over 4000 draws
    assert env.from_state_calls + env.plain_calls == n


def test_curriculum_resets_empty_bank_falls_back_to_plain():
    """reset_curriculum on but the bank empty -> every reset is a plain reset (no draw is
    attempted), so the very first episodes collect normally before any near-fall coverage."""
    cfg = TrainConfig(reset_curriculum=True, reset_p=1.0, bank_return_thresh=150.0)
    tr = _trainer(cfg)
    env = _FakeEnv()
    for _ in range(10):
        tr.reset_env(env)
    assert env.from_state_calls == 0
    assert env.plain_calls == 10


def test_collect_step_snapshots_state_and_banks_on_low_return_done():
    """collect_step() snapshots env.get_state() pre-step and feeds note_step; a low-return
    episode ending in collect_step banks its states (end-to-end through the public collect path,
    still no sim — the fake env supplies synthetic states/obs)."""

    class _StepEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self._t = 0

        def step(self, action):  # noqa: ANN001
            self._t += 1
            done = self._t >= 3
            return np.zeros(self.obs_dim, np.float32), 1.0, done

    cfg = TrainConfig(reset_curriculum=True, bank_return_thresh=150.0, seed_steps=0)
    tr = _trainer(cfg)
    env = _StepEnv()
    obs = tr.reset_env(env)
    done = False
    while not done:
        obs, _r, done = tr.collect_step(env, obs)
    assert len(tr.bank) == 3  # 3 pre-step state snapshots harvested on the low-return episode


# --- LIVE roundtrip documentation (NOT executed here) -----------------------------


@pytest.mark.skip(reason="live dm_control roundtrip — orchestrator runs this on a real env")
def test_state_roundtrip_DOC():
    """DOCUMENTS the intended live assertion the orchestrator verifies on a REAL DMCEnv.

    The roundtrip contract for jepa_ctrl.envs.DMCEnv:

        env = DMCEnv("quadruped-walk", seed=0)
        env.reset()
        env.step(some_action)                 # advance to a non-trivial physical state
        saved = env.get_state()               # full physics state (qpos+qvel+act+...)
        obs_a = env.reset(from_state=saved)    # set_state + after_reset, obs from the task
        env.reset()                            # perturb to a DIFFERENT state
        env.set_state(saved)                   # restore exactly
        obs_b = env.reset(from_state=saved)    # rebuild obs at the restored state

    Expected (the orchestrator asserts on the live env):
        np.allclose(env.get_state(), saved)   after set_state(saved)
        np.allclose(obs_a, obs_b)             reset(from_state=saved) is reproducible
        obs from reset(from_state=saved) == the obs the encoder sees at `saved`
          i.e. task.get_observation(physics) flattened in the SAME _flat_obs key order.

    Here we only assert shapes on synthetic data so the test file is import-clean without a
    simulator; the behavioural assertion is the skip docstring above.
    """
    saved = np.zeros(20, np.float64)
    assert saved.shape == (20,)
