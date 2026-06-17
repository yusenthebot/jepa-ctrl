from __future__ import annotations

import numpy as np
import pytest

from jepa_ctrl.envs import DMCEnv, parse_task


def test_parse_task():
    assert parse_task("cheetah-run") == ("cheetah", "run")
    assert parse_task("ball_in_cup-catch") == ("ball_in_cup", "catch")
    with pytest.raises(ValueError):
        parse_task("nohyphen")


def test_dmc_env_shapes_and_step():
    # cartpole-swingup is tiny + fast; obs/act dims derived at runtime, never hardcoded.
    env = DMCEnv("cartpole-swingup", seed=0, action_repeat=2)
    try:
        assert env.act_dim == 1
        assert env.obs_dim > 0
        assert env.act_low.shape == (1,) and env.act_high.shape == (1,)

        obs = env.reset()
        assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32

        obs2, r, done = env.step(np.zeros(env.act_dim, np.float32))
        assert obs2.shape == (env.obs_dim,)
        assert isinstance(r, float)
        assert isinstance(done, bool)

        # action clipping: huge action is clipped into bounds (no error)
        env.step(np.full(env.act_dim, 1e6, np.float32))
    finally:
        env.close()
