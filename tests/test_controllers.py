from __future__ import annotations

import numpy as np
import pytest

from jepa_ctrl.controllers import RandomController, ZeroController, make_controller


def test_random_controller_within_bounds_and_seeded():
    low = np.array([-1.0, -0.5], np.float32)
    high = np.array([1.0, 0.5], np.float32)
    c1 = RandomController(low, high, seed=7)
    c2 = RandomController(low, high, seed=7)
    for _ in range(50):
        a = c1.act(np.zeros(4))
        assert np.all(a >= low) and np.all(a <= high)
        assert np.allclose(a, c2.act(np.zeros(4)))  # same seed -> same stream


def test_zero_controller():
    c = ZeroController(act_dim=3)
    assert np.allclose(c.act(np.ones(5)), np.zeros(3))


def test_make_controller_unknown():
    class _Env:
        act_low = np.zeros(2, np.float32)
        act_high = np.ones(2, np.float32)
        act_dim = 2

    assert isinstance(make_controller("random", _Env(), seed=0), RandomController)
    assert isinstance(make_controller("zero", _Env(), seed=0), ZeroController)
    with pytest.raises(ValueError):
        make_controller("bogus", _Env())
