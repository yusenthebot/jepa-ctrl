from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Round-2's JEPA-MPPI planner implements this same interface."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class RandomController(Controller):
    """Uniform random within the action bounds. Seeded for cross-seed reproducibility."""

    def __init__(self, act_low: np.ndarray, act_high: np.ndarray, seed: int = 0) -> None:
        self.act_low = np.asarray(act_low, np.float32)
        self.act_high = np.asarray(act_high, np.float32)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, _obs: np.ndarray) -> np.ndarray:  # obs unused by design; round-2 planner uses it
        return self._rng.uniform(self.act_low, self.act_high).astype(np.float32)


class ZeroController(Controller):
    """No-op baseline: holds the zero action."""

    def __init__(self, act_dim: int, **_) -> None:  # **_ mirrors make_controller's uniform kwargs
        self.act_dim = int(act_dim)

    def reset(self) -> None:
        pass

    def act(self, _obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.act_dim, np.float32)


def make_controller(name: str, env, seed: int = 0) -> Controller:
    if name == "random":
        return RandomController(env.act_low, env.act_high, seed=seed)
    if name == "zero":
        return ZeroController(env.act_dim)
    raise ValueError(f"unknown controller {name!r} (have: random, zero)")
