from __future__ import annotations

import gc
import os

# Offscreen GPU rendering on the laptop (no X display needed). Must be set before
# dm_control triggers any GL context. EGL uses the RTX 5080 directly.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from dm_control import suite


def parse_task(task: str) -> tuple[str, str]:
    """'cheetah-run' -> ('cheetah', 'run'); 'ball_in_cup-catch' -> ('ball_in_cup', 'catch')."""
    domain, _, t = task.partition("-")
    if not t:
        raise ValueError(f"task must be 'domain-task', got {task!r}")
    return domain, t


class DMCEnv:
    """Thin dm_control suite wrapper: flat float32 obs, action_repeat, offscreen render.

    obs_dim / act_dim are derived from the specs at runtime (never hardcoded) — the
    finger-spin / EnvPool obs-dim disagreement is avoided by construction.
    """

    def __init__(
        self,
        task: str,
        seed: int,
        action_repeat: int = 2,
        render_height: int = 240,
        render_width: int = 320,
        camera_id: int = 0,
    ) -> None:
        domain, t = parse_task(task)
        self.task = task
        self.action_repeat = int(action_repeat)
        self._rh, self._rw, self._cam = render_height, render_width, camera_id
        self._env = suite.load(domain, t, task_kwargs={"random": seed})

        spec = self._env.observation_spec()
        self._obs_keys = list(spec.keys())
        self.obs_dim = int(sum(int(np.prod(spec[k].shape)) for k in self._obs_keys))

        aspec = self._env.action_spec()
        self.act_dim = int(np.prod(aspec.shape))
        self.act_low = np.broadcast_to(aspec.minimum, (self.act_dim,)).astype(np.float32).copy()
        self.act_high = np.broadcast_to(aspec.maximum, (self.act_dim,)).astype(np.float32).copy()

    def _flat_obs(self, obs) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs[k], np.float32).ravel() for k in self._obs_keys]
        )

    def reset(self) -> np.ndarray:
        return self._flat_obs(self._env.reset().observation)

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        action = np.clip(
            np.asarray(action, np.float32).reshape(self.act_dim), self.act_low, self.act_high
        )
        total_r = 0.0
        ts = None
        for _ in range(self.action_repeat):
            ts = self._env.step(action)
            total_r += float(ts.reward or 0.0)
            if ts.last():
                break
        return self._flat_obs(ts.observation), total_r, bool(ts.last())

    def render(self) -> np.ndarray:
        return self._env.physics.render(height=self._rh, width=self._rw, camera_id=self._cam)

    def control_timestep(self) -> float:
        return float(self._env.control_timestep())

    def close(self) -> None:
        try:
            del self._env
        finally:
            gc.collect()
