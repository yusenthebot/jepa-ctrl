from __future__ import annotations

import gc
import os
from collections import deque

# Offscreen GPU rendering on the laptop (no X display needed). Must be set before dm_control
# triggers any GL context. EGL uses the RTX 5080 directly. Mirrors envs.py.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from dm_control import suite

from .envs import parse_task
from .pixels import ProceduralDistractor, composite_distractor


class PixelDMCEnv:
    """Pixel dm_control wrapper: obs = a frame stack of k rendered RGB frames at size x size.

    obs is a uint8 array of shape (3*k, size, size) = (9, 84, 84) for the default k=3, 84px.
    With distractor=True the rendered background (segmentation object id == -1) is replaced each
    step by a temporally-coherent ProceduralDistractor frame, so the robot pixels are real and
    the BACKGROUND is time-varying clutter — the distractor-robustness head-to-head setup. The
    pure compositing + distractor pieces live in pixels.py and are unit-tested sim-free; this
    class only wires them to the live sim (NOT run in unit tests — orchestrator smokes it live).

    act_dim / bounds are derived from the action spec at runtime (never hardcoded). The caller
    owns the env lifecycle (close()).
    """

    def __init__(
        self,
        task: str,
        seed: int,
        action_repeat: int = 2,
        size: int = 84,
        frame_stack: int = 3,
        camera_id: int = 0,
        distractor: bool = False,
        distractor_seed: int | None = None,
    ) -> None:
        domain, t = parse_task(task)
        self.task = task
        self.action_repeat = int(action_repeat)
        self.size = int(size)
        self.frame_stack = int(frame_stack)
        self._cam = int(camera_id)
        self.distractor = bool(distractor)
        self._env = suite.load(domain, t, task_kwargs={"random": seed})

        aspec = self._env.action_spec()
        self.act_dim = int(np.prod(aspec.shape))
        self.act_low = np.broadcast_to(aspec.minimum, (self.act_dim,)).astype(np.float32).copy()
        self.act_high = np.broadcast_to(aspec.maximum, (self.act_dim,)).astype(np.float32).copy()

        self.obs_shape = (3 * self.frame_stack, self.size, self.size)
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._step_count = 0
        self._last_rgb = np.zeros((self.size, self.size, 3), np.uint8)
        d_seed = seed if distractor_seed is None else int(distractor_seed)
        self._distractor = (
            ProceduralDistractor(self.size, self.size, seed=d_seed) if self.distractor else None
        )

    # --- rendering ----------------------------------------------------------------
    def _render_rgb(self) -> np.ndarray:
        """Render one (size, size, 3) uint8 RGB frame, distractor-composited if enabled."""
        phys = self._env.physics
        rgb = phys.render(height=self.size, width=self.size, camera_id=self._cam)
        if self._distractor is not None:
            seg = phys.render(
                height=self.size, width=self.size, camera_id=self._cam, segmentation=True
            )
            distractor_rgb = self._distractor.frame(self._step_count)
            rgb = composite_distractor(rgb, seg, distractor_rgb)
        return np.asarray(rgb, np.uint8)

    def _stacked_obs(self) -> np.ndarray:
        """Concatenate the k cached HxWx3 frames on the channel axis -> (3*k, size, size)."""
        # each frame HWC -> CHW, then stack on channel axis
        chw = [f.transpose(2, 0, 1) for f in self._frames]
        return np.concatenate(chw, axis=0).astype(np.uint8)

    def _push_frame(self, rgb: np.ndarray) -> None:
        self._last_rgb = rgb
        self._frames.append(rgb)

    # --- gym-ish API --------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._env.reset()
        self._step_count = 0
        rgb = self._render_rgb()
        self._frames.clear()
        for _ in range(self.frame_stack):  # fill the stack with the first frame
            self._frames.append(rgb)
        self._last_rgb = rgb
        return self._stacked_obs()

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
        self._step_count += 1
        self._push_frame(self._render_rgb())
        return self._stacked_obs(), total_r, bool(ts.last())

    def render(self) -> np.ndarray:
        """The most recent (possibly distracted) RGB frame, HxWx3 uint8 — for video / verify."""
        return self._last_rgb

    def control_timestep(self) -> float:
        return float(self._env.control_timestep())

    def close(self) -> None:
        try:
            del self._env
        finally:
            gc.collect()
