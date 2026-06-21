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
from .pixels import ProceduralDistractor, composite_distractor, mask_background


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
        masked_target: bool = False,
    ) -> None:
        domain, t = parse_task(task)
        self.task = task
        self.action_repeat = int(action_repeat)
        self.size = int(size)
        self.frame_stack = int(frame_stack)
        self._cam = int(camera_id)
        self.distractor = bool(distractor)
        # R20: also maintain a parallel ROBOT-ONLY (background-zeroed) frame stack for the JEPA
        # masked consistency target. masked_obs() returns the stacked robot-only obs.
        self.masked_target = bool(masked_target)
        self._env = suite.load(domain, t, task_kwargs={"random": seed})

        aspec = self._env.action_spec()
        self.act_dim = int(np.prod(aspec.shape))
        self.act_low = np.broadcast_to(aspec.minimum, (self.act_dim,)).astype(np.float32).copy()
        self.act_high = np.broadcast_to(aspec.maximum, (self.act_dim,)).astype(np.float32).copy()

        self.obs_shape = (3 * self.frame_stack, self.size, self.size)
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._mframes: deque[np.ndarray] = deque(maxlen=self.frame_stack)  # robot-only (masked)
        self._step_count = 0
        self._last_rgb = np.zeros((self.size, self.size, 3), np.uint8)
        d_seed = seed if distractor_seed is None else int(distractor_seed)
        self._distractor = (
            ProceduralDistractor(self.size, self.size, seed=d_seed) if self.distractor else None
        )

    # --- rendering ----------------------------------------------------------------
    def _render_pair(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Render once; return (full, masked). `full` is the distractor-composited (or clean) RGB
        the agent sees; `masked` is the robot-only (background-zeroed) RGB for the JEPA target
        stream (None unless masked_target). A SINGLE clean render + seg feeds both, so they are
        pixel-aligned."""
        phys = self._env.physics
        clean = np.asarray(phys.render(height=self.size, width=self.size, camera_id=self._cam))
        seg = None
        if self._distractor is not None or self.masked_target:
            seg = phys.render(
                height=self.size, width=self.size, camera_id=self._cam, segmentation=True
            )
        if self._distractor is not None:
            full = composite_distractor(clean, seg, self._distractor.frame(self._step_count))
        else:
            full = clean
        masked = mask_background(clean, seg) if self.masked_target else None
        return np.asarray(full, np.uint8), (None if masked is None else np.asarray(masked, np.uint8))

    def _render_rgb(self) -> np.ndarray:
        """Back-compat: the full (possibly distracted) frame only."""
        return self._render_pair()[0]

    def _stack(self, frames: deque) -> np.ndarray:
        """Concatenate the k cached HxWx3 frames on the channel axis -> (3*k, size, size)."""
        chw = [f.transpose(2, 0, 1) for f in frames]
        return np.concatenate(chw, axis=0).astype(np.uint8)

    def _stacked_obs(self) -> np.ndarray:
        return self._stack(self._frames)

    def masked_obs(self) -> np.ndarray:
        """The stacked ROBOT-ONLY (background-zeroed) obs, (3*k,size,size) uint8. Requires
        masked_target=True. Mirrors the full obs stack so the buffer stores aligned pairs."""
        if not self.masked_target:
            raise RuntimeError("masked_obs() requires masked_target=True")
        return self._stack(self._mframes)

    def _push_frame(self, full: np.ndarray, masked: np.ndarray | None = None) -> None:
        self._last_rgb = full
        self._frames.append(full)
        if self.masked_target:
            self._mframes.append(masked if masked is not None else np.zeros_like(full))

    # --- gym-ish API --------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._env.reset()
        self._step_count = 0
        full, masked = self._render_pair()
        self._frames.clear()
        self._mframes.clear()
        for _ in range(self.frame_stack):  # fill both stacks with the first frame
            self._frames.append(full)
            if self.masked_target:
                self._mframes.append(masked if masked is not None else np.zeros_like(full))
        self._last_rgb = full
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
        full, masked = self._render_pair()
        self._push_frame(full, masked)
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
