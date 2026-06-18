from __future__ import annotations

import numpy as np

# dm_control's segmentation render encodes "no geom" (sky / background) as object id -1.
BACKGROUND_ID = -1


def composite_distractor(
    rgb: np.ndarray, seg: np.ndarray, distractor_rgb: np.ndarray
) -> np.ndarray:
    """Replace background pixels with the distractor frame; keep robot pixels untouched.

    `rgb` and `distractor_rgb` are (H, W, 3) uint8. `seg` is the dm_control segmentation
    map: either (H, W) of object ids or (H, W, C) where channel 0 is the object id (the
    raw `physics.render(..., segmentation=True)` output is (H, W, 2): [object_id, geom_type]).
    A pixel is background iff its object id == BACKGROUND_ID (-1); those pixels take the
    distractor color, everything else keeps the rendered robot. Pure / no sim dependency.
    Returns a NEW (H, W, 3) uint8 array (inputs are never mutated).
    """
    rgb = np.asarray(rgb)
    distractor_rgb = np.asarray(distractor_rgb)
    seg = np.asarray(seg)
    if rgb.shape != distractor_rgb.shape or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            f"rgb and distractor_rgb must match as (H,W,3); got {rgb.shape} vs "
            f"{distractor_rgb.shape}"
        )
    obj_id = seg[..., 0] if seg.ndim == 3 else seg
    if obj_id.shape != rgb.shape[:2]:
        raise ValueError(f"seg {obj_id.shape} does not match rgb spatial {rgb.shape[:2]}")
    is_bg = obj_id == BACKGROUND_ID  # (H, W) bool
    out = rgb.astype(np.uint8, copy=True)
    out[is_bg] = distractor_rgb.astype(np.uint8, copy=False)[is_bg]
    return out


class ProceduralDistractor:
    """Deterministic, temporally-coherent procedural background video generator.

    `.frame(t)` returns an (H, W, 3) uint8 frame of a few smoothly drifting sinusoidal color
    fields. Coherent in time: phase advances continuously with t, so consecutive frames are
    highly correlated (not iid noise) — exactly the kind of structured time-varying clutter a
    reconstruction world model wastes capacity on while a JEPA latent model can ignore. No
    download, fully deterministic per (h, w, seed). Pure numpy, no torch / sim dependency.
    """

    def __init__(self, h: int, w: int, seed: int = 0, n_waves: int = 3) -> None:
        if h <= 0 or w <= 0:
            raise ValueError(f"h, w must be positive; got {h}x{w}")
        self.h = int(h)
        self.w = int(w)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # per-wave: spatial frequencies (cycles over the frame), drift speed, phase, RGB tint.
        self._n = int(n_waves)
        self._fx = rng.uniform(0.5, 2.5, self._n)
        self._fy = rng.uniform(0.5, 2.5, self._n)
        self._speed = rng.uniform(0.04, 0.16, self._n)  # radians of phase per frame step
        self._phase = rng.uniform(0.0, 2.0 * np.pi, self._n)
        self._tint = rng.uniform(0.3, 1.0, (self._n, 3))
        ys = np.linspace(0.0, 2.0 * np.pi, self.h, endpoint=False)
        xs = np.linspace(0.0, 2.0 * np.pi, self.w, endpoint=False)
        self._gy, self._gx = np.meshgrid(ys, xs, indexing="ij")  # (H, W) each

    def frame(self, t: int) -> np.ndarray:
        """(H, W, 3) uint8 background frame at integer time step `t` (phase = speed * t)."""
        acc = np.zeros((self.h, self.w, 3), dtype=np.float64)
        for i in range(self._n):
            phase = self._phase[i] + self._speed[i] * float(t)
            field = np.sin(self._fx[i] * self._gx + self._fy[i] * self._gy + phase)  # [-1,1]
            field = 0.5 * (field + 1.0)  # -> [0,1]
            acc += field[..., None] * self._tint[i][None, None, :]
        acc /= max(1, self._n)  # mean over waves -> [0,1]
        return np.clip(acc * 255.0, 0, 255).astype(np.uint8)
