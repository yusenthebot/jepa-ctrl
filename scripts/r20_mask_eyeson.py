#!/usr/bin/env python
"""R20 Step-2 eyes-on de-risk: BEFORE the ~24h masked-target campaign, visually confirm the
segmentation mask is clean enough to separate the JEPA streams. Renders cheetah-run at 64x64 and
saves a contact sheet of [full+distractor | clean robot | masked robot-only] for several frames.

KILL signals to look for (the judge's pre-registered risks):
  (1) clean-baseline killer: robot pixels — ESPECIALLY the foot/ground-contact seam — getting zeroed
      in the masked column (classified as background). If the feet vanish, the contact cue cheetah
      needs is gone and the clean baseline will regress => experiment dead.
  (2) no-invariance killer: background distractor pixels leaking THROUGH the mask into the masked
      column (robot/bg bleed at edges). If the masked column still shows distractor clutter, the
      target stream carries distractor info and there is no invariance pressure.
Usage: MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python scripts/r20_mask_eyeson.py [size]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite

from jepa_ctrl.pixels import ProceduralDistractor, composite_distractor, mask_background

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
OUT = "runs/R20_mask_eyeson.png"
N_FRAMES = 5

env = suite.load("cheetah", "run", task_kwargs={"random": 0})
env.reset()
distractor = ProceduralDistractor(SIZE, SIZE, seed=0)

rows = []
rng = np.random.default_rng(0)
aspec = env.action_spec()
for t in range(N_FRAMES):
    for _ in range(8):  # advance the gait so frames differ
        a = rng.uniform(aspec.minimum, aspec.maximum).astype(np.float32)
        ts = env.step(a)
        if ts.last():
            env.reset()
    phys = env.physics
    clean = np.asarray(phys.render(height=SIZE, width=SIZE, camera_id=0), np.uint8)
    seg = phys.render(height=SIZE, width=SIZE, camera_id=0, segmentation=True)
    full = composite_distractor(clean, seg, distractor.frame(t * 8))
    masked = mask_background(clean, seg)
    # robot pixel fraction (sanity: feet present?)
    obj = seg[..., 0] if seg.ndim == 3 else seg
    frac = float((obj != -1).mean())
    rows.append((full, clean, masked, frac))

fig, axes = plt.subplots(N_FRAMES, 3, figsize=(7.5, 2.4 * N_FRAMES))
cols = ["online: full + distractor", "clean robot render", "TARGET: masked robot-only"]
for r, (full, clean, masked, frac) in enumerate(rows):
    for c, img in enumerate((full, clean, masked)):
        ax = axes[r, c] if N_FRAMES > 1 else axes[c]
        ax.imshow(img)
        ax.axis("off")
        if r == 0:
            ax.set_title(cols[c], fontsize=9)
    (axes[r, 2] if N_FRAMES > 1 else axes[2]).set_title(
        f"robot px {frac:.1%}", fontsize=8
    )
fig.suptitle(f"R20 masked-target eyes-on (cheetah-run {SIZE}x{SIZE})", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
fracs = [r[3] for r in rows]
print(f"saved -> {OUT}  robot_px_frac mean={np.mean(fracs):.3f} min={np.min(fracs):.3f}")
env.close()
