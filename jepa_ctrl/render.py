from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_mp4(frames, path, fps: float = 30) -> str:
    """Write a list of HxWx3 uint8 frames to an mp4 (libx264 via imageio-ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        [np.asarray(f, np.uint8) for f in frames],
        fps=max(5, int(round(fps))),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    return str(path)


def contact_sheet(frames, path, nrow: int = 3, ncol: int = 4, title=None, step_returns=None) -> str:
    """Evenly-spaced keyframe grid (PNG) annotated with step index + cumulative return.

    This is the human-readable REAL-VERIFY surface: a single image the orchestrator
    Reads back to confirm the sim is actually being driven.
    """
    if len(frames) == 0:
        raise ValueError("no frames to render")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = nrow * ncol
    idx = np.linspace(0, len(frames) - 1, num=min(n, len(frames))).astype(int)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.2))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for j, i in enumerate(idx):
        ax = axes[j]
        ax.imshow(frames[i])
        lbl = f"t={i}"
        if step_returns is not None and i < len(step_returns):
            lbl += f"  R={step_returns[i]:.0f}"
        ax.set_title(lbl, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(path), dpi=90)
    plt.close(fig)
    return str(path)
