from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """Immutable evaluation config. dm_control task is 'domain-task' (e.g. 'cheetah-run')."""

    task: str = "cheetah-run"
    controller: str = "random"
    seeds: tuple[int, ...] = (0, 1, 2)
    episodes: int = 3
    action_repeat: int = 2
    render: bool = True
    render_height: int = 240
    render_width: int = 320
    camera_id: int = 0
    max_env_steps: int = 2000  # safety cap; dm_control enforces the real time_limit (1000)
    outdir: str = "runs"
