from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EvalConfig
from .controllers import Controller, make_controller
from .envs import DMCEnv
from .metrics import aggregate_returns
from .render import contact_sheet, save_mp4
from .seeding import set_seed


def evaluate_episode(
    env: DMCEnv,
    controller: Controller,
    render: bool = False,
    max_env_steps: int = 2000,
) -> tuple[float, list[np.ndarray], list[float], bool]:
    """Run one episode. Returns (total_return, frames, cumulative_returns, truncated).

    `truncated` is True iff the safety cap fired before the env signalled done — a
    cap-hit episode is NOT a real terminal and must be distinguishable from one.
    """
    controller.reset()
    obs = env.reset()
    total = 0.0
    steps = 0
    frames: list[np.ndarray] = []
    cum: list[float] = []
    if render:
        frames.append(env.render())
        cum.append(0.0)
    done = False
    while not done and steps < max_env_steps:
        action = controller.act(obs)
        obs, reward, done = env.step(action)
        total += reward
        steps += 1
        if render:
            frames.append(env.render())
            cum.append(total)
    truncated = (not done) and steps >= max_env_steps
    return total, frames, cum, truncated


def evaluate_controller(cfg: EvalConfig) -> dict:
    """Cross-seed evaluation. Renders the first episode of EVERY seed (no cherry-picking)."""
    per_seed_means: list[float] = []
    renders: list[dict] = []
    any_truncated = False

    for seed in cfg.seeds:
        set_seed(seed)
        env = DMCEnv(
            cfg.task,
            seed=seed,
            action_repeat=cfg.action_repeat,
            render_height=cfg.render_height,
            render_width=cfg.render_width,
            camera_id=cfg.camera_id,
        )
        controller = make_controller(cfg.controller, env, seed=seed)

        ep_returns: list[float] = []
        for ep in range(cfg.episodes):
            do_render = cfg.render and ep == 0
            total, frames, cum, truncated = evaluate_episode(
                env, controller, render=do_render, max_env_steps=cfg.max_env_steps
            )
            ep_returns.append(total)
            any_truncated = any_truncated or truncated
            if do_render and frames:
                outdir = Path(cfg.outdir) / cfg.task / f"seed{seed}"
                dt = env.control_timestep() * cfg.action_repeat
                fps = max(10, round(1.0 / dt)) if dt > 0 else 30
                mp4 = save_mp4(frames, outdir / "rollout.mp4", fps=fps)
                sheet = contact_sheet(
                    frames,
                    outdir / "contact_sheet.png",
                    title=f"{cfg.task} | {cfg.controller} | seed {seed} | ep0 return {total:.1f}",
                    step_returns=cum,
                )
                renders.append(
                    {"seed": seed, "mp4": mp4, "contact_sheet": sheet, "fps": fps,
                     "n_frames": len(frames), "ep0_return": total, "truncated": truncated}
                )
        per_seed_means.append(float(np.mean(ep_returns)))
        env.close()  # free MuJoCo every seed — no instance leak over a long loop

    agg = aggregate_returns(per_seed_means)
    result = {
        "task": cfg.task,
        "controller": cfg.controller,
        "episodes": cfg.episodes,
        "action_repeat": cfg.action_repeat,
        "seeds": list(cfg.seeds),
        "aggregate": agg,
        "any_truncated": any_truncated,
        "renders": renders,
    }
    outdir = Path(cfg.outdir) / cfg.task
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{cfg.controller}_result.json").write_text(json.dumps(result, indent=2))
    return result
